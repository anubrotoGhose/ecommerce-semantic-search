import os
import re
import uuid
import json
import logging
from typing import List
from fastapi import FastAPI, UploadFile, BackgroundTasks
from dotenv import load_dotenv
from tqdm import tqdm

import duckdb
import mysql.connector
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)

from openai import AzureOpenAI

# =========================================================
# ENV + LOGGING
# =========================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =========================================================
# CONFIG
# =========================================================

MYSQL_CONFIG = {
    "host": os.getenv("mysql_db_host", "localhost"),
    "port": int(os.getenv("mysql_db_port", 3306)),
    "user": os.getenv("mysql_db_user"),
    "password": os.getenv("mysql_db_password"),
    "database": os.getenv("mysql_db_name"),
    "autocommit": False
}

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = "amazon_fashion"

EMBEDDING_DIM = 1536

azure_client = AzureOpenAI(
    api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT")
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_AZURE_OPENAI_MODEL")

# =========================================================
# HELPERS
# =========================================================

def safe_join(val):
    if isinstance(val, list):
        return " ".join(map(str, val))
    if isinstance(val, dict):
        return " ".join(map(str, val.values()))
    return val or ""

def extract_image(row: dict) -> str:
    return (
        row.get("image_large")
        or row.get("image_hi_res")
        or row.get("image_url")
        or row.get("image_thumb")
        or ""
    )

def build_bag_of_words(texts: List[str]) -> str:
    return " ".join(t for t in texts if t)

def get_azure_embedding(text: str) -> List[float]:
    if not text.strip():
        raise ValueError("Empty text for embedding")

    response = azure_client.embeddings.create(
        input=[text.strip()],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

# =========================================================
# MYSQL SETUP
# =========================================================

MYSQL_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fashion_products (
    product_id VARCHAR(64) PRIMARY KEY,
    parent_asin VARCHAR(32),

    title TEXT,
    description TEXT,
    features TEXT,
    store_name VARCHAR(255),
    main_category VARCHAR(128),

    price FLOAT DEFAULT 0.0,
    average_rating FLOAT DEFAULT 0.0,
    rating_number INT DEFAULT 0,

    image TEXT,

    seasons TEXT,
    styles TEXT,
    occasions TEXT,
    genders TEXT,
    ages TEXT,
    article_types TEXT,
    materials TEXT,
    colors TEXT,
    sizes TEXT,
    vibe TEXT,

    bag_of_words LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_SQL = """
INSERT INTO fashion_products VALUES (
%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
)
"""

# =========================================================
# MILVUS SETUP
# =========================================================

def setup_milvus():
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    if utility.has_collection(MILVUS_COLLECTION):
        return Collection(MILVUS_COLLECTION)

    fields = [
        FieldSchema("product_id", DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),

        FieldSchema("parent_asin", DataType.VARCHAR, max_length=32),
        FieldSchema("title", DataType.VARCHAR, max_length=512),
        FieldSchema("price", DataType.FLOAT),
        FieldSchema("average_rating", DataType.FLOAT),
        FieldSchema("rating_number", DataType.INT64),
        FieldSchema("image", DataType.VARCHAR, max_length=1024),
        FieldSchema("store_name", DataType.VARCHAR, max_length=255),
        FieldSchema("main_category", DataType.VARCHAR, max_length=128),
        FieldSchema("styles", DataType.VARCHAR, max_length=255),
        FieldSchema("occasions", DataType.VARCHAR, max_length=255),
        FieldSchema("genders", DataType.VARCHAR, max_length=64),
        FieldSchema("materials", DataType.VARCHAR, max_length=255)
    ]

    schema = CollectionSchema(fields, "Amazon Fashion Embeddings")
    collection = Collection(MILVUS_COLLECTION, schema)

    collection.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 64}
        }
    )

    return collection

# =========================================================
# BACKGROUND PIPELINE
# =========================================================

def ingestion_pipeline(jsonl_path: str):
    logger.info("Starting ingestion pipeline")

    # ---------------- DUCKDB ----------------
    con = duckdb.connect("fashion.duckdb")
    con.execute("""
        CREATE OR REPLACE TABLE amazon_fashion_raw AS
        SELECT * FROM read_json_auto(?)
    """, [jsonl_path])

    df = con.execute("SELECT * FROM amazon_fashion_raw").df()
    con.close()

    logger.info(f"DuckDB loaded {len(df)} rows")

    # ---------------- MYSQL ----------------
    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = mysql_conn.cursor()
    cursor.execute(MYSQL_TABLE_SQL)
    mysql_conn.commit()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MySQL Insert"):
        product_id = str(uuid.uuid4())

        title = row.get("title") or ""
        description = safe_join(row.get("description"))
        features = safe_join(row.get("features"))
        store_name = row.get("store") or ""

        bag_of_words = build_bag_of_words([
            title, description, features, store_name
        ])

        cursor.execute(
            INSERT_SQL,
            (
                product_id,
                row.get("parent_asin"),
                title,
                description,
                features,
                store_name,
                row.get("main_category"),
                float(row.get("price") or 0.0),
                float(row.get("average_rating") or 0.0),
                int(row.get("rating_number") or 0),
                extract_image(row),
                "", "", "", "", "", "", "", "", "", "",
                bag_of_words
            )
        )

    mysql_conn.commit()
    cursor.close()
    mysql_conn.close()
    logger.info("MySQL ingestion done")

    # ---------------- MILVUS ----------------
    collection = setup_milvus()
    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = mysql_conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM fashion_products")

    rows = cursor.fetchall()

    for row in tqdm(rows, desc="Embedding → Milvus"):
        embedding = get_azure_embedding(row["bag_of_words"])

        collection.insert([
            [row["product_id"]],
            [embedding],
            [row["parent_asin"]],
            [row["title"]],
            [row["price"]],
            [row["average_rating"]],
            [row["rating_number"]],
            [row["image"]],
            [row["store_name"]],
            [row["main_category"]],
            [row["styles"]],
            [row["occasions"]],
            [row["genders"]],
            [row["materials"]],
        ])

    collection.flush()
    cursor.close()
    mysql_conn.close()
    logger.info("Milvus ingestion done")

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="Fashion Ingestion Pipeline")

@app.post("/ingest/jsonl")
async def ingest_jsonl(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    path = f"/tmp/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    background_tasks.add_task(ingestion_pipeline, path)

    return {
        "status": "accepted",
        "message": "Ingestion started in background"
    }
