import os
import uuid
import argparse
import logging
import re
import json
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np

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
BATCH_SIZE = 10

azure_client = AzureOpenAI(
    api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT")
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_AZURE_OPENAI_MODEL")

# =========================================================
# METADATA ENRICHMENT DICTIONARIES
# =========================================================

# Underage-related keywords for regex matching
UNDERAGE_KEYWORDS = [
    r'\bkid\b', r'\bkids\b', r'\bchild\b', r'\bchildren\b',
    r'\bboy\b', r'\bboys\b', r'\bgirl\b', r'\bgirls\b',
    r'\bbaby\b', r'\bbabies\b', r'\binfant\b', r'\binfants\b',
    r'\btoddler\b', r'\btoddlers\b', r'\bnewborn\b',
    r'\bjunior\b', r'\bjuniors\b', r'\byouth\b',
    r'\blittle\b', r'\btiny\b', r'\bmini\b',
    r'\bpreschool\b', r'\bschool[\s-]?age\b',
    r'\bteen\b', r'\bteens\b', r'\bteenager\b',
    r'\badolescent\b', r'\bpre[\s-]?teen\b'
]

CATEGORY_KEYWORDS = {
    # Tops
    "shirt", "shirts", "tshirt", "t-shirt", "tee", "top", "tops", "blouse",
    "tunic", "kurta", "kurti", "crop_top", "crop", "tank", "tanktop",
    "camisole", "henley", "polo",
    
    # Outerwear
    "hoodie", "sweatshirt", "sweater", "cardigan",
    "blazer", "jacket", "coat", "parka", "windcheater",
    "windbreaker", "overcoat", "poncho", "cape", "vest",
    "trench", "bomber", "puffer", "anorak",
    
    # Bottoms
    "jeans", "denim", "trousers", "pants", "chinos",
    "leggings", "joggers", "shorts", "skirt", "palazzo",
    "culottes", "trackpants", "capris", "bermuda",
    "cargo", "khakis", "slacks",
    
    # Ethnic
    "saree", "sari", "lehenga", "salwar", "suit",
    "anarkali", "kurta_set", "dhoti", "sherwani",
    "churidar", "pathani", "nehru", "bandhgala",
    
    # Dresses
    "dress", "gown", "maxi", "midi", "mini",
    "frock", "jumpsuit", "romper", "pinafore",
    "bodycon", "shift", "wrap", "sundress",
    
    # Footwear
    "shoes", "sneakers", "boots", "sandals",
    "heels", "flats", "slippers", "loafers",
    "trainers", "flipflops", "mules", "wedges",
    "oxfords", "brogues", "espadrilles", "clogs",
    "pumps", "stilettos", "mocassins", "ballet",
    
    # Accessories
    "watch", "bag", "handbag", "backpack", "purse",
    "wallet", "belt", "scarf", "stole", "shawl",
    "gloves", "socks", "cap", "beanie", "tie",
    "sunglasses", "jewellery", "jewelry", "hat",
    "necklace", "earrings", "bracelet", "ring",
    "headband", "hairband", "clips", "bows",
    
    # Inner / Others
    "innerwear", "sleepwear", "nightwear", "lingerie",
    "activewear", "sportswear", "swimwear", "swimsuit",
    "bikini", "trunks", "briefs", "boxers", "bra",
    "panties", "camisole", "thermal", "undershirt",
    
    # Generic
    "clothes", "clothing", "outfit", "wear", "apparel",
    "garment", "attire", "uniform"
}

STYLE_KEYWORDS = {
    "casual": ["casual", "daily", "everyday", "relaxed", "comfortable"],
    "formal": ["formal", "office", "business", "blazer", "professional", "corporate"],
    "sports": ["sports", "athletic", "gym", "running", "fitness", "performance"],
    "ethnic": ["ethnic", "kurta", "saree", "lehenga", "traditional", "cultural"],
    "party": ["party", "evening", "club", "night", "cocktail"],
    "smart_casual": ["smart", "smart_casual", "semi_formal"],
    "athleisure": ["athleisure", "activewear"],
    "streetwear": ["streetwear", "urban", "hip", "edgy"],
    "boho": ["boho", "bohemian", "hippie"],
    "vintage": ["vintage", "retro", "classic"],
    "chic": ["chic", "fashionable", "stylish"],
    "minimalist": ["minimalist", "simple", "clean"],
    "luxury": ["luxury", "premium", "designer", "haute"],
    "modest": ["modest", "conservative"],
    "trendy": ["trendy", "contemporary", "modern"]
}

OCCASION_KEYWORDS = {
    "winter": ["winter", "jacket", "hoodie", "sweater", "cold"],
    "summer": ["summer", "cotton", "linen", "warm", "lightweight"],
    "wedding": ["wedding", "bridal", "groom", "marriage", "ceremony"],
    "festival": ["festival", "festive", "diwali", "eid", "christmas", "holi"],
    "beach": ["beach", "swim", "resort", "vacation", "tropical"],
    "office": ["office", "work", "meeting", "professional"],
    "gym": ["gym", "workout", "fitness", "training", "exercise"],
    "date": ["date", "romantic", "dinner"],
    "travel": ["travel", "holiday", "vacation", "trip"],
    "college": ["college", "school", "campus", "university"],
    "yoga": ["yoga", "meditation", "zen"],
    "interview": ["interview", "job"],
    "celebration": ["celebration", "party", "birthday"],
    "outdoor": ["outdoor", "hiking", "camping", "adventure"]
}

MATERIAL_KEYWORDS = {
    "cotton": ["cotton", "organic_cotton", "khadi"],
    "wool": ["wool", "woolen", "merino"],
    "leather": ["leather", "genuine_leather"],
    "denim": ["denim", "jeans"],
    "polyester": ["polyester", "poly"],
    "silk": ["silk", "raw_silk", "mulberry"],
    "linen": ["linen"],
    "velvet": ["velvet"],
    "satin": ["satin"],
    "fleece": ["fleece"],
    "chiffon": ["chiffon"],
    "georgette": ["georgette"],
    "rayon": ["rayon", "viscose"],
    "modal": ["modal"],
    "spandex": ["spandex", "lycra", "elastane"],
    "suede": ["suede"],
    "nylon": ["nylon"],
    "acrylic": ["acrylic"],
    "faux_leather": ["faux_leather", "vegan_leather", "pu_leather"]
}

GENDER_MAP = {
    # Male
    "men": "male", "man": "male", "male": "male",
    "boy": "male", "boys": "male", "mens": "male",
    "gentlemen": "male", "gents": "male", "masculine": "male",
    
    # Female
    "women": "female", "woman": "female", "female": "female",
    "girl": "female", "girls": "female",
    "ladies": "female", "womens": "female", "feminine": "female",
    
    # Neutral
    "unisex": "unisex", "all": "unisex", "everyone": "unisex",
    "neutral": "unisex"
}

# =========================================================
# HELPERS
# =========================================================

def safe_join(val):
    if val is None:
        return ""
    if isinstance(val, np.ndarray):
        return " ".join(map(str, val.tolist())) if val.size else ""
    if isinstance(val, list):
        return " ".join(map(str, val))
    if isinstance(val, dict):
        return " ".join(map(str, val.values()))
    return str(val)

def extract_image(row: dict) -> str:
    """
    Extract the best quality image URL from the images field.
    Handles various formats: list, string, dict, or numpy array.
    Priority: hi_res > large > thumb
    """
    images = row.get("images")
    
    # Handle None
    if images is None:
        return ""
    
    # Handle numpy array - convert to list
    if isinstance(images, np.ndarray):
        if images.size == 0:
            return ""
        images = images.tolist()
    
    # Handle if images is a string (JSON string)
    if isinstance(images, str):
        if not images.strip():
            return ""
        try:
            images = json.loads(images)
        except (json.JSONDecodeError, TypeError):
            return ""
    
    # Handle if images is a list
    if isinstance(images, list):
        if len(images) == 0:
            return ""
        
        first_image = images[0]
        
        # Handle if first_image is a dict
        if isinstance(first_image, dict):
            return (first_image.get("hi_res") or 
                   first_image.get("large") or 
                   first_image.get("thumb") or "")
        
        # Handle if first_image is a string URL
        if isinstance(first_image, str):
            return first_image
    
    # Handle if images is directly a dict (single image)
    if isinstance(images, dict):
        return (images.get("hi_res") or 
               images.get("large") or 
               images.get("thumb") or "")
    
    return ""

def build_bag_of_words(parts: List[str]) -> str:
    return " ".join(p for p in parts if p).lower()

def detect_for_underage(bag_of_words: str) -> bool:
    """
    Use regex to detect if product is for underage individuals
    """
    for pattern in UNDERAGE_KEYWORDS:
        if re.search(pattern, bag_of_words, re.IGNORECASE):
            return True
    return False

def enrich_metadata(text: str) -> dict:
    tokens = set(text.split())
    
    def match(mapping):
        return ", ".join(
            key for key, words in mapping.items()
            if any(w in tokens for w in words)
        )
    
    detected_genders = {GENDER_MAP[t] for t in tokens if t in GENDER_MAP}
    
    def resolve_gender(genders: set) -> str:
        if "male" in genders and "female" in genders:
            return "unisex"
        if "unisex" in genders:
            return "unisex"
        if "male" in genders:
            return "male"
        if "female" in genders:
            return "female"
        return "unisex"
    
    return {
        "styles": match(STYLE_KEYWORDS),
        "occasions": match(OCCASION_KEYWORDS),
        "materials": match(MATERIAL_KEYWORDS),
        "genders": resolve_gender(detected_genders),
        "for_underage": detect_for_underage(text)
    }

def get_azure_embeddings_batch(texts: List[str]) -> List[List[float]]:
    response = azure_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [item.embedding for item in response.data]

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
    for_underage BOOLEAN DEFAULT FALSE,
    bag_of_words LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_SQL = """
INSERT INTO fashion_products (
    product_id, parent_asin, title, description, features,
    store_name, main_category, price, average_rating,
    rating_number, image, seasons, styles, occasions,
    genders, ages, article_types, materials,
    colors, sizes, vibe, for_underage, bag_of_words
) VALUES (
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
)
"""

# =========================================================
# MILVUS SETUP
# =========================================================

def setup_milvus():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    
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
        FieldSchema("materials", DataType.VARCHAR, max_length=255),
        FieldSchema("for_underage", DataType.BOOL)
    ]
    
    schema = CollectionSchema(fields, "Amazon Fashion Embeddings")
    collection = Collection(MILVUS_COLLECTION, schema)
    
    collection.create_index(
        "embedding",
        {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 64}}
    )
    return collection

# =========================================================
# INGESTION PIPELINE
# =========================================================

def ingestion_pipeline(jsonl_path: str, limit: int | None = None):
    logger.info(f"Starting ingestion: {jsonl_path}")
    
    con = duckdb.connect("fashion.duckdb")
    sql = "SELECT * FROM read_json_auto(?)"
    if limit:
        sql += f" LIMIT {limit}"
    df = con.execute(sql, [jsonl_path]).df()
    con.close()
    
    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = mysql_conn.cursor()
    cursor.execute(MYSQL_TABLE_SQL)
    mysql_conn.commit()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MySQL Insert"):
        title = row.get("title") or ""
        description = safe_join(row.get("description"))
        features = safe_join(row.get("features"))
        store_name = row.get("store") or ""
        
        bag = build_bag_of_words([title, description, features, store_name])
        enriched = enrich_metadata(bag)
        
        cursor.execute(
            INSERT_SQL,
            (
                str(uuid.uuid4()),
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
                "",
                enriched["styles"],
                enriched["occasions"],
                enriched["genders"],
                "",
                "",
                enriched["materials"],
                "",
                "",
                "",
                enriched["for_underage"],
                bag
            )
        )
    
    mysql_conn.commit()
    cursor.close()
    mysql_conn.close()
    logger.info("MySQL ingestion completed")
    
    collection = setup_milvus()
    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = mysql_conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM fashion_products")
    rows = cursor.fetchall()
    
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Milvus Embedding"):
        batch = [r for r in rows[i:i+BATCH_SIZE] if r["bag_of_words"].strip()]
        if not batch:
            continue
        
        embeddings = get_azure_embeddings_batch([r["bag_of_words"] for r in batch])
        
        collection.insert([
            [r["product_id"] for r in batch],
            embeddings,
            [r["parent_asin"] or "" for r in batch],
            [r["title"] or "" for r in batch],
            [float(r["price"]) if r["price"] is not None else 0.0 for r in batch],
            [float(r["average_rating"]) for r in batch],
            [int(r["rating_number"]) for r in batch],
            [r["image"] or "" for r in batch],
            [r["store_name"] or "" for r in batch],
            [r["main_category"] or "" for r in batch],
            [r["styles"] or "" for r in batch],
            [r["occasions"] or "" for r in batch],
            [r["genders"] or "" for r in batch],
            [r["materials"] or "" for r in batch],
            [bool(r["for_underage"]) for r in batch]
        ])
    
    collection.flush()
    cursor.close()
    mysql_conn.close()
    logger.info("Milvus ingestion completed")

# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon Fashion Ingestion CLI")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    
    ingestion_pipeline(args.jsonl, args.limit)
