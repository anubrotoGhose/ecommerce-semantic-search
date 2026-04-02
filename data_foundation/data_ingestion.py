import os
import time
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from decimal import Decimal

import mysql.connector
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from openai import AzureOpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv

load_dotenv()

# ======================
# Configuration
# ======================

MYSQL_CONFIG = {
    "host": os.getenv("mysql_db_host"),
    "port": int(os.getenv("mysql_db_port", "3306")),
    "user": os.getenv("mysql_db_user"),
    "password": os.getenv("mysql_db_password"),
    "database": "semantic_fashion_db_fl",
}

MILVUS_HOST = "10.169.101.75"
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = "amazon_fashion_catalog"

EMBEDDING_MODEL = os.getenv("EMBEDDING_AZURE_OPENAI_MODEL")
EMBEDDING_DIM = 1536  # text-embedding-ada-002 dimension
BATCH_SIZE = 10
PROGRESS_FILE = "ingestion_progress.json"
LOG_FILE = "ingestion.log"

# Field size limits (in bytes)
FIELD_LIMITS = {
    "product_id": 64,
    "parent_asin": 32,
    "title": 512,
    "image": 1024,
    "store_name": 255,
    "main_category": 128,
    "styles": 255,
    "occasions": 255,
    "genders": 64,
    "materials": 255,
}

# ======================
# Logging Setup
# ======================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ======================
# Azure OpenAI Client (Load Once)
# ======================

azure_client = AzureOpenAI(
    api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT"),
)

logger.info("Azure OpenAI client initialized")

# ======================
# Retry Logic for Rate Limiting
# ======================

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: logger.warning(
        f"Rate limit hit. Retrying in {retry_state.next_action.sleep} seconds..."
    ),
)
def get_azure_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Azure OpenAI with retry logic for rate limiting."""
    try:
        response = azure_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return [item.embedding for item in response.data]
    except RateLimitError as e:
        logger.error(f"Rate limit error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise


# ======================
# Progress Tracking
# ======================

class ProgressTracker:
    """Track ingestion progress to enable resumability."""

    def __init__(self, filepath: str = PROGRESS_FILE):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load progress from file."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                return json.load(f)
        return {
            "last_processed_id": 0,
            "total_processed": 0,
            "last_updated": None,
            "batches_completed": 0,
        }

    def save(self):
        """Save progress to file."""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)
        logger.debug(f"Progress saved: {self.data}")

    def update(self, last_id: int, batch_size: int):
        """Update progress after successful batch."""
        self.data["last_processed_id"] = last_id
        self.data["total_processed"] += batch_size
        self.data["batches_completed"] += 1
        self.save()

    def get_last_processed_id(self) -> int:
        """Get the last successfully processed metadata_id."""
        return self.data["last_processed_id"]


# ======================
# Milvus Collection Setup
# ======================

def create_milvus_collection():
    """Create or get existing Milvus collection with specified schema."""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    # Define schema with specified field limits
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
        FieldSchema("for_underage", DataType.BOOL),
    ]

    schema = CollectionSchema(
        fields=fields, description="Amazon Fashion product metadata with embeddings"
    )

    # Create or get collection
    if utility.has_collection(MILVUS_COLLECTION):
        collection = Collection(MILVUS_COLLECTION)
        logger.info(f"Using existing collection: {MILVUS_COLLECTION}")
    else:
        collection = Collection(name=MILVUS_COLLECTION, schema=schema)
        logger.info(f"Created new collection: {MILVUS_COLLECTION}")

    return collection


# ======================
# MySQL Data Fetching
# ======================

def fetch_mysql_batch(
    cursor, last_processed_id: int, batch_size: int
) -> List[Dict[str, Any]]:
    """Fetch a batch of records from MySQL."""
    query = """
        SELECT
            metadata_id, product_id, parent_asin, title, price,
            average_rating, rating_number, image, store_name,
            main_category, styles, occasions, materials, genders,
            for_underage, bag_of_words
        FROM product_metadata
        WHERE metadata_id > %s
        ORDER BY metadata_id ASC
        LIMIT %s
    """

    cursor.execute(query, (last_processed_id, batch_size))
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    return [dict(zip(columns, row)) for row in rows]


# ======================
# Data Transformation
# ======================

def truncate_to_bytes(text: str, max_bytes: int) -> str:
    """
    Truncate string to fit within max_bytes when encoded as UTF-8.
    Milvus VARCHAR max_length is measured in bytes, not characters.
    """
    if not text:
        return ""
    
    # Encode to bytes and check length
    encoded = text.encode('utf-8', errors='ignore')
    if len(encoded) <= max_bytes:
        return text
    
    # Truncate bytes and decode back, handling potential split characters
    truncated = encoded[:max_bytes]
    # Decode with error handling for potentially incomplete multi-byte characters
    result = truncated.decode('utf-8', errors='ignore')
    
    # Log truncation for monitoring
    if len(result) < len(text):
        logger.debug(f"Truncated field from {len(text)} to {len(result)} chars (byte limit: {max_bytes})")
    
    return result


def transform_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Transform MySQL row to match Milvus schema with automatic truncation."""
    
    # Convert product_id to string for VARCHAR primary key
    product_id_str = str(row["product_id"]) if row["product_id"] else "0"
    
    return {
        "product_id": truncate_to_bytes(product_id_str, FIELD_LIMITS["product_id"]),
        "parent_asin": truncate_to_bytes(row["parent_asin"] or "", FIELD_LIMITS["parent_asin"]),
        "title": truncate_to_bytes(row["title"] or "", FIELD_LIMITS["title"]),
        "price": float(row["price"]) if row["price"] else 0.0,
        "average_rating": float(row["average_rating"]) if row["average_rating"] else 0.0,
        "rating_number": int(row["rating_number"]) if row["rating_number"] else 0,
        "image": truncate_to_bytes(row["image"] or "", FIELD_LIMITS["image"]),
        "store_name": truncate_to_bytes(row["store_name"] or "", FIELD_LIMITS["store_name"]),
        "main_category": truncate_to_bytes(row["main_category"] or "", FIELD_LIMITS["main_category"]),
        "styles": truncate_to_bytes(row["styles"] or "", FIELD_LIMITS["styles"]),
        "occasions": truncate_to_bytes(row["occasions"] or "", FIELD_LIMITS["occasions"]),
        "genders": truncate_to_bytes(row["genders"] or "", FIELD_LIMITS["genders"]),
        "materials": truncate_to_bytes(row["materials"] or "", FIELD_LIMITS["materials"]),
        "for_underage": bool(row["for_underage"]) if row["for_underage"] is not None else False,
        "bag_of_words": row["bag_of_words"] or "",  # Used for embeddings, not stored in Milvus
    }


# ======================
# Main Ingestion Pipeline
# ======================

# ======================
# Main Ingestion Pipeline (OPTIMIZED)
# ======================

def ingest_data():
    """Main ingestion pipeline with progress tracking and rate limiting."""
    tracker = ProgressTracker()
    last_processed_id = tracker.get_last_processed_id()

    logger.info(f"Starting ingestion from metadata_id > {last_processed_id}")

    # Connect to MySQL
    connection = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    logger.info("Connected to MySQL")

    # Setup Milvus collection
    collection = create_milvus_collection()

    total_inserted = 0
    start_time = time.time()

    try:
        while True:
            # Fetch batch from MySQL
            batch_rows = fetch_mysql_batch(cursor, last_processed_id, BATCH_SIZE)

            if not batch_rows:
                logger.info("No more records to process. Ingestion complete!")
                break

            logger.info(
                f"Processing batch: {len(batch_rows)} records "
                f"(IDs: {batch_rows[0]['metadata_id']} - {batch_rows[-1]['metadata_id']})"
            )

            # Transform data
            transformed_data = [transform_row(row) for row in batch_rows]

            # Extract bag_of_words for embedding (fallback to title if empty)
            texts = [
                row["bag_of_words"] if row["bag_of_words"] else row["title"]
                for row in transformed_data
            ]

            # Get embeddings with rate limiting
            try:
                embeddings = get_azure_embeddings_batch(texts)
                logger.info(f"Generated {len(embeddings)} embeddings")
            except Exception as e:
                logger.error(f"Failed to get embeddings after retries: {e}")
                break

            # Add embeddings to data - CREATE LIST OF DICTS
            milvus_data = []
            for i, data_row in enumerate(transformed_data):
                record = {k: v for k, v in data_row.items() if k != "bag_of_words"}
                record["embedding"] = embeddings[i]
                milvus_data.append(record)

            # Insert into Milvus - NO FLUSH NEEDED
            try:
                result = collection.insert(milvus_data)
                # REMOVED: collection.flush() - Let Milvus handle this automatically
                total_inserted += len(result.primary_keys)
                logger.info(
                    f"✓ Inserted batch into Milvus: {len(result.primary_keys)} records "
                    f"(Total: {total_inserted})"
                )
            except Exception as e:
                logger.error(f"Failed to insert into Milvus: {e}")
                logger.error(f"Sample data structure: {milvus_data[0].keys()}")
                logger.error(f"Sample record: {milvus_data[0]}")
                break

            # Update progress
            last_processed_id = batch_rows[-1]["metadata_id"]
            tracker.update(last_processed_id, len(batch_rows))

            # REMOVED: time.sleep(1) - Unnecessary delay

    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        raise
    finally:
        cursor.close()
        connection.close()
        elapsed = time.time() - start_time
        logger.info(f"MySQL connection closed. Total time: {elapsed:.2f}s, "
                   f"Avg: {elapsed/total_inserted if total_inserted > 0 else 0:.3f}s per record")

    # Flush once at the end to ensure all data is persisted
    logger.info("Final flush to ensure data persistence...")
    collection.flush()
    
    # Create index after all data is inserted
    logger.info("Creating indexes on Milvus collection...")
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {
            "M": 16,
            "efConstruction": 200
        },
    }

    try:
        if not collection.has_index():
            collection.create_index(field_name="embedding", index_params=index_params)
            logger.info("✓ HNSW index created on embedding field")
        else:
            logger.info("✓ Index already exists")
        
        # Load collection into memory for searching
        collection.load()
        logger.info(f"✓ Collection loaded. Total records: {collection.num_entities}")
    except Exception as e:
        logger.warning(f"Error creating index or loading collection: {e}")



# ======================
# Entry Point
# ======================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting MySQL to Milvus Ingestion Pipeline")
    logger.info("=" * 60)
    ingest_data()
    logger.info("=" * 60)
    logger.info("Ingestion pipeline completed successfully!")
    logger.info("=" * 60)
