import os
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
    "database": "semantic_fashion_db_fl",
    "autocommit": False
}

MYSQL_BATCH_SIZE = 5000
DUCKDB_FETCH_SIZE = 5000

# =========================================================
# METADATA ENRICHMENT
# =========================================================

UNDERAGE_KEYWORDS = [
    r'\bkid\b', r'\bkids\b', r'\bchild\b', r'\bchildren\b',
    r'\bboy\b', r'\bboys\b', r'\bgirl\b', r'\bgirls\b',
    r'\bbaby\b', r'\bbabies\b', r'\binfant\b', r'\btoddler\b',
    r'\bteen\b', r'\bteens\b', r'\bteenager\b'
]

UNDERAGE_REGEX = re.compile("|".join(UNDERAGE_KEYWORDS), re.IGNORECASE)

STYLE_KEYWORDS = {
    "casual": ["casual", "daily"],
    "formal": ["formal", "office"],
    "sports": ["sports", "gym"],
    "ethnic": ["kurta", "saree"],
    "party": ["party", "evening"]
}

OCCASION_KEYWORDS = {
    "winter": ["winter", "jacket"],
    "summer": ["summer", "cotton"],
    "wedding": ["wedding", "bridal"],
    "travel": ["travel", "trip"]
}

MATERIAL_KEYWORDS = {
    "cotton": ["cotton"],
    "wool": ["wool"],
    "denim": ["denim", "jeans"],
    "polyester": ["polyester"]
}

GENDER_MAP = {
    "men": "male", "man": "male",
    "women": "female", "woman": "female",
    "boys": "male", "girls": "female",
    "unisex": "unisex"
}

# =========================================================
# HELPERS
# =========================================================

def safe_join(val):
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(map(str, val))
    if isinstance(val, dict):
        return " ".join(map(str, val.values()))
    if isinstance(val, np.ndarray):
        return " ".join(map(str, val.tolist()))
    return str(val)

def safe_truncate(text, max_length):
    """Safely truncate text to max_length"""
    if text is None:
        return ""
    text = str(text)
    return text[:max_length] if len(text) > max_length else text

def build_bag_of_words(parts: List[str]) -> str:
    return " ".join(p for p in parts if p).lower()

def detect_for_underage(text: str) -> bool:
    return bool(UNDERAGE_REGEX.search(text))

def enrich_metadata(text: str) -> dict:
    tokens = set(text.split())

    def match(mapping):
        return ", ".join(
            key for key, words in mapping.items()
            if any(w in tokens for w in words)
        )

    genders = {GENDER_MAP[t] for t in tokens if t in GENDER_MAP}

    if "male" in genders and "female" in genders:
        gender = "unisex"
    elif genders:
        gender = list(genders)[0]
    else:
        gender = "unisex"

    return {
        "styles": match(STYLE_KEYWORDS),
        "occasions": match(OCCASION_KEYWORDS),
        "materials": match(MATERIAL_KEYWORDS),
        "genders": gender,
        "for_underage": detect_for_underage(text)
    }

# =========================================================
# MYSQL DDL
# =========================================================

MYSQL_PRODUCTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS products (
    product_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    parent_asin VARCHAR(20),
    title VARCHAR(1000),
    description TEXT,
    store VARCHAR(200),
    main_category VARCHAR(100),
    price DECIMAL(10,2),
    average_rating DECIMAL(3,1),
    rating_number INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_parent_asin (parent_asin),
    INDEX idx_category (main_category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

MYSQL_METADATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS product_metadata (
    metadata_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT UNSIGNED,
    styles TEXT,
    occasions TEXT,
    materials TEXT,
    genders VARCHAR(20),
    for_underage BOOLEAN,
    bag_of_words LONGTEXT,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    INDEX idx_product (product_id),
    INDEX idx_gender (genders)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

MYSQL_FEATURES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS product_features (
    feature_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT UNSIGNED,
    feature_text TEXT,
    feature_order INT,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    INDEX idx_product (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

MYSQL_IMAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS product_images (
    image_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT UNSIGNED,
    variant VARCHAR(20),
    thumb_url VARCHAR(1000),
    large_url VARCHAR(1000),
    hi_res_url VARCHAR(1000),
    image_order INT,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    INDEX idx_product (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

# =========================================================
# INSERT SQL
# =========================================================

INSERT_PRODUCT_SQL = """
INSERT INTO products
(parent_asin, title, description, store, main_category, price, average_rating, rating_number)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
"""

INSERT_METADATA_SQL = """
INSERT INTO product_metadata
(product_id, styles, occasions, materials, genders, for_underage, bag_of_words)
VALUES (%s,%s,%s,%s,%s,%s,%s)
"""

INSERT_FEATURE_SQL = """
INSERT INTO product_features
(product_id, feature_text, feature_order)
VALUES (%s,%s,%s)
"""

INSERT_IMAGE_SQL = """
INSERT INTO product_images
(product_id, variant, thumb_url, large_url, hi_res_url, image_order)
VALUES (%s,%s,%s,%s,%s,%s)
"""

# =========================================================
# INGESTION PIPELINE (STREAMING)
# =========================================================

def ingestion_pipeline(jsonl_path: str, limit: int | None = None):
    logger.info("Connecting to DuckDB (streaming mode)")
    con = duckdb.connect()
    sql = f"SELECT * FROM read_json_auto('{jsonl_path}')"
    if limit:
        sql += f" LIMIT {limit}"
    cur = con.execute(sql)

    columns = [d[0] for d in cur.description]

    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = mysql_conn.cursor()

    # Optimize MySQL for bulk insert
    logger.info("Optimizing MySQL for bulk insert")
    cursor.execute("SET autocommit=0")
    cursor.execute("SET unique_checks=0")
    cursor.execute("SET foreign_key_checks=0")
    cursor.execute("SET sql_log_bin=0")

    cursor.execute(MYSQL_PRODUCTS_TABLE_SQL)
    cursor.execute(MYSQL_METADATA_TABLE_SQL)
    cursor.execute(MYSQL_FEATURES_TABLE_SQL)
    cursor.execute(MYSQL_IMAGES_TABLE_SQL)
    mysql_conn.commit()

    product_batch = []
    metadata_batch = []
    features_batch = []
    images_batch = []

    pbar = tqdm(unit="rows", desc="Ingesting")

    processed = 0
    batch_count = 0

    while True:
        rows = cur.fetchmany(DUCKDB_FETCH_SIZE)
        if not rows:
            break

        for row_tuple in rows:
            row = dict(zip(columns, row_tuple))
            processed += 1
            pbar.update(1)

            title = safe_truncate(row.get("title") or "", 1000)
            description = safe_join(row.get("description"))
            features = row.get("features", [])
            store = safe_truncate(row.get("store") or "", 200)

            bag = build_bag_of_words([title, description, safe_join(features), store])
            meta = enrich_metadata(bag)

            product_batch.append((
                safe_truncate(row.get("parent_asin"), 20),
                title,
                description,
                store,
                safe_truncate(row.get("main_category"), 100),
                float(row.get("price") or 0),
                float(row.get("average_rating") or 0),
                int(row.get("rating_number") or 0)
            ))

            temp_idx = len(product_batch) - 1

            metadata_batch.append((
                temp_idx,
                meta["styles"],
                meta["occasions"],
                meta["materials"],
                meta["genders"],
                meta["for_underage"],
                bag
            ))

            if isinstance(features, list):
                for i, f in enumerate(features, 1):
                    # Truncate feature text - now using TEXT so no hard limit
                    feature_text = str(f) if f else ""
                    features_batch.append((temp_idx, feature_text, i))

            images = row.get("images", [])
            if isinstance(images, list):
                for img_idx, img in enumerate(images[:5], 1):
                    if isinstance(img, dict):
                        images_batch.append((
                            temp_idx,
                            safe_truncate(img.get("variant"), 20),
                            safe_truncate(img.get("thumb"), 1000),
                            safe_truncate(img.get("large"), 1000),
                            safe_truncate(img.get("hi_res"), 1000),
                            img_idx
                        ))

            if len(product_batch) >= MYSQL_BATCH_SIZE:
                batch_count += 1
                flush_batches(cursor, mysql_conn,
                              product_batch, metadata_batch,
                              features_batch, images_batch)
                logger.info(f"Flushed batch {batch_count}: {len(product_batch)} products")
                product_batch.clear()
                metadata_batch.clear()
                features_batch.clear()
                images_batch.clear()

    if product_batch:
        batch_count += 1
        flush_batches(cursor, mysql_conn,
                      product_batch, metadata_batch,
                      features_batch, images_batch)
        logger.info(f"Flushed final batch {batch_count}: {len(product_batch)} products")

    # Re-enable checks
    logger.info("Re-enabling MySQL checks")
    cursor.execute("SET unique_checks=1")
    cursor.execute("SET foreign_key_checks=1")
    cursor.execute("SET sql_log_bin=1")

    pbar.close()
    cursor.close()
    mysql_conn.close()
    con.close()

    logger.info("="*60)
    logger.info(f"Completed ingestion: {processed} rows")
    logger.info(f"Total batches processed: {batch_count}")
    logger.info("="*60)

# =========================================================
# FLUSH FUNCTION
# =========================================================

def flush_batches(cursor, conn,
                  product_batch, metadata_batch,
                  features_batch, images_batch):
    """
    Insert products first, then update foreign keys in related tables
    """
    
    # Insert all products
    cursor.executemany(INSERT_PRODUCT_SQL, product_batch)
    conn.commit()

    # Get the first inserted product_id for this batch
    cursor.execute("SELECT LAST_INSERT_ID()")
    first_id = cursor.fetchone()[0]
    
    # Update metadata batch with actual product_ids
    updated_metadata = []
    for temp_idx, styles, occasions, materials, genders, for_underage, bag in metadata_batch:
        actual_product_id = first_id + temp_idx
        updated_metadata.append((
            actual_product_id,
            styles,
            occasions,
            materials,
            genders,
            for_underage,
            bag
        ))
    
    # Update features batch with actual product_ids
    updated_features = []
    for temp_idx, feature_text, feature_order in features_batch:
        actual_product_id = first_id + temp_idx
        updated_features.append((
            actual_product_id,
            feature_text,
            feature_order
        ))
    
    # Update images batch with actual product_ids
    updated_images = []
    for temp_idx, variant, thumb, large, hi_res, img_order in images_batch:
        actual_product_id = first_id + temp_idx
        updated_images.append((
            actual_product_id,
            variant,
            thumb,
            large,
            hi_res,
            img_order
        ))

    # Insert all related data with correct product_ids
    if updated_metadata:
        cursor.executemany(INSERT_METADATA_SQL, updated_metadata)
    
    if updated_features:
        cursor.executemany(INSERT_FEATURE_SQL, updated_features)
    
    if updated_images:
        cursor.executemany(INSERT_IMAGE_SQL, updated_images)

    conn.commit()

# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon Fashion JSONL to MySQL - Optimized for large datasets")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file")
    parser.add_argument("--limit", type=int, help="Limit number of rows (for testing)")
    args = parser.parse_args()

    ingestion_pipeline(args.jsonl, args.limit)
