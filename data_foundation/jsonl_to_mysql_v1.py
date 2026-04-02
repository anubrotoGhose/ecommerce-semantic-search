import os
import argparse
import csv
import tempfile
from pathlib import Path
from tqdm import tqdm
import duckdb
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

MYSQL_CONFIG = {
    "host": os.getenv("mysql_db_host", "localhost"),
    "port": int(os.getenv("mysql_db_port", 3306)),
    "user": os.getenv("mysql_db_user"),
    "password": os.getenv("mysql_db_password"),
    "database": "semantic_fashion_db_fl",
    "allow_local_infile": True  # Fixed: was 'local_infile'
}

# Process in chunks to limit memory
CHUNK_SIZE = 50000  # Process 50K rows at a time

def process_chunk_to_csv(rows, columns, temp_dir, chunk_id, start_product_id):
    """
    Process a chunk of rows and write to CSV files.
    Returns next product_id to use.
    Memory efficient: only processes one chunk at a time.
    """
    products_csv = os.path.join(temp_dir, f"products_{chunk_id}.csv")
    metadata_csv = os.path.join(temp_dir, f"metadata_{chunk_id}.csv")
    features_csv = os.path.join(temp_dir, f"features_{chunk_id}.csv")
    images_csv = os.path.join(temp_dir, f"images_{chunk_id}.csv")
    
    prod_file = open(products_csv, 'w', newline='', encoding='utf-8')
    meta_file = open(metadata_csv, 'w', newline='', encoding='utf-8')
    feat_file = open(features_csv, 'w', newline='', encoding='utf-8')
    img_file = open(images_csv, 'w', newline='', encoding='utf-8')
    
    # Use | as delimiter to avoid conflicts with tabs in data
    prod_writer = csv.writer(prod_file, delimiter='|', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    meta_writer = csv.writer(meta_file, delimiter='|', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    feat_writer = csv.writer(feat_file, delimiter='|', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    img_writer = csv.writer(img_file, delimiter='|', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    
    product_id = start_product_id
    metadata_id = start_product_id
    feature_id = start_product_id * 10
    image_id = start_product_id * 10
    
    for row_tuple in rows:
        row = dict(zip(columns, row_tuple))
        
        # Product
        title = (row.get("title") or "")[:1000].replace('\n', ' ').replace('\r', ' ')
        description_raw = row.get("description") or ""
        if isinstance(description_raw, list):
            description = " ".join(str(x) for x in description_raw)[:5000]
        else:
            description = str(description_raw)[:5000]
        description = description.replace('\n', ' ').replace('\r', ' ')
        store = (row.get("store") or "")[:200].replace('\n', ' ').replace('\r', ' ')
        
        prod_writer.writerow([
            product_id,
            (row.get("parent_asin") or "")[:20],
            title,
            description,
            store,
            (row.get("main_category") or "")[:100],
            float(row.get("price") or 0),
            float(row.get("average_rating") or 0),
            int(row.get("rating_number") or 0)
        ])
        
        # Metadata
        bag = f"{title} {description} {store}"[:10000]
        gender = "unisex"
        if "women" in bag.lower():
            gender = "female"
        elif "men" in bag.lower():
            gender = "male"
        
        meta_writer.writerow([
            metadata_id,
            product_id,
            "casual",
            "summer",
            "cotton",
            gender,
            0,
            bag
        ])
        metadata_id += 1
        
        # Features (first 5)
        features = row.get("features", [])
        if isinstance(features, list):
            for i, f in enumerate(features[:5], 1):
                feat_text = str(f)[:1000].replace('\n', ' ').replace('\r', ' ')
                feat_writer.writerow([
                    feature_id,
                    product_id,
                    feat_text,
                    i
                ])
                feature_id += 1
        
        # Images (first 3)
        images = row.get("images", [])
        if isinstance(images, list):
            for i, img in enumerate(images[:3], 1):
                if isinstance(img, dict):
                    img_writer.writerow([
                        image_id,
                        product_id,
                        (img.get("variant") or "")[:20],
                        (img.get("thumb") or "")[:1000],
                        (img.get("large") or "")[:1000],
                        (img.get("hi_res") or "")[:1000],
                        i
                    ])
                    image_id += 1
        
        product_id += 1
    
    prod_file.close()
    meta_file.close()
    feat_file.close()
    img_file.close()
    
    return product_id, [products_csv, metadata_csv, features_csv, images_csv]

def load_csv_into_mysql(cursor, csv_files):
    """Load CSV files into MySQL using LOAD DATA LOCAL INFILE"""
    products_csv, metadata_csv, features_csv, images_csv = csv_files
    
    # Products
    cursor.execute(f"""
    LOAD DATA LOCAL INFILE '{products_csv}'
    INTO TABLE products
    FIELDS TERMINATED BY '|'
    ESCAPED BY '\\\\'
    LINES TERMINATED BY '\\n'
    (product_id, parent_asin, title, description, store, main_category, price, average_rating, rating_number)
    """)
    
    # Metadata
    cursor.execute(f"""
    LOAD DATA LOCAL INFILE '{metadata_csv}'
    INTO TABLE product_metadata
    FIELDS TERMINATED BY '|'
    ESCAPED BY '\\\\'
    LINES TERMINATED BY '\\n'
    (metadata_id, product_id, styles, occasions, materials, genders, for_underage, bag_of_words)
    """)
    
    # Features
    cursor.execute(f"""
    LOAD DATA LOCAL INFILE '{features_csv}'
    INTO TABLE product_features
    FIELDS TERMINATED BY '|'
    ESCAPED BY '\\\\'
    LINES TERMINATED BY '\\n'
    (feature_id, product_id, feature_text, feature_order)
    """)
    
    # Images
    cursor.execute(f"""
    LOAD DATA LOCAL INFILE '{images_csv}'
    INTO TABLE product_images
    FIELDS TERMINATED BY '|'
    ESCAPED BY '\\\\'
    LINES TERMINATED BY '\\n'
    (image_id, product_id, variant, thumb_url, large_url, hi_res_url, image_order)
    """)

def ingest_streaming(jsonl_path: str):
    """
    Memory-efficient ingestion using chunked LOAD DATA INFILE
    Peak memory: ~300-500 MB (similar to current approach)
    Speed: 10-20x faster than INSERT
    """
    
    print(f"Starting memory-efficient ingestion: {jsonl_path}")
    
    # Setup temp directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    # Setup MySQL
    print("Connecting to MySQL...")
    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = mysql_conn.cursor()
    
    # Create tables
    print("Creating tables...")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id BIGINT UNSIGNED PRIMARY KEY,
        parent_asin VARCHAR(20),
        title VARCHAR(1000),
        description TEXT,
        store VARCHAR(200),
        main_category VARCHAR(100),
        price DECIMAL(10,2),
        average_rating DECIMAL(3,1),
        rating_number INT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS product_metadata (
        metadata_id BIGINT UNSIGNED PRIMARY KEY,
        product_id BIGINT UNSIGNED,
        styles TEXT,
        occasions TEXT,
        materials TEXT,
        genders VARCHAR(20),
        for_underage BOOLEAN,
        bag_of_words LONGTEXT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS product_features (
        feature_id BIGINT UNSIGNED PRIMARY KEY,
        product_id BIGINT UNSIGNED,
        feature_text TEXT,
        feature_order INT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS product_images (
        image_id BIGINT UNSIGNED PRIMARY KEY,
        product_id BIGINT UNSIGNED,
        variant VARCHAR(20),
        thumb_url VARCHAR(1000),
        large_url VARCHAR(1000),
        hi_res_url VARCHAR(1000),
        image_order INT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """)
    
    # Optimize
    print("Optimizing MySQL settings...")
    cursor.execute("SET foreign_key_checks=0")
    cursor.execute("SET unique_checks=0")
    cursor.execute("SET autocommit=0")
    mysql_conn.commit()
    
    # Process JSONL in chunks
    print("Processing JSONL in chunks...")
    con = duckdb.connect()
    sql = f"SELECT * FROM read_json_auto('{jsonl_path}')"
    cur = con.execute(sql)
    columns = [d[0] for d in cur.description]
    
    chunk_id = 0
    start_product_id = 1
    processed = 0
    
    with tqdm(desc="Overall progress", unit=" rows") as pbar:
        while True:
            # Fetch chunk (memory efficient - only loads CHUNK_SIZE rows)
            rows = cur.fetchmany(CHUNK_SIZE)
            if not rows:
                break
            
            print(f"\n[Chunk {chunk_id + 1}] Processing {len(rows)} rows...")
            
            # Process to CSV
            next_product_id, csv_files = process_chunk_to_csv(
                rows, columns, temp_dir, chunk_id, start_product_id
            )
            
            # Load into MySQL
            print(f"[Chunk {chunk_id + 1}] Loading into MySQL...")
            try:
                load_csv_into_mysql(cursor, csv_files)
                mysql_conn.commit()
            except Exception as e:
                print(f"Error loading chunk {chunk_id}: {e}")
                # Show first few lines of CSV for debugging
                with open(csv_files[0], 'r') as f:
                    print("First 3 lines of products CSV:")
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        print(line[:200])
                raise
            
            # Cleanup CSV files to save disk space
            for f in csv_files:
                os.remove(f)
            
            processed += len(rows)
            pbar.update(len(rows))
            print(f"[Chunk {chunk_id + 1}] ✓ Loaded. Total: {processed} rows")
            
            start_product_id = next_product_id
            chunk_id += 1
    
    con.close()
    
    # Add indexes
    print("\nCreating indexes...")
    cursor.execute("ALTER TABLE products ADD INDEX idx_parent_asin (parent_asin)")
    cursor.execute("ALTER TABLE product_metadata ADD INDEX idx_product (product_id)")
    cursor.execute("ALTER TABLE product_features ADD INDEX idx_product (product_id)")
    cursor.execute("ALTER TABLE product_images ADD INDEX idx_product (product_id)")
    
    # Re-enable checks
    print("Re-enabling checks...")
    cursor.execute("SET foreign_key_checks=1")
    cursor.execute("SET unique_checks=1")
    cursor.execute("SET autocommit=1")
    mysql_conn.commit()
    
    cursor.close()
    mysql_conn.close()
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ Complete! Processed {processed} products")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()
    
    ingest_streaming(args.jsonl)
