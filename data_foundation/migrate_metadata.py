import os
import time
import mysql.connector
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = "amazon_fashion"

MYSQL_CONFIG = {
    "host": os.getenv("mysql_db_host", "localhost"),
    "port": int(os.getenv("mysql_db_port", 3306)),
    "user": os.getenv("mysql_db_user"),
    "password": os.getenv("mysql_db_password"),
    "database": os.getenv("mysql_db_name"),
    "autocommit": False
}

# Batch size for processing (adjust based on memory)
BATCH_SIZE = 1000

# ==========================================
# MYSQL TABLE CREATION
# ==========================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS product_metadata (
    product_id VARCHAR(255) PRIMARY KEY,
    title TEXT,
    price DECIMAL(10, 2),
    average_rating DECIMAL(3, 2),
    rating_number INT,
    image TEXT,
    store_name VARCHAR(500),
    main_category VARCHAR(255),
    genders VARCHAR(50),
    styles TEXT,
    occasions TEXT,
    materials TEXT,
    for_underage BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_price (price),
    INDEX idx_genders (genders),
    INDEX idx_rating (average_rating),
    INDEX idx_category (main_category),
    INDEX idx_underage (for_underage)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

# ==========================================
# MILVUS DATA EXTRACTION
# ==========================================

def connect_milvus():
    """Connect to Milvus and load collection"""
    print(f"🔗 Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    
    if not utility.has_collection(MILVUS_COLLECTION):
        raise ValueError(f"Collection '{MILVUS_COLLECTION}' does not exist!")
    
    collection = Collection(MILVUS_COLLECTION)
    collection.load()
    
    print(f"✅ Connected to collection: {MILVUS_COLLECTION}")
    print(f"📊 Total entities: {collection.num_entities}")
    
    return collection


def get_collection_schema_fields(collection):
    """Get all non-embedding fields from collection schema"""
    schema = collection.schema
    
    # Get all field names except the embedding field
    output_fields = []
    for field in schema.fields:
        # Skip embedding/vector fields (typically FLOAT_VECTOR or BINARY_VECTOR)
        if field.dtype.name not in ['FLOAT_VECTOR', 'BINARY_VECTOR']:
            output_fields.append(field.name)
    
    print(f"📋 Fields to extract: {output_fields}")
    return output_fields


def extract_all_data_from_milvus(collection) -> List[Dict[str, Any]]:
    """
    Extract all metadata from Milvus collection (excluding embeddings)
    Using query() with pagination for large datasets
    """
    print("\n🔄 Extracting data from Milvus...")
    
    # Get fields excluding embeddings
    output_fields = get_collection_schema_fields(collection)
    
    all_data = []
    offset = 0
    
    # Use query_iterator for efficient pagination [web:52]
    try:
        # Method 1: Using query_iterator (Milvus 2.3+)
        print("📦 Using query_iterator for extraction...")
        iterator = collection.query_iterator(
            batch_size=BATCH_SIZE,
            expr="",  # Empty expression to get all records
            output_fields=output_fields
        )
        
        batch_count = 0
        while True:
            result = iterator.next()
            if not result:
                iterator.close()
                break
            
            all_data.extend(result)
            batch_count += 1
            print(f"  ✓ Batch {batch_count}: {len(result)} records (Total: {len(all_data)})")
        
    except AttributeError:
        # Method 2: Fallback to regular query() for older Milvus versions [web:53][web:55]
        print("📦 Using regular query() for extraction...")
        
        # Query all records with a condition that matches everything
        # Assuming product_id or id field exists
        primary_key_field = collection.schema.primary_field.name
        
        result = collection.query(
            expr=f"{primary_key_field} >= ''",  # Match all (adjust based on field type)
            output_fields=output_fields,
            limit=16384  # Milvus query limit
        )
        
        all_data = result
        print(f"  ✓ Extracted {len(all_data)} records")
    
    print(f"✅ Total records extracted: {len(all_data)}")
    return all_data


# ==========================================
# MYSQL DATA INSERTION
# ==========================================

def create_mysql_table(cursor):
    """Create MySQL table for product metadata"""
    print("\n🗄️  Creating MySQL table...")
    cursor.execute(CREATE_TABLE_SQL)
    print("✅ Table 'product_metadata' ready")


def insert_data_to_mysql(data: List[Dict[str, Any]], conn, cursor):
    """
    Batch insert data into MySQL using executemany for performance
    """
    if not data:
        print("⚠️  No data to insert")
        return
    
    print(f"\n💾 Inserting {len(data)} records into MySQL...")
    
    # Prepare INSERT statement
    insert_sql = """
    INSERT INTO product_metadata (
        product_id, title, price, average_rating, rating_number,
        image, store_name, main_category, genders, styles,
        occasions, materials, for_underage
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON DUPLICATE KEY UPDATE
        title = VALUES(title),
        price = VALUES(price),
        average_rating = VALUES(average_rating),
        rating_number = VALUES(rating_number),
        image = VALUES(image),
        store_name = VALUES(store_name),
        main_category = VALUES(main_category),
        genders = VALUES(genders),
        styles = VALUES(styles),
        occasions = VALUES(occasions),
        materials = VALUES(materials),
        for_underage = VALUES(for_underage),
        updated_at = CURRENT_TIMESTAMP
    """
    
    # Process in batches for performance [web:46][web:56]
    total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    
    start_time = time.time()
    
    for batch_num in range(total_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_end = min((batch_num + 1) * BATCH_SIZE, len(data))
        batch = data[batch_start:batch_end]
        
        # Prepare batch data
        batch_values = []
        for record in batch:
            # Handle array/list fields - convert to JSON string or comma-separated
            styles = ','.join(record.get('styles', [])) if isinstance(record.get('styles'), list) else record.get('styles', '')
            occasions = ','.join(record.get('occasions', [])) if isinstance(record.get('occasions'), list) else record.get('occasions', '')
            materials = ','.join(record.get('materials', [])) if isinstance(record.get('materials'), list) else record.get('materials', '')
            
            batch_values.append((
                record.get('product_id'),
                record.get('title'),
                record.get('price'),
                record.get('average_rating'),
                record.get('rating_number'),
                record.get('image'),
                record.get('store_name'),
                record.get('main_category'),
                record.get('genders'),
                styles,
                occasions,
                materials,
                record.get('for_underage', False)
            ))
        
        # Batch insert using executemany [web:46][web:58]
        cursor.executemany(insert_sql, batch_values)
        
        # Commit every batch
        conn.commit()
        
        print(f"  ✓ Batch {batch_num + 1}/{total_batches}: {len(batch)} records inserted")
    
    elapsed = time.time() - start_time
    print(f"✅ Inserted {len(data)} records in {elapsed:.2f}s ({len(data)/elapsed:.0f} records/sec)")


# ==========================================
# MAIN MIGRATION FUNCTION
# ==========================================

def migrate_milvus_to_mysql():
    """Main migration function"""
    print("=" * 60)
    print("🚀 MILVUS TO MYSQL METADATA MIGRATION")
    print("=" * 60)
    
    mysql_conn = None
    
    try:
        # Step 1: Connect to Milvus
        collection = connect_milvus()
        
        # Step 2: Extract data from Milvus (excluding embeddings)
        all_data = extract_all_data_from_milvus(collection)
        
        if not all_data:
            print("❌ No data extracted from Milvus. Exiting.")
            return
        
        # Step 3: Connect to MySQL
        print(f"\n🔗 Connecting to MySQL at {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}")
        mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = mysql_conn.cursor()
        print("✅ Connected to MySQL")
        
        # Step 4: Create table
        create_mysql_table(cursor)
        
        # Step 5: Insert data
        insert_data_to_mysql(all_data, mysql_conn, cursor)
        
        # Step 6: Verify
        cursor.execute("SELECT COUNT(*) FROM product_metadata")
        count = cursor.fetchone()[0]
        print(f"\n✅ Migration completed! Total records in MySQL: {count}")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        if mysql_conn:
            mysql_conn.rollback()
        raise
    
    finally:
        if mysql_conn:
            mysql_conn.close()
            print("🔒 MySQL connection closed")
        
        connections.disconnect("default")
        print("🔒 Milvus connection closed")


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def verify_migration():
    """Verify data integrity after migration"""
    print("\n" + "=" * 60)
    print("🔍 VERIFYING MIGRATION")
    print("=" * 60)
    
    try:
        # Connect to Milvus
        collection = connect_milvus()
        milvus_count = collection.num_entities
        
        # Connect to MySQL
        mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = mysql_conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM product_metadata")
        mysql_count = cursor.fetchone()[0]
        
        print(f"📊 Milvus total entities: {milvus_count}")
        print(f"📊 MySQL total records: {mysql_count}")
        
        if milvus_count == mysql_count:
            print("✅ Record counts match!")
        else:
            print(f"⚠️  Count mismatch: {abs(milvus_count - mysql_count)} records difference")
        
        # Sample data check
        cursor.execute("SELECT product_id, title, price FROM product_metadata LIMIT 5")
        sample = cursor.fetchall()
        
        print("\n📋 Sample MySQL records:")
        for record in sample:
            print(f"  • {record[0]}: {record[1][:50]}... (₹{record[2]})")
        
        cursor.close()
        mysql_conn.close()
        connections.disconnect("default")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    try:
        migrate_milvus_to_mysql()
        verify_migration()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Migration interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
