import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

# Database configurations
SOURCE_DB = "semantic_fashion_db_fl"

MYSQL_CONFIG = {
    "host": os.getenv("mysql_db_host", "localhost"),
    "port": int(os.getenv("mysql_db_port", 3306)),
    "user": os.getenv("mysql_db_user"),
    "password": os.getenv("mysql_db_password"),
    "autocommit": False
}


def column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table"""
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = %s 
        AND TABLE_NAME = %s 
        AND COLUMN_NAME = %s
    """, (SOURCE_DB, table_name, column_name))
    return cursor.fetchone()[0] > 0


def add_missing_columns():
    """Add missing columns to product_metadata table"""
    print("🔧 Adding missing columns to product_metadata...")

    conn = mysql.connector.connect(**MYSQL_CONFIG, database=SOURCE_DB)
    cursor = conn.cursor()

    try:
        # Define columns to add
        columns_to_add = [
            ("title", "VARCHAR(1000)", "parent_asin"),
            ("price", "DECIMAL(10,2)", "title"),
            ("average_rating", "DECIMAL(3,1)", "price"),
            ("rating_number", "INT", "average_rating"),
            ("image", "VARCHAR(1000)", "rating_number"),
            ("store_name", "VARCHAR(200)", "image"),
            ("main_category", "VARCHAR(100)", "store_name")
        ]
        
        added_count = 0
        
        for col_name, col_type, after_col in columns_to_add:
            if not column_exists(cursor, "product_metadata", col_name):
                print(f"  Adding column: {col_name}")
                cursor.execute(f"""
                    ALTER TABLE product_metadata
                    ADD COLUMN {col_name} {col_type} AFTER {after_col}
                """)
                conn.commit()
                added_count += 1
            else:
                print(f"  Column {col_name} already exists, skipping")
        
        print(f"\n✅ Added {added_count} new columns!")

        # Verify
        print("\n📋 Updated product_metadata schema:")
        cursor.execute("DESCRIBE product_metadata")
        for col in cursor.fetchall():
            print(f"  - {col[0]} ({col[1]})")

    except Exception as e:
        print(f"❌ Error adding columns: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def populate_product_metadata_fast():
    """Ultra-fast batch update using single SQL statement"""
    print("\n🚀 Starting fast data population...")

    conn = mysql.connector.connect(**MYSQL_CONFIG, database=SOURCE_DB)
    cursor = conn.cursor()

    try:
        # Check counts before update
        print("\n📊 Before update:")
        cursor.execute("SELECT COUNT(*) as count FROM product_metadata")
        meta_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) as count FROM products")
        prod_count = cursor.fetchone()[0]
        print(f"  product_metadata records: {meta_count}")
        print(f"  products records: {prod_count}")

        print("\n💾 Executing batch update...")

        cursor.execute("""
            UPDATE product_metadata pm
            INNER JOIN products p ON pm.product_id = p.product_id
            LEFT JOIN (
                SELECT product_id,
                       COALESCE(hi_res_url, large_url, thumb_url) as image_url
                FROM product_images
                WHERE image_order = 1
            ) img ON p.product_id = img.product_id
            SET
                pm.title = p.title,
                pm.price = p.price,
                pm.average_rating = p.average_rating,
                pm.rating_number = p.rating_number,
                pm.image = img.image_url,
                pm.store_name = p.store,
                pm.main_category = p.main_category
        """)

        conn.commit()
        affected = cursor.rowcount
        print(f"✅ Updated {affected} records in one shot!")

        # Verification
        print("\n🔍 Verification:")
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(title) as with_title,
                COUNT(CASE WHEN price > 0 THEN 1 END) as with_price,
                COUNT(average_rating) as with_rating,
                COUNT(rating_number) as with_rating_count,
                COUNT(CASE WHEN image != '' AND image IS NOT NULL THEN 1 END) as with_valid_image,
                COUNT(store_name) as with_store,
                COUNT(main_category) as with_category,
                AVG(price) as avg_price,
                AVG(average_rating) as avg_rating
            FROM product_metadata
        """)

        stats = cursor.fetchone()
        print(f"  Total records: {stats[0]}")
        print(f"  With title: {stats[1]}")
        print(f"  With price > 0: {stats[2]}")
        print(f"  With rating: {stats[3]}")
        print(f"  With rating count: {stats[4]}")
        print(f"  With valid image URL: {stats[5]}")
        print(f"  With store name: {stats[6]}")
        print(f"  With category: {stats[7]}")
        print(f"  Average price: ${stats[8]:.2f}" if stats[8] else "  Average price: N/A")
        print(f"  Average rating: {stats[9]:.2f}/5.0" if stats[9] else "  Average rating: N/A")

        # Sample check
        print("\n📋 Sample updated records:")
        cursor.execute("""
            SELECT
                product_id, parent_asin, title, price, average_rating,
                rating_number, store_name, main_category, image
            FROM product_metadata
            WHERE title IS NOT NULL
            LIMIT 5
        """)

        rows = cursor.fetchall()
        for row in rows:
            print(f"\n  • Product ID: {row[0]}")
            print(f"    ASIN: {row[1]}")
            print(f"    Title: {row[2][:60]}..." if row[2] else "    Title: None")
            print(f"    Price: ${row[3]}" if row[3] else "    Price: $0.00")
            print(f"    Rating: {row[4]}/5.0 ({row[5]} reviews)" if row[4] else "    Rating: None")
            print(f"    Store: {row[6]}" if row[6] else "    Store: None")
            print(f"    Category: {row[7]}" if row[7] else "    Category: None")
            print(f"    Image: {row[8][:70]}..." if row[8] else "    Image: None")

        # Check for NULL images
        print("\n📷 Image statistics:")
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN image IS NOT NULL AND image != '' THEN 1 END) as with_image,
                COUNT(CASE WHEN image IS NULL OR image = '' THEN 1 END) as without_image
            FROM product_metadata
        """)
        img_stats = cursor.fetchone()
        print(f"  Total products: {img_stats[0]}")
        print(f"  With images: {img_stats[1]} ({img_stats[1]/img_stats[0]*100:.1f}%)")
        print(f"  Without images: {img_stats[2]} ({img_stats[2]/img_stats[0]*100:.1f}%)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
        print("\n🔒 Database connections closed")


if __name__ == "__main__":
    # Step 1: Add missing columns
    add_missing_columns()

    print("\n" + "="*60)

    # Step 2: Populate data
    populate_product_metadata_fast()

    print("\n" + "="*60)
    print("✅ ALL DONE! product_metadata table is now fully populated.")
    print("="*60)
