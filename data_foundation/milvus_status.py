import os
from dotenv import load_dotenv
from pymilvus import connections, Collection
import json

load_dotenv()

MILVUS_HOST = "10.169.101.73"
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = "amazon_fashion_products"

# Connect to Milvus
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(MILVUS_COLLECTION)

print(f"Collection: {MILVUS_COLLECTION}")
print(f"Description: {collection.description}")
print(f"Schema fields: {[field.name for field in collection.schema.fields]}")

# Check if index exists
has_index = collection.has_index()
print(f"Has index: {has_index}")

# Get total count (works without loading)
total_count = collection.num_entities
print(f"\n{'='*80}")
print(f"Total records in collection: {total_count}")
print(f"{'='*80}\n")

# All available fields (excluding embedding vector)
output_fields = [
    "product_id",       # Primary key (VARCHAR)
    "parent_asin",      # VARCHAR(32)
    "title",            # VARCHAR(512)
    "price",            # FLOAT
    "average_rating",   # FLOAT
    "rating_number",    # INT64
    "image",            # VARCHAR(1024)
    "store_name",       # VARCHAR(255)
    "main_category",    # VARCHAR(128)
    "styles",           # VARCHAR(255)
    "occasions",        # VARCHAR(255)
    "genders",          # VARCHAR(64)
    "materials",        # VARCHAR(255)
    "for_underage"      # BOOL
]

if total_count > 0:
    # Try to load collection (only if index exists)
    if has_index:
        try:
            print("Loading collection into memory...")
            collection.load()
            print("✓ Collection loaded successfully\n")
        except Exception as e:
            print(f"⚠ Could not load collection: {e}")
            print("Note: You can still query without loading, but searches won't work.\n")
    else:
        print("⚠ Index not found. Collection is still being ingested or index hasn't been created yet.")
        print("You can view data, but vector search is not available.\n")
    
    # First 5 records
    print("=" * 80)
    print("FIRST 5 RECORDS:")
    print("=" * 80)
    
    try:
        first_5 = collection.query(
            expr="product_id != ''",  # Match all non-empty product_ids
            output_fields=output_fields,
            limit=5
        )
        
        for idx, item in enumerate(first_5, 1):
            print(f"\n[{idx}] Product ID: {item['product_id']}")
            print(json.dumps(item, indent=2, default=str))
    except Exception as e:
        print(f"Error querying records: {e}")
    
    # Last 5 records
    if total_count >= 5:
        print("\n" + "=" * 80)
        print("LAST 5 RECORDS:")
        print("=" * 80)
        
        try:
            # Get all product_ids first, then query the last 5
            all_ids = collection.query(
                expr="product_id != ''",
                output_fields=["product_id"],
                limit=total_count
            )
            
            if len(all_ids) >= 5:
                last_5_ids = [item["product_id"] for item in all_ids[-5:]]
                
                # Build expr with proper escaping
                ids_str = ", ".join([f'"{id}"' for id in last_5_ids])
                
                # Query the last 5 records
                last_5 = collection.query(
                    expr=f"product_id in [{ids_str}]",
                    output_fields=output_fields
                )
                
                for idx, item in enumerate(last_5, 1):
                    print(f"\n[{idx}] Product ID: {item['product_id']}")
                    print(json.dumps(item, indent=2, default=str))
        except Exception as e:
            print(f"Error querying last records: {e}")
    
    # Sample statistics
    print("\n" + "=" * 80)
    print("COLLECTION STATISTICS:")
    print("=" * 80)
    
    try:
        # Get some sample stats
        sample_size = min(100, total_count)
        sample = collection.query(
            expr="product_id != ''",
            output_fields=output_fields,
            limit=sample_size
        )
        
        if sample:
            # Calculate statistics
            prices = [item['price'] for item in sample if item['price'] > 0]
            ratings = [item['average_rating'] for item in sample if item['average_rating'] > 0]
            
            print(f"\nSample size: {len(sample)} records")
            
            if prices:
                print(f"\nPrice Statistics:")
                print(f"  - Average: ${sum(prices)/len(prices):.2f}")
                print(f"  - Min: ${min(prices):.2f}")
                print(f"  - Max: ${max(prices):.2f}")
            
            if ratings:
                print(f"\nRating Statistics:")
                print(f"  - Average: {sum(ratings)/len(ratings):.2f}")
                print(f"  - Min: {min(ratings):.2f}")
                print(f"  - Max: {max(ratings):.2f}")
            
            # Category distribution
            categories = {}
            genders = {}
            styles = {}
            for item in sample:
                cat = item.get('main_category', 'Unknown')
                gender = item.get('genders', 'Unknown')
                style = item.get('styles', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
                genders[gender] = genders.get(gender, 0) + 1
                styles[style] = styles.get(style, 0) + 1
            
            print(f"\nTop Categories (sample):")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {cat}: {count}")
            
            print(f"\nGender Distribution (sample):")
            for gender, count in sorted(genders.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {gender}: {count}")
            
            print(f"\nTop Styles (sample):")
            for style, count in sorted(styles.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {style}: {count}")
            
            # Check for underage products
            underage_count = sum(1 for item in sample if item.get('for_underage', False))
            print(f"\nUnderage products: {underage_count}/{len(sample)}")
            
            # Store distribution
            stores = {}
            for item in sample:
                store = item.get('store_name', 'Unknown')
                stores[store] = stores.get(store, 0) + 1
            
            print(f"\nTop Stores (sample):")
            for store, count in sorted(stores.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {store}: {count}")
    
    except Exception as e:
        print(f"Error calculating statistics: {e}")

else:
    print("\n⚠ No records found in collection!")
    print("The ingestion may still be in progress.")

# Release collection if it was loaded
try:
    collection.release()
    print("\n" + "=" * 80)
    print("✓ Collection released")
except:
    print("\n" + "=" * 80)
    print("✓ Script completed")
print("=" * 80)
