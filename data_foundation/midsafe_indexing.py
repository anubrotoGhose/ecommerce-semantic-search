import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
import time

load_dotenv()

MILVUS_HOST = "10.169.101.75"
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = "amazon_fashion_catalog"

def safe_create_index():
    """
    Safely create index even while ingestion is running.
    This won't interfere with ongoing inserts.
    """
    print("Connecting to Milvus...")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    
    if not utility.has_collection(MILVUS_COLLECTION):
        print(f"❌ Collection '{MILVUS_COLLECTION}' does not exist!")
        return
    
    collection = Collection(MILVUS_COLLECTION)
    
    print(f"\n{'='*80}")
    print(f"Collection: {MILVUS_COLLECTION}")
    print(f"Total records: {collection.num_entities}")
    print(f"{'='*80}\n")
    
    # Check if index already exists
    if collection.has_index():
        print("✓ Index already exists. No action needed.")
        print("\n📝 Note: New records from ongoing ingestion will be")
        print("   automatically added to this existing index.\n")
        return
    
    # Create index
    print("Creating HNSW index on embedding field...")
    print("⏳ This may take a few minutes depending on data size...\n")
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    }
    
    start_time = time.time()
    
    try:
        # This is SAFE to run during ingestion
        collection.create_index(field_name="embedding", index_params=index_params)
        elapsed = time.time() - start_time
        
        print(f"✅ Index created successfully in {elapsed:.2f} seconds!")
        print(f"\n📊 Indexed records: {collection.num_entities}")
        print("\n📝 Important notes:")
        print("   • New records from ongoing ingestion will be auto-indexed")
        print("   • No need to rerun this script")
        print("   • Data is now searchable via vector similarity\n")
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return
    
    # Load collection into memory for searching
    print("Loading collection into memory...")
    try:
        collection.load()
        print("✅ Collection loaded and ready for search!\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not load collection: {e}")
        print("    You may need to run collection.load() later.\n")

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("SAFE MID-INGESTION INDEXING SCRIPT")
    print(f"{'='*80}\n")
    safe_create_index()
    print(f"{'='*80}\n")