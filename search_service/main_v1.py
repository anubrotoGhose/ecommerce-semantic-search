from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection
from openai import AzureOpenAI
import spacy
import re
from symspellpy import SymSpell
import time
import logging
load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fashion Search API")

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# CONFIGURATION
# ==========================================

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

# Connect to Milvus
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(MILVUS_COLLECTION)
collection.load()

# ==========================================
# KNOWLEDGE BASE (Your existing data)
# ==========================================

GEO_DB = {
    # India
    "himachal": ["winter", "thermal", "cold", "layered"],
    "manali": ["snow", "waterproof", "insulated"],
    "kashmir": ["heavy wool", "pashmina", "cold", "winter"],
    "leh": ["extreme cold", "thermal", "down jacket"],
    "ladakh": ["extreme cold", "windproof", "thermal"],
    "goa": ["beach", "resort", "summer", "lightweight"],
    "kerala": ["cotton", "humid", "breathable"],
    "rajasthan": ["lightweight", "vibrant", "ethnic"],
    "jaipur": ["bandhani", "traditional", "festive"],
    "udaipur": ["royal", "ethnic", "wedding"],
    "mumbai": ["monsoon", "breathable", "humid"],
    "delhi": ["trendy", "urban", "layered"],
    "bangalore": ["casual", "comfortable"],
    "chennai": ["cotton", "summer", "humid"],
    "kolkata": ["traditional", "festive"],
    
    # International
    "paris": ["chic", "fashion", "formal"],
    "london": ["trench coat", "formal", "rainy"],
    "dubai": ["luxury", "summer", "modest"],
    "thailand": ["swimwear", "beach", "tropical"],
    "bali": ["resort", "tropical", "boho"],
    "maldives": ["resort", "swimwear", "luxury"],
    "canada": ["extreme winter", "parka", "snow"],
    "new york": ["urban", "streetwear"],
    "italy": ["luxury", "fashion", "tailored"]
}

CATEGORY_KEYWORDS = {
    "shirt", "shirts", "tshirt", "t-shirt", "tee", "top", "blouse",
    "tunic", "kurta", "kurti", "crop_top", "tank", "tanktop",
    "hoodie", "sweatshirt", "sweater", "cardigan",
    "blazer", "jacket", "coat", "parka", "windcheater",
    "windbreaker", "overcoat",
    "jeans", "denim", "trousers", "pants", "chinos",
    "leggings", "joggers", "shorts", "skirt", "palazzo",
    "culottes", "trackpants",
    "saree", "sari", "lehenga", "salwar", "suit",
    "anarkali", "kurta_set", "dhoti", "sherwani",
    "dress", "gown", "maxi", "midi", "mini",
    "frock", "jumpsuit", "romper",
    "shoes", "sneakers", "boots", "sandals",
    "heels", "flats", "slippers", "loafers",
    "trainers", "flipflops",
    "watch", "bag", "handbag", "backpack", "purse",
    "wallet", "belt", "scarf", "stole", "shawl",
    "gloves", "socks", "cap", "beanie",
    "sunglasses", "jewellery", "jewelry", "hat",
    "innerwear", "sleepwear", "nightwear",
    "activewear", "sportswear",
    "clothes", "clothing", "outfit", "wear", "apparel"
}

STYLE_KEYWORDS = {
    "formal", "casual", "smart_casual", "ethnic", "party",
    "sporty", "athleisure", "streetwear",
    "boho", "vintage", "chic", "minimalist",
    "classic", "retro", "luxury", "modest",
    "royal", "festive", "traditional",
    "modern", "trendy", "elegant", "bold"
}

OCCASION_KEYWORDS = {
    "wedding", "reception", "engagement",
    "office", "work", "meeting",
    "gym", "workout", "fitness",
    "date", "night out",
    "vacation", "holiday", "travel",
    "college", "school",
    "yoga", "running",
    "festival", "diwali", "eid", "christmas",
    "interview", "party", "celebration",
    "beach", "resort"
}

MATERIAL_KEYWORDS = {
    "cotton", "organic_cotton", "khadi",
    "silk", "raw_silk", "wool", "merino",
    "leather", "faux_leather", "suede",
    "linen", "velvet",
    "polyester", "nylon", "acrylic",
    "denim", "satin",
    "fleece", "chiffon", "georgette",
    "rayon", "viscose", "modal", "spandex"
}

GENDER_MAP = {
    "men": "male", "man": "male", "male": "male",
    "boy": "male", "boys": "male", "mens": "male",
    "gentlemen": "male", "gents": "male",
    "women": "female", "woman": "female", "female": "female",
    "girl": "female", "girls": "female",
    "ladies": "female", "womens": "female",
    "kid": "kids", "kids": "kids", "child": "kids",
    "children": "kids", "baby": "kids", "infant": "kids",
    "toddler": "kids",
    "unisex": "unisex", "all": "unisex", "everyone": "unisex"
}

PRICE_PATTERNS = [
    (re.compile(r"under\s?₹?\s?(\d+)", re.I), "max"),
    (re.compile(r"below\s?₹?\s?(\d+)", re.I), "max"),
    (re.compile(r"less than\s?₹?\s?(\d+)", re.I), "max"),
    (re.compile(r"above\s?₹?\s?(\d+)", re.I), "min"),
    (re.compile(r"over\s?₹?\s?(\d+)", re.I), "min"),
    (re.compile(r"between\s?₹?\s?(\d+)\s?(?:and|to)\s?₹?\s?(\d+)", re.I), "range")
]

# ==========================================
# INITIALIZATION
# ==========================================

nlp = spacy.load("en_core_web_sm", disable=["ner"])
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

def build_dictionary():
    structural = {
        "under": 100000, "below": 100000, "above": 100000, "between": 100000,
        "for": 100000, "and": 100000, "in": 100000, "with": 100000,
        "but": 100000, "not": 100000, "black": 50000, "white": 50000,
        "red": 50000, "blue": 50000, "green": 50000
    }
    for w, f in structural.items():
        sym_spell.create_dictionary_entry(w, f)
    
    all_knowledge_sets = [
        CATEGORY_KEYWORDS, STYLE_KEYWORDS, OCCASION_KEYWORDS,
        MATERIAL_KEYWORDS, GENDER_MAP.keys(), GEO_DB.keys()
    ]
    for kw_set in all_knowledge_sets:
        for word in kw_set:
            sym_spell.create_dictionary_entry(word, 80000)

build_dictionary()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_query_for_spellcheck(text):
    words = text.split()
    corrected_words = []
    for word in words:
        if any(char.isdigit() for char in word):
            corrected_words.append(word)
        else:
            suggestions = sym_spell.lookup(word, verbosity=2, max_edit_distance=2)
            if suggestions:
                corrected_words.append(suggestions[0].term)
            else:
                corrected_words.append(word)
    return " ".join(corrected_words)

def get_geo_expansion(text_tokens):
    geo_keywords = []
    text_set = set(text_tokens)
    for loc, keywords in GEO_DB.items():
        if loc in text_set:
            geo_keywords.extend(keywords[:2])
    return list(set(geo_keywords))

def remove_excluded_from_query(query: str, exclude_keywords: list):
    doc = nlp(query)
    cleaned_tokens = []
    exclude_set = set(exclude_keywords)
    for token in doc:
        if token.lemma_ not in exclude_set:
            cleaned_tokens.append(token.text)
    return " ".join(cleaned_tokens)

def process_search_query(raw_query: str):
    improved_query = clean_query_for_spellcheck(raw_query.lower())
    doc = nlp(improved_query)
    
    result = {
        "suggested_query": improved_query,
        "improved_query": improved_query,
        "include_keywords": [],
        "exclude_keywords": [],
        "price": {"min": None, "max": None},
        "gender": "unisex",
        "style": [],
        "occasion": [],
        "material": [],
        "category": [],
        "family_friendly": False,
        "child_friendly": False
    }
    
    tokens = []
    detected_genders = set()
    
    for token in doc:
        lemma = token.lemma_
        tokens.append(lemma)
        
        is_negated = any(c.dep_ == "neg" for c in token.children) or \
                     any(c.dep_ == "neg" for c in token.head.children)
        
        if token.pos_ in ["NOUN", "ADJ", "PROPN"] and not token.is_stop:
            if lemma in CATEGORY_KEYWORDS:
                (result["exclude_keywords"] if is_negated else result["category"]).append(lemma)
                if not is_negated:
                    result["include_keywords"].append(lemma)
            
            elif lemma in STYLE_KEYWORDS:
                result["style"].append(lemma)
                result["include_keywords"].append(lemma)
            
            elif lemma in OCCASION_KEYWORDS:
                result["occasion"].append(lemma)
            
            elif lemma in MATERIAL_KEYWORDS:
                (result["exclude_keywords"] if is_negated else result["material"]).append(lemma)
                if not is_negated:
                    result["include_keywords"].append(lemma)
            
            elif lemma in GENDER_MAP and not is_negated:
                detected_genders.add(GENDER_MAP[lemma])
            
            elif lemma == "family":
                result["family_friendly"] = True
            
            else:
                (result["exclude_keywords"] if is_negated else result["include_keywords"]).append(lemma)
    
    # Gender resolution
    if "kids" in detected_genders:
        result["gender"] = "kids"
        result["child_friendly"] = True
    elif "male" in detected_genders and "female" in detected_genders:
        result["gender"] = "unisex"
    elif "male" in detected_genders:
        result["gender"] = "male"
    elif "female" in detected_genders:
        result["gender"] = "female"
    
    # Geo expansion
    result["include_keywords"].extend(get_geo_expansion(tokens))
    
    # Price extraction
    for p, t in PRICE_PATTERNS:
        m = p.search(improved_query)
        if m:
            if t == "max":
                result["price"]["max"] = int(m.group(1))
            elif t == "min":
                result["price"]["min"] = int(m.group(1))
            else:
                result["price"]["min"] = int(m.group(1))
                result["price"]["max"] = int(m.group(2))
            break
    
    result["include_keywords"] = list(set(result["include_keywords"]) - set(result["exclude_keywords"]))
    result["exclude_keywords"] = list(set(result["exclude_keywords"]))
    
    if result["exclude_keywords"]:
        result["improved_query"] = remove_excluded_from_query(
            result["improved_query"],
            result["exclude_keywords"]
        )
    
    return result

def get_embedding(text: str):
    """Get embedding from Azure OpenAI"""
    response = azure_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


class SearchRequest(BaseModel):
    user_query: str
    ratings: Optional[float] = None
    price: Optional[float] = None
    gender: Optional[str] = None
    brand: Optional[str] = None
    page: Optional[int] = 1
    page_size: Optional[int] = 20

class ProductResponse(BaseModel):
    product_id: str
    title: str
    price: float
    average_rating: float
    rating_number: int
    image: str
    store_name: str
    main_category: str
    genders: str
    styles: str
    occasions: str
    materials: str
    for_underage: bool

@app.get("/")
def read_root():
    return {"message": "Hello from Secure HTTPS!"}


async def get_search_results(user_query: str, page: int, page_size: int) -> Dict[str, Any]:
    # This function is now integrated into the main search endpoint
    processed = process_search_query(user_query)
        
        # Get embedding for the improved query
    query_embedding = get_embedding(processed["improved_query"])
    
    # Build Milvus filter expression
    filter_conditions = []
    
    # Gender filter
    if processed["gender"] != "unisex":
        if processed["gender"] == "kids":
            filter_conditions.append("for_underage == True")
        else:
            filter_conditions.append(f'genders == "{processed["gender"]}"')
    
    # Price filter
    if processed["price"]["min"] is not None:
        filter_conditions.append(f'price >= {processed["price"]["min"]}')
    if processed["price"]["max"] is not None:
        filter_conditions.append(f'price <= {processed["price"]["max"]}')
    
    # Combine filters
    filter_expr = " && ".join(filter_conditions) if filter_conditions else ""
    
    # Calculate pagination
    offset = (page - 1) * page_size
    limit = page_size
    search_limit = offset + limit
    ef_value = max(search_limit, 100)
    # Search in Milvus
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": ef_value}  # 🔧 FIXED: Dynamic ef based on limit
    }
    
    output_fields = [
        "product_id", "title", "price", "average_rating", 
        "rating_number", "image", "store_name", "main_category",
        "genders", "styles", "occasions", "materials", "for_underage"
    ]
    
    # Perform vector search
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=search_limit,  # Fetch more to handle offset
        expr=filter_expr if filter_expr else None,
        output_fields=output_fields
    )
    
    # Extract products with pagination
    all_hits = results[0] if results else []
    paginated_hits = all_hits[offset:offset + limit]
    
    products = []
    for hit in paginated_hits:
        products.append({
            "product_id": hit.entity.get("product_id"),
            "title": hit.entity.get("title"),
            "price": hit.entity.get("price"),
            "average_rating": hit.entity.get("average_rating"),
            "rating_number": hit.entity.get("rating_number"),
            "image": hit.entity.get("image"),
            "store_name": hit.entity.get("store_name"),
            "main_category": hit.entity.get("main_category"),
            "genders": hit.entity.get("genders"),
            "styles": hit.entity.get("styles"),
            "occasions": hit.entity.get("occasions"),
            "materials": hit.entity.get("materials"),
            "for_underage": hit.entity.get("for_underage"),
            "similarity_score": hit.score
        })
    
    total_count = len(all_hits)
    total_pages = (total_count + page_size - 1) // page_size
    
    logger.info(f"Search completed in {end_time - start_time:.2f}")
    return {
        "status_code": 200,
        "response_time": time.time() - start_time,
        "suggested_query": processed["suggested_query"],
        "improved_query": processed["improved_query"],
        "filters_applied": {
            "gender": processed["gender"],
            "price": processed["price"],
            "child_friendly": processed["child_friendly"],
            "styles": processed["style"],
            "occasions": processed["occasion"],
            "materials": processed["material"]
        },
        "products": products,
        "total_count": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }
    


@app.post("/search", tags=["search"])
async def search_main(request: SearchRequest, background_tasks: BackgroundTasks):
    try:
        start_time = time.time()
        user_query = request.user_query.strip()
        
        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        end_time = time.time()
        logger.info(f"Total Query processing completed in {end_time - start_time:.2f} seconds")
        return await get_search_results(user_query, request.page, request.page_size)
        
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    