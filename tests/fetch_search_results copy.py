from typing import Dict, Any, Optional
import time
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection
from openai import AzureOpenAI
import spacy
import re
from symspellpy import SymSpell
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = "amazon_fashion"
EMBEDDING_DIM = 1536

# Pagination limits
MAX_PAGE_SIZE = 500
MAX_TOTAL_RESULTS = 10000  # Maximum results to fetch from Milvus

azure_client = AzureOpenAI(
    api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT")
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_AZURE_OPENAI_MODEL")

# ==========================================
# GLOBAL SINGLETONS (Lazy Loading)
# ==========================================

_collection = None
_nlp = None
_sym_spell = None

def get_collection():
    """Lazy load Milvus collection"""
    global _collection
    if _collection is None:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        _collection = Collection(MILVUS_COLLECTION)
        _collection.load()
        logger.info(f"Milvus collection '{MILVUS_COLLECTION}' loaded")
    return _collection

def get_nlp():
    """Lazy load spaCy model"""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        logger.info("spaCy model loaded")
    return _nlp

def get_sym_spell():
    """Lazy load and build SymSpell dictionary"""
    global _sym_spell
    if _sym_spell is None:
        _sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        _build_dictionary(_sym_spell)
        logger.info("SymSpell dictionary built")
    return _sym_spell

# ==========================================
# KNOWLEDGE BASE
# ==========================================

GEO_DB = {
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
    "windbreaker", "overcoat", "jeans", "denim", "trousers", 
    "pants", "chinos", "leggings", "joggers", "shorts", "skirt", 
    "palazzo", "culottes", "trackpants", "saree", "sari", "lehenga", 
    "salwar", "suit", "anarkali", "kurta_set", "dhoti", "sherwani",
    "dress", "gown", "maxi", "midi", "mini", "frock", "jumpsuit", 
    "romper", "shoes", "sneakers", "boots", "sandals", "heels", 
    "flats", "slippers", "loafers", "trainers", "flipflops",
    "watch", "bag", "handbag", "backpack", "purse", "wallet", 
    "belt", "scarf", "stole", "shawl", "gloves", "socks", "cap", 
    "beanie", "sunglasses", "jewellery", "jewelry", "hat",
    "innerwear", "sleepwear", "nightwear", "activewear", 
    "sportswear", "clothes", "clothing", "outfit", "wear", "apparel"
}

STYLE_KEYWORDS = {
    "formal", "casual", "smart_casual", "ethnic", "party",
    "sporty", "athleisure", "streetwear", "boho", "vintage", 
    "chic", "minimalist", "classic", "retro", "luxury", "modest",
    "royal", "festive", "traditional", "modern", "trendy", 
    "elegant", "bold"
}

OCCASION_KEYWORDS = {
    "wedding", "reception", "engagement", "office", "work", 
    "meeting", "gym", "workout", "fitness", "date", "night out",
    "vacation", "holiday", "travel", "college", "school",
    "yoga", "running", "festival", "diwali", "eid", "christmas",
    "interview", "party", "celebration", "beach", "resort"
}

MATERIAL_KEYWORDS = {
    "cotton", "organic_cotton", "khadi", "silk", "raw_silk", 
    "wool", "merino", "leather", "faux_leather", "suede",
    "linen", "velvet", "polyester", "nylon", "acrylic",
    "denim", "satin", "fleece", "chiffon", "georgette",
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
# HELPER FUNCTIONS
# ==========================================

def _build_dictionary(sym_spell):
    """Build SymSpell dictionary with better coverage"""
    structural = {
        "under": 100000, "below": 100000, "above": 100000, "between": 100000,
        "for": 100000, "and": 100000, "in": 100000, "with": 100000,
        "but": 100000, "not": 100000,
        # Colors
        "black": 90000, "white": 90000, "red": 90000, "blue": 90000, 
        "green": 90000, "yellow": 90000, "pink": 90000, "purple": 90000,
        # Common verbs
        "fetch": 95000, "want": 95000, "need": 95000, "going": 95000,
        "looking": 95000, "search": 95000, "find": 95000, "get": 95000,
        # Pronouns
        "i": 100000, "me": 100000, "my": 100000
    }
    
    for w, f in structural.items():
        sym_spell.create_dictionary_entry(w, f)
    
    all_knowledge_sets = [
        CATEGORY_KEYWORDS, STYLE_KEYWORDS, OCCASION_KEYWORDS,
        MATERIAL_KEYWORDS, GENDER_MAP.keys(), GEO_DB.keys()
    ]
    
    for kw_set in all_knowledge_sets:
        for word in kw_set:
            sym_spell.create_dictionary_entry(word, 85000)

def clean_query_for_spellcheck(text):
    """Spell check but preserve important words"""
    sym_spell = get_sym_spell()
    words = text.split()
    corrected_words = []
    
    preserve_words = {
        "fetch", "me", "i", "want", "need", "looking", "search",
        "find", "get", "show", "give", "am", "going", "to", "skirt", "skirts"
    }
    
    for word in words:
        word_lower = word.lower()
        
        if any(char.isdigit() for char in word):
            corrected_words.append(word)
        elif word_lower in preserve_words:
            corrected_words.append(word)
        elif word_lower in CATEGORY_KEYWORDS or word_lower in GENDER_MAP or word_lower in GEO_DB:
            corrected_words.append(word)
        else:
            suggestions = sym_spell.lookup(word, verbosity=2, max_edit_distance=2)
            if suggestions and suggestions[0].distance <= 1:
                corrected_words.append(suggestions[0].term)
            else:
                corrected_words.append(word)
    
    return " ".join(corrected_words)

def get_geo_expansion(text_tokens):
    """Extract geo keywords"""
    geo_keywords = []
    text_set = set(text_tokens)
    for loc, keywords in GEO_DB.items():
        if loc in text_set:
            geo_keywords.extend(keywords[:2])
    return list(set(geo_keywords))

def process_search_query(raw_query: str):
    """Process user query and extract filters with improved handling"""
    nlp = get_nlp()
    
    # FIX 1: Preserve original query
    original_query = raw_query.strip()
    
    # Spell check
    improved_query = clean_query_for_spellcheck(raw_query.lower())
    doc = nlp(improved_query)
    
    result = {
        "suggested_query": original_query,
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
    
    for i, token in enumerate(doc):
        lemma = token.lemma_
        tokens.append(lemma)
        
        #  FIX 2: Better negation detection
        is_negated = False
        
        if any(c.dep_ == "neg" for c in token.children):
            is_negated = True
        
        if token.head.dep_ == "neg" or any(c.dep_ == "neg" for c in token.head.children):
            is_negated = True
        
        for j in range(max(0, i-2), i):
            if doc[j].lemma_ in ["not", "no", "never"]:
                if not any(doc[k].pos_ == "CCONJ" for k in range(j, i)):
                    is_negated = True
                    break
        
        if token.is_stop or token.pos_ in ["ADP", "DET", "PRON", "AUX", "VERB", "PART"]:
            continue
        
        if token.pos_ in ["NOUN", "ADJ", "PROPN"]:
            if lemma in CATEGORY_KEYWORDS:
                if is_negated:
                    result["exclude_keywords"].append(lemma)
                else:
                    result["category"].append(lemma)
                    result["include_keywords"].append(lemma)
            
            elif lemma in STYLE_KEYWORDS:
                if not is_negated:
                    result["style"].append(lemma)
                    result["include_keywords"].append(lemma)
            
            elif lemma in OCCASION_KEYWORDS:
                if not is_negated:
                    result["occasion"].append(lemma)
                    result["include_keywords"].append(lemma)
            
            elif lemma in MATERIAL_KEYWORDS:
                if is_negated:
                    result["exclude_keywords"].append(lemma)
                else:
                    result["material"].append(lemma)
                    result["include_keywords"].append(lemma)
            
            elif lemma in GENDER_MAP and not is_negated:
                detected_genders.add(GENDER_MAP[lemma])
            
            elif lemma == "family":
                result["family_friendly"] = True
            
            elif lemma in ["red", "blue", "green", "black", "white", "yellow", "pink", "purple"]:
                if is_negated:
                    result["exclude_keywords"].append(lemma)
                else:
                    result["include_keywords"].append(lemma)
    
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
    
    # FIX 3: Geo expansion only to include_keywords
    geo_expansions = get_geo_expansion(tokens)
    result["include_keywords"].extend(geo_expansions)
    
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
    
    return result

def get_embedding(text: str):
    """Get embedding from Azure OpenAI"""
    response = azure_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

async def get_search_results(user_query: str, page: int, page_size: int) -> Dict[str, Any]:
    """
    FIXED: Main search function with proper pagination
    Fetches ALL results first, then paginates in-memory
    """
    start_time = time.time()
    logger.info(f"Processing search for query: '{user_query}' (page: {page}, size: {page_size})")
    
    # Validate and cap page_size
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    page = max(1, int(page))
    
    logger.info(f"Adjusted - Page: {page}, PageSize: {page_size}")
    
    collection = get_collection()
    processed = process_search_query(user_query)

    logger.info(f"Processed query: {processed}")
    
    # Get embedding for the improved query
    embedding_start = time.time()
    query_embedding = get_embedding(processed["improved_query"])
    logger.info(f"Embedding generation took {time.time() - embedding_start:.2f}s")
    
    # Build Milvus filter expression
    filter_conditions = []
    
    if processed["gender"] != "unisex":
        if processed["gender"] == "kids":
            filter_conditions.append("for_underage == True")
        else:
            filter_conditions.append(f'genders == "{processed["gender"]}"')
    
    if processed["price"]["min"] is not None:
        filter_conditions.append(f'price >= {processed["price"]["min"]}')
    if processed["price"]["max"] is not None:
        filter_conditions.append(f'price <= {processed["price"]["max"]}')
    
    filter_expr = " && ".join(filter_conditions) if filter_conditions else ""
    
    # FIX: Fetch ALL results up to MAX_TOTAL_RESULTS (like ChromaDB POC)
    search_limit = MAX_TOTAL_RESULTS
    ef_value = max(search_limit, 100)
    
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": ef_value}
    }
    
    output_fields = [
        "product_id", "title", "price", "average_rating", 
        "rating_number", "image", "store_name", "main_category",
        "genders", "styles", "occasions", "materials", "for_underage"
    ]
    
    logger.info(f"Fetching {search_limit} results from Milvus")
    search_start = time.time()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=search_limit,  #  Get ALL results
        expr=filter_expr if filter_expr else None,
        output_fields=output_fields
    )
    logger.info(f"Milvus search took {time.time() - search_start:.2f}s")
    
    # Extract ALL products first
    all_hits = results[0] if results else []
    
    all_products = []
    for hit in all_hits:
        all_products.append({
            "product_id": hit.entity.get("product_id"),
            "title": hit.entity.get("title"),
            "price": round(hit.entity.get("price", 0.0), 2),  #  Round to 2 decimals
            "average_rating": round(hit.entity.get("average_rating", 0.0), 1),  #  Round to 1 decimal
            "rating_number": hit.entity.get("rating_number"),
            "image": hit.entity.get("image"),
            "store_name": hit.entity.get("store_name"),
            "main_category": hit.entity.get("main_category"),
            "genders": hit.entity.get("genders"),
            "styles": hit.entity.get("styles"),
            "occasions": hit.entity.get("occasions"),
            "materials": hit.entity.get("materials"),
            "for_underage": hit.entity.get("for_underage"),
            "similarity_score": round(hit.score, 4)  #  Bonus: Round similarity score to 4 decimals
        })

    
    #  FIX: Calculate pagination BEFORE slicing (like ChromaDB POC)
    total_count = len(all_products)
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1
    has_next = page < total_pages
    has_prev = page > 1
    
    #  FIX: Apply pagination slicing
    offset = (page - 1) * page_size
    start_idx = offset
    end_idx = offset + page_size
    paginated_products = all_products[start_idx:end_idx]
    
    response_time = time.time() - start_time
    
    logger.info(f"Returning {len(paginated_products)} products for page {page}/{total_pages} (total: {total_count})")
    logger.info(f"Total search completed in {response_time:.2f}s")
    
    return {
        "status_code": 200,
        "response_time": response_time,
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
        "products": paginated_products,
        "total_count": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_next": has_next,
        "has_previous": has_prev
    }
