from typing import Dict, Any, Optional, List
import time
import os
import json
import pickle
import hashlib
import numpy as np
from dotenv import load_dotenv
from pymilvus import connections, Collection
from openai import AzureOpenAI
import spacy
import re
from symspellpy import SymSpell
import logging
import redis
import threading
# from sentence_transformers import CrossEncoder

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
MILVUS_COLLECTION = "amazon_fashion_products"
EMBEDDING_DIM = 1536


# Pagination limits
MAX_PAGE_SIZE = 500
MAX_TOTAL_RESULTS = 10000


# Cache configuration
CACHE_TTL = 3600  # 1 hour
SEMANTIC_SIMILARITY_THRESHOLD = 0.96
CACHE_ENABLED = True


azure_client = AzureOpenAI(
    api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT")
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_AZURE_OPENAI_MODEL")


# ==========================================
# REDIS SETUP
# ==========================================


_redis_client = None


def get_redis_client():
    """Lazy load Redis client with fallback handling"""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=False,  # Binary mode for embeddings
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            _redis_client.ping()
            logger.info("✅ Redis connection established")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"❌ Redis not available: {e}. Caching disabled.")
            _redis_client = None
    return _redis_client


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


cross_encoder_model = None


# def get_cross_encoder():
#     """Lazy load cross-encoder model for reranking."""
#     global cross_encoder_model
#     if cross_encoder_model is None:
#         cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
#         logger.info("✅ Cross-encoder model loaded: ms-marco-MiniLM-L6-v2")
#     return cross_encoder_model


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


# Extended color list
COLOR_KEYWORDS = {
    "red", "blue", "green", "black", "white", "yellow", "pink", "purple",
    "orange", "brown", "grey", "gray", "beige", "navy", "maroon", "teal",
    "olive", "gold", "silver", "copper", "bronze", "cream", "ivory",
    "magenta", "cyan", "indigo", "violet", "lime", "mint", "coral", "peach"
}

BRAND_KEYWORDS = {
    # Sports brands
    "nike", "adidas", "puma", "reebok", "under armour", "new balance",
    "asics", "fila", "skechers", "converse", "vans",
    
    # Luxury brands
    "gucci", "prada", "versace", "armani", "burberry", "dior",
    "chanel", "louis vuitton", "hermes", "balenciaga", "fendi",
    "givenchy", "valentino", "ysl", "saint laurent",
    
    # Fashion brands
    "zara", "h&m", "gap", "uniqlo", "forever 21", "mango",
    "levi's", "levis", "wrangler", "tommy hilfiger", "calvin klein",
    "polo", "ralph lauren", "lacoste", "diesel",
    
    # Indian brands
    "fabindia", "biba", "w", "aurelia", "manyavar", "raymond",
    "peter england", "allen solly", "van heusen", "louis philippe",
    
    # Athletic
    "champion", "umbro", "kappa", "lotto", "diadora",
}

# Quality-related keywords that trigger reranking
QUALITY_KEYWORDS = {
    # Quality indicators
    "quality", "good", "best", "top", "premium", "excellent", "superior",
    "high-quality", "finest", "great", "amazing", "perfect", "outstanding",
    
    # Trust indicators
    "trustworthy", "trusted", "reliable", "authentic", "genuine", "original",
    "verified", "certified", "legit", "reputable",
    
    # Popularity indicators
    "popular", "trending", "bestseller", "best-seller", "top-rated", 
    "highly-rated", "recommended", "favorite", "loved",
    
    # Value indicators
    "worth", "value", "deal", "bargain"
}


# ==========================================
# CACHING UTILITIES
# ==========================================


def _generate_cache_key(query: str, page: int, page_size: int) -> str:
    """Generate deterministic cache key for exact query match"""
    normalized_query = query.lower().strip()
    key_str = f"search:{normalized_query}:p{page}:ps{page_size}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _generate_filter_cache_key(improved_query: str) -> str:
    """Generate cache key for processed filters"""
    normalized = improved_query.lower().strip()
    return f"filters:{hashlib.md5(normalized.encode()).hexdigest()}"


def _generate_embedding_cache_key(improved_query: str) -> str:
    """Generate cache key for query embeddings"""
    normalized = improved_query.lower().strip()
    return f"embedding:{hashlib.md5(normalized.encode()).hexdigest()}"


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def _check_semantic_cache(embedding: list, processed_filters: Dict, redis_client) -> Optional[Dict]:
    """Check for semantically similar cached queries with filter validation"""
    try:
        embedding_keys = redis_client.keys("embedding:*")
        
        if not embedding_keys:
            return None
        
        query_vec = np.array(embedding, dtype=np.float32)
        best_match = None
        best_similarity = 0.0
        
        for key in embedding_keys[:100]:
            try:
                cached_embedding_bytes = redis_client.get(key)
                if cached_embedding_bytes:
                    cached_embedding = pickle.loads(cached_embedding_bytes)
                    cached_vec = np.array(cached_embedding, dtype=np.float32)
                    
                    similarity = _cosine_similarity(query_vec, cached_vec)
                    
                    if similarity > best_similarity and similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                        best_similarity = similarity
                        best_match = key
            except Exception as e:
                logger.debug(f"Error processing cached embedding {key}: {e}")
                continue
        
        if best_match:
            # Extract filter hash from embedding key and validate filters
            try:
                filter_hash = best_match.decode('utf-8').split(':')[1] if isinstance(best_match, bytes) else best_match.split(':')[1]
                filter_cache_key = f"filters:{filter_hash}"
                cached_filters = _cache_get(filter_cache_key, redis_client)
                
                if cached_filters and _filters_match(processed_filters, cached_filters):
                    logger.info(f"🎯 Semantic cache HIT! Similarity: {best_similarity:.4f} + Filters MATCH ✅")
                    return {"similarity": best_similarity, "embedding_key": best_match}
                else:
                    logger.info(f"⚠️ Semantic similarity {best_similarity:.4f} but filters MISMATCH ❌")
                    return None
            except Exception as e:
                logger.warning(f"Filter validation failed: {e}")
                return None
        
        return None
        
    except Exception as e:
        logger.warning(f"Semantic cache check failed: {e}")
        return None


def _filters_match(current_filters: Dict, cached_filters: Dict) -> bool:
    """
    Compare two filter dictionaries for semantic cache validation
    Returns True only if all critical filters match
    """
    try:
        # 1. Gender must match exactly
        if current_filters.get("gender") != cached_filters.get("gender"):
            return False
        
        # 2. Child-friendly must match exactly
        if current_filters.get("child_friendly") != cached_filters.get("child_friendly"):
            return False
        
        # 3. Price range with 10% tolerance
        current_price = current_filters.get("price", {})
        cached_price = cached_filters.get("price", {})
        
        # Check min price
        if (current_price.get("min") is None) != (cached_price.get("min") is None):
            return False
        if current_price.get("min") is not None:
            current_min = current_price["min"]
            cached_min = cached_price["min"]
            tolerance = max(current_min, cached_min) * 0.1
            if abs(current_min - cached_min) > tolerance:
                return False
        
        # Check max price
        if (current_price.get("max") is None) != (cached_price.get("max") is None):
            return False
        if current_price.get("max") is not None:
            current_max = current_price["max"]
            cached_max = cached_price["max"]
            tolerance = max(current_max, cached_max) * 0.1
            if abs(current_max - cached_max) > tolerance:
                return False
        
        # 4. Exclude keywords must match exactly
        current_exclude = set(current_filters.get("exclude_keywords", []))
        cached_exclude = set(cached_filters.get("exclude_keywords", []))
        if current_exclude != cached_exclude:
            return False
        
        # 5. Include keywords need 50% overlap (if both exist)
        current_include = set(current_filters.get("include_keywords", []))
        cached_include = set(cached_filters.get("include_keywords", []))
        if current_include and cached_include:
            overlap = len(current_include & cached_include) / max(len(current_include), len(cached_include))
            if overlap < 0.5:
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Filter comparison failed: {e}")
        return False


def _cache_get(key: str, redis_client) -> Optional[Any]:
    """Safely get data from Redis cache"""
    try:
        if redis_client is None:
            return None
        data = redis_client.get(key)
        if data:
            return pickle.loads(data)
        return None
    except Exception as e:
        logger.warning(f"Cache GET failed for key {key}: {e}")
        return None


def _cache_set(key: str, value: Any, redis_client, ttl: int = CACHE_TTL) -> bool:
    """Safely set data in Redis cache"""
    try:
        if redis_client is None:
            return False
        redis_client.setex(key, ttl, pickle.dumps(value))
        return True
    except Exception as e:
        logger.warning(f"Cache SET failed for key {key}: {e}")
        return False


def _background_cache_remaining_pages(
    all_products: list,
    user_query: str,
    page_size: int,
    current_page: int,
    redis_client
):
    """Background thread to cache remaining page slices"""
    try:
        total_pages = (len(all_products) + page_size - 1) // page_size
        
        for page in range(current_page + 1, total_pages + 1):
            offset = (page - 1) * page_size
            paginated_products = all_products[offset:offset + page_size]
            
            cache_key = _generate_cache_key(user_query, page, page_size)
            
            cached_response = {
                "products": paginated_products,
                "total_count": len(all_products),
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
            
            _cache_set(cache_key, cached_response, redis_client)
            logger.debug(f"🔄 Background cached page {page}/{total_pages}")
        
        logger.info(f"✅ Background caching completed for {total_pages - current_page} pages")
        
    except Exception as e:
        logger.error(f"Background caching failed: {e}")


# ==========================================
# HELPER FUNCTIONS
# ==========================================


def _build_dictionary(sym_spell):
    """Build SymSpell dictionary with better coverage"""
    structural = {
        "under": 100000, "below": 100000, "above": 100000, "between": 100000,
        "for": 100000, "and": 100000, "in": 100000, "with": 100000,
        "but": 100000, "not": 100000,
        "black": 90000, "white": 90000, "red": 90000, "blue": 90000, 
        "green": 90000, "yellow": 90000, "pink": 90000, "purple": 90000,
        "fetch": 95000, "want": 95000, "need": 95000, "going": 95000,
        "looking": 95000, "search": 95000, "find": 95000, "get": 95000,
        "i": 100000, "me": 100000, "my": 100000
    }
    
    for w, f in structural.items():
        sym_spell.create_dictionary_entry(w, f)
    
    all_knowledge_sets = [
        CATEGORY_KEYWORDS, STYLE_KEYWORDS, OCCASION_KEYWORDS,
        MATERIAL_KEYWORDS, GENDER_MAP.keys(), GEO_DB.keys(), 
        COLOR_KEYWORDS, BRAND_KEYWORDS  # ✅ ADDED
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
        elif (word_lower in CATEGORY_KEYWORDS or word_lower in GENDER_MAP or 
              word_lower in GEO_DB or word_lower in COLOR_KEYWORDS or
              word_lower in BRAND_KEYWORDS):  # ✅ ADDED
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
    """Process user query with comprehensive negation handling"""
    nlp = get_nlp()
    
    original_query = raw_query.strip()
    
    # STEP 1: Spell check (this becomes suggested_query)
    spell_checked_query = clean_query_for_spellcheck(raw_query.lower())
    
    doc = nlp(spell_checked_query)
    
    result = {
        "original_query": original_query,
        "suggested_query": spell_checked_query,  # Spell-checked, before exclusion removal
        "improved_query": spell_checked_query,   # Will be cleaned after exclusion removal
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
    
    # ==========================================
    # COMPREHENSIVE NEGATION PATTERNS
    # ==========================================
    
    negation_patterns = [
        # === DIRECT NEGATION ===
        (r'\b(?:not|no)\s+(\w+)', 'exclude'),
        (r'\b(?:don\'t|dont|do\s+not)\s+(?:want|need|like|show|give|get)\s+(\w+)', 'exclude'),
        (r'\b(?:doesn\'t|doesnt|does\s+not)\s+(?:want|need|like)\s+(\w+)', 'exclude'),
        (r'\b(?:won\'t|wont|will\s+not)\s+(?:wear|use|buy)\s+(\w+)', 'exclude'),
        (r'\b(?:can\'t|cant|cannot|can\s+not)\s+(?:wear|stand|tolerate)\s+(\w+)', 'exclude'),
        
        # === EXCEPT/EXCLUDING ===
        (r'\bexcept(?:ing)?\s+(?:for\s+)?(\w+)', 'exclude'),
        (r'\b(?:apart|aside)\s+from\s+(\w+)', 'exclude'),
        (r'\bother\s+than\s+(\w+)', 'exclude'),
        (r'\bsave\s+(?:for\s+)?(\w+)', 'exclude'),
        (r'\bbar(?:ring)?\s+(\w+)', 'exclude'),
        
        # === BUT PATTERNS ===
        (r'\bbut\s+(?:not|no)\s+(\w+)', 'exclude'),
        (r'\b(?:anything|everything|all)\s+but\s+(\w+)', 'exclude'),
        (r'\b(?:just|only)\s+not\s+(\w+)', 'exclude'),
        
        # === WITHOUT/LACKING ===
        (r'\bwithout\s+(?:any\s+)?(\w+)', 'exclude'),
        (r'\blacking\s+(\w+)', 'exclude'),
        (r'\bminus\s+(\w+)', 'exclude'),
        (r'\b(?:free|devoid)\s+of\s+(\w+)', 'exclude'),
        
        # === AVERSION/DISLIKE ===
        (r'\b(?:hate|dislike|despise|detest)\s+(\w+)', 'exclude'),
        (r'\b(?:avoid|skip|omit)\s+(\w+)', 'exclude'),
        (r'\b(?:refuse|reject)\s+(\w+)', 'exclude'),
        (r'\b(?:allergic|averse)\s+to\s+(\w+)', 'exclude'),
        
        # === PREFERENCE INVERSION ===
        (r'\b(?:prefer|rather)\s+(?:anything|everything)\s+(?:over|than)\s+(\w+)', 'exclude'),
        (r'\banything\s+(?:over|than|instead\s+of)\s+(\w+)', 'exclude'),
        
        # === NEGATIVE IMPERATIVES ===
        (r'\b(?:never|no\s+way)\s+(\w+)', 'exclude'),
        (r'\b(?:absolutely|definitely)\s+not\s+(\w+)', 'exclude'),
        
        # === COMPARATIVE NEGATION ===
        (r'\brather\s+than\s+(\w+)', 'exclude'),
        (r'\binstead\s+of\s+(\w+)', 'exclude'),
        (r'\bas\s+opposed\s+to\s+(\w+)', 'exclude'),
        
        # === MINIMIZERS (Negative Polarity Items) ===
        (r'\bnot\s+a\s+(?:single|bit\s+of)\s+(\w+)', 'exclude'),
        (r'\b(?:hardly|barely|scarcely)\s+any\s+(\w+)', 'exclude'),
        (r'\bnone\s+of\s+(?:the\s+)?(\w+)', 'exclude'),
        (r'\bneither\s+(\w+)', 'exclude'),
    ]
    
    # Track words to exclude
    words_to_remove = set()
    negation_phrases = set()
    
    # Apply rule-based patterns
    for pattern, action in negation_patterns:
        matches = re.finditer(pattern, spell_checked_query, re.IGNORECASE)
        for match in matches:
            excluded_word = match.group(1).lower()
            
            # Check if it's a known keyword
            if (excluded_word in CATEGORY_KEYWORDS or 
                excluded_word in STYLE_KEYWORDS or
                excluded_word in OCCASION_KEYWORDS or
                excluded_word in MATERIAL_KEYWORDS or
                excluded_word in COLOR_KEYWORDS or
                excluded_word in BRAND_KEYWORDS):  # ✅ ADDED
                
                if action == 'exclude':
                    if excluded_word not in result["exclude_keywords"]:
                        result["exclude_keywords"].append(excluded_word)
                    words_to_remove.add(excluded_word)
                    negation_phrases.add(match.group(0))
                    logger.info(f"🚫 Pattern detected: '{excluded_word}' from '{match.group(0)}'")
    
    # ==========================================
    # SPECIAL PATTERN: "X but not Y"
    # ==========================================
    
    complex_pattern = r'(\w+)\s+but\s+not\s+(\w+)'
    matches = re.finditer(complex_pattern, spell_checked_query, re.IGNORECASE)
    for match in matches:
        include_word = match.group(1).lower()
        exclude_word = match.group(2).lower()
        
        if (include_word in CATEGORY_KEYWORDS or include_word in COLOR_KEYWORDS):
            if include_word not in result["include_keywords"]:
                result["include_keywords"].append(include_word)
        
        if (exclude_word in CATEGORY_KEYWORDS or exclude_word in COLOR_KEYWORDS):
            if exclude_word not in result["exclude_keywords"]:
                result["exclude_keywords"].append(exclude_word)
            words_to_remove.add(exclude_word)
        
        logger.info(f"✨ Complex pattern: include '{include_word}', exclude '{exclude_word}'")
    
    # ==========================================
    # SPACY-BASED NEGATION (fallback)
    # ==========================================
    
    tokens = []
    detected_genders = set()
    
    for i, token in enumerate(doc):
        lemma = token.lemma_
        tokens.append(lemma)
        
        is_negated = False
        
        # Dependency-based negation
        if any(c.dep_ == "neg" for c in token.children):
            is_negated = True
        
        if token.head.dep_ == "neg" or any(c.dep_ == "neg" for c in token.head.children):
            is_negated = True
        
        # Lookback for negation
        for j in range(max(0, i-4), i):
            if doc[j].lemma_ in ["not", "no", "never", "except", "without", "excluding", "minus", "save", "bar"]:
                if not any(doc[k].pos_ == "CCONJ" for k in range(j, i)):
                    is_negated = True
                    break
        
        if token.is_stop or token.pos_ in ["ADP", "DET", "PRON", "AUX", "VERB", "PART"]:
            continue
        
        if token.pos_ in ["NOUN", "ADJ", "PROPN"]:
            if lemma in CATEGORY_KEYWORDS:
                if is_negated:
                    if lemma not in result["exclude_keywords"]:
                        result["exclude_keywords"].append(lemma)
                        words_to_remove.add(lemma)
                else:
                    if lemma not in result["category"]:
                        result["category"].append(lemma)
                    if lemma not in result["include_keywords"]:
                        result["include_keywords"].append(lemma)
            
            elif lemma in STYLE_KEYWORDS:
                if is_negated:
                    if lemma not in result["exclude_keywords"]:
                        result["exclude_keywords"].append(lemma)
                        words_to_remove.add(lemma)
                else:
                    if lemma not in result["style"]:
                        result["style"].append(lemma)
                    if lemma not in result["include_keywords"]:
                        result["include_keywords"].append(lemma)
            
            elif lemma in OCCASION_KEYWORDS:
                if is_negated:
                    if lemma not in result["exclude_keywords"]:
                        result["exclude_keywords"].append(lemma)
                        words_to_remove.add(lemma)
                else:
                    if lemma not in result["occasion"]:
                        result["occasion"].append(lemma)
                    if lemma not in result["include_keywords"]:
                        result["include_keywords"].append(lemma)
            
            elif lemma in MATERIAL_KEYWORDS:
                if is_negated:
                    if lemma not in result["exclude_keywords"]:
                        result["exclude_keywords"].append(lemma)
                        words_to_remove.add(lemma)
                else:
                    if lemma not in result["material"]:
                        result["material"].append(lemma)
                    if lemma not in result["include_keywords"]:
                        result["include_keywords"].append(lemma)
            
            elif lemma in GENDER_MAP and not is_negated:
                detected_genders.add(GENDER_MAP[lemma])
            
            elif lemma == "family":
                result["family_friendly"] = True
            
            elif lemma in COLOR_KEYWORDS:
                if is_negated:
                    if lemma not in result["exclude_keywords"]:
                        result["exclude_keywords"].append(lemma)
                        words_to_remove.add(lemma)
                else:
                    if lemma not in result["include_keywords"]:
                        result["include_keywords"].append(lemma)
    
    # Gender detection
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
    geo_expansions = get_geo_expansion(tokens)
    result["include_keywords"].extend(geo_expansions)
    
    # Price extraction
    for p, t in PRICE_PATTERNS:
        m = p.search(spell_checked_query)
        if m:
            if t == "max":
                result["price"]["max"] = int(m.group(1))
            elif t == "min":
                result["price"]["min"] = int(m.group(1))
            else:
                result["price"]["min"] = int(m.group(1))
                result["price"]["max"] = int(m.group(2))
            break
    
    # ✅ ADDED: Quality keyword detection
    for token in tokens:
        if token.lower() in QUALITY_KEYWORDS:
            result["quality_keywords_detected"] = True
            logger.info(f"🌟 Quality keyword detected: '{token}'")
            break
    
    # ==========================================
    # CLEAN UP KEYWORDS
    # ==========================================
    
    result["include_keywords"] = list(set(result["include_keywords"]) - set(result["exclude_keywords"]))
    result["exclude_keywords"] = list(set(result["exclude_keywords"]))
    
    # ==========================================
    # CLEAN IMPROVED QUERY (Remove excluded words and negation phrases)
    # ==========================================
    
    cleaned_query = spell_checked_query
    
    # Remove negation phrases
    for phrase in negation_phrases:
        cleaned_query = re.sub(r'\b' + re.escape(phrase) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
    
    # Remove comprehensive negation patterns
    negation_removals = [
        r'\b(?:don\'t|dont|do\s+not|doesn\'t|doesnt|does\s+not)\s+(?:want|need|like|show|give|get)\b',
        r'\b(?:won\'t|wont|will\s+not)\s+(?:wear|use|buy)\b',
        r'\b(?:can\'t|cant|cannot|can\s+not)\s+(?:wear|stand|tolerate)\b',
        r'\b(?:hate|dislike|despise|detest|avoid|skip|omit|refuse|reject)\b',
        r'\b(?:not|no)\s+',
        r'\bexcept(?:ing)?\s+(?:for\s+)?',
        r'\b(?:apart|aside)\s+from\s+',
        r'\bother\s+than\s+',
        r'\bsave\s+(?:for\s+)?',
        r'\bbar(?:ring)?\s+',
        r'\bbut\s+(?:not|no)\s+',
        r'\b(?:anything|everything|all)\s+but\s+',
        r'\b(?:just|only)\s+not\s+',
        r'\bwithout\s+(?:any\s+)?',
        r'\blacking\s+',
        r'\bminus\s+',
        r'\b(?:free|devoid)\s+of\s+',
        r'\b(?:allergic|averse)\s+to\s+',
        r'\b(?:prefer|rather)\s+(?:anything|everything)\s+(?:over|than)\s+',
        r'\banything\s+(?:over|than|instead\s+of)\s+',
        r'\b(?:never|no\s+way)\s+',
        r'\b(?:absolutely|definitely)\s+not\s+',
        r'\brather\s+than\s+',
        r'\binstead\s+of\s+',
        r'\bas\s+opposed\s+to\s+',
        r'\bnot\s+a\s+(?:single|bit\s+of)\s+',
        r'\b(?:hardly|barely|scarcely)\s+any\s+',
        r'\bnone\s+of\s+(?:the\s+)?',
        r'\bneither\s+',
    ]
    
    for pattern in negation_removals:
        cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
    
    # Remove excluded words
    for word in words_to_remove:
        cleaned_query = re.sub(r'\b' + re.escape(word) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
    
    # Clean spaces
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    
    # Fallback
    if len(cleaned_query) < 3:
        if result["category"]:
            cleaned_query = " ".join(result["category"])
        elif result["include_keywords"]:
            cleaned_query = " ".join(result["include_keywords"])
        else:
            cleaned_query = "fashion clothing"
    
    result["improved_query"] = cleaned_query
    
    logger.info(f"🔤 Original: '{original_query}'")
    logger.info(f"✅ Spell-checked (suggested): '{spell_checked_query}'")
    logger.info(f"✨ Cleaned (improved): '{cleaned_query}'")
    logger.info(f"📥 Include: {result['include_keywords']}, 🚫 Exclude: {result['exclude_keywords']}")
    
    return result


def _calculate_quality_score(product: Dict, max_rating: float = 5.0, rating_weight: float = 0.6) -> float:
    """
    Calculate a normalized quality score combining rating and review count
    
    Formula: quality_score = (normalized_rating * rating_weight) + (normalized_popularity * (1 - rating_weight))
    
    Args:
        product: Product dict with average_rating and rating_number
        max_rating: Maximum possible rating (default 5.0)
        rating_weight: Weight for rating vs popularity (0.0 to 1.0, default 0.6)
    
    Returns:
        Quality score between 0.0 and 1.0
    """
    avg_rating = product.get("average_rating", 0.0)
    rating_count = product.get("rating_number", 0)
    
    # Normalize rating (0 to 1)
    normalized_rating = avg_rating / max_rating if max_rating > 0 else 0.0
    
    # Normalize popularity using log scale (handles wide range of review counts)
    # Products with more reviews are more trusted
    if rating_count > 0:
        # Log scale: 1 review = 0, 10 reviews = 0.23, 100 reviews = 0.46, 1000+ reviews = 0.69+
        normalized_popularity = min(1.0, np.log10(rating_count + 1) / 4.0)
    else:
        normalized_popularity = 0.0
    
    # Combine with weights
    quality_score = (normalized_rating * rating_weight) + (normalized_popularity * (1 - rating_weight))
    
    # Apply Bayesian adjustment to penalize items with very few reviews
    # Products with <10 reviews get slightly penalized
    if rating_count < 10:
        confidence_penalty = rating_count / 10.0
        quality_score *= confidence_penalty
    
    return quality_score


# def _rerank_by_quality(products: list, distance_weight: float = 0.4, quality_weight: float = 0.6) -> list:
#     """
#     Rerank products by combining semantic similarity distance with quality score
    
#     Args:
#         products: List of product dicts with 'distance', 'average_rating', 'rating_number'
#         distance_weight: Weight for semantic similarity (0.0 to 1.0, default 0.4)
#         quality_weight: Weight for quality score (0.0 to 1.0, default 0.6)
    
#     Returns:
#         Reranked list of products sorted by combined score (descending)
#     """
#     reranking_time_start = time.time()
#     if not products:
#         return products
    
#     # Calculate quality scores for all products
#     for product in products:
#         product["quality_score"] = _calculate_quality_score(product)
    
#     # Normalize distances (lower L2 distance = better, so invert it)
#     distances = [p.get("distance", 0.0) for p in products]
#     max_distance = max(distances) if distances else 1.0
#     min_distance = min(distances) if distances else 0.0
#     distance_range = max_distance - min_distance if max_distance > min_distance else 1.0
    
#     for product in products:
#         # Invert and normalize: lower distance = higher score
#         raw_distance = product.get("distance", 0.0)
#         normalized_distance = 1.0 - ((raw_distance - min_distance) / distance_range)
#         product["normalized_distance"] = normalized_distance
        
#         # Combined score
#         product["combined_score"] = (
#             (normalized_distance * distance_weight) + 
#             (product["quality_score"] * quality_weight)
#         )
    
#     # Sort by combined score (descending)
#     reranked = sorted(products, key=lambda x: x.get("combined_score", 0.0), reverse=True)
    
#     logger.info(f"🔄 Reranked {len(products)} products by quality (top score: {reranked[0].get('combined_score', 0):.4f})")
#     logger.info(f"Reranking took {time.time() - reranking_time_start:.2f}s")
#     return reranked

# def _rerank_by_quality(
#     products: list, 
#     initial_top_k: int = 100,
#     distance_weight: float = 0.4, 
#     quality_weight: float = 0.6
# ) -> list:
#     """
#     Two-stage reranking: First select top candidates by distance, then rerank by quality.
    
#     This is more efficient than reranking all products, especially for large result sets.
    
#     Args:
#         products: List of product dicts with 'distance', 'average_rating', 'rating_number'
#         initial_top_k: Number of top products to select by distance (default: 100)
#         distance_weight: Weight for semantic similarity (0.0 to 1.0, default 0.4)
#         quality_weight: Weight for quality score (0.0 to 1.0, default 0.6)
    
#     Returns:
#         Reranked list of products sorted by combined score (descending)
#     """
#     reranking_time_start = time.time()
    
#     if not products:
#         return products
    
#     total_products = len(products)
    
#     # Stage 1: Select top K candidates by distance (lower distance = better)
#     # Sort by distance ascending (best matches first)
#     top_candidates = sorted(products, key=lambda x: x.get("distance", float('inf')))[:initial_top_k]
    
#     logger.info(f"📊 Stage 1: Selected top {len(top_candidates)} products from {total_products} by distance")
    
#     # Stage 2: Rerank top candidates by quality + distance
#     # Calculate quality scores for top candidates only
#     for product in top_candidates:
#         product["quality_score"] = _calculate_quality_score(product)
    
#     # Normalize distances within the top candidates
#     distances = [p.get("distance", 0.0) for p in top_candidates]
#     max_distance = max(distances) if distances else 1.0
#     min_distance = min(distances) if distances else 0.0
#     distance_range = max_distance - min_distance if max_distance > min_distance else 1.0
    
#     for product in top_candidates:
#         # Invert and normalize: lower distance = higher score
#         raw_distance = product.get("distance", 0.0)
#         normalized_distance = 1.0 - ((raw_distance - min_distance) / distance_range)
#         product["normalized_distance"] = normalized_distance
        
#         # Combined score
#         product["combined_score"] = (
#             (normalized_distance * distance_weight) + 
#             (product["quality_score"] * quality_weight)
#         )
    
#     # Sort by combined score (descending)
#     reranked = sorted(top_candidates, key=lambda x: x.get("combined_score", 0.0), reverse=True)
    
#     reranking_time = time.time() - reranking_time_start
    
#     logger.info(
#         f"🔄 Stage 2: Reranked {len(reranked)} products by quality "
#         f"(top score: {reranked[0].get('combined_score', 0):.4f})"
#     )
#     logger.info(f"✅ Total reranking took {reranking_time:.2f}s")
    
#     return reranked

def _rerank_by_quality(
    products: list, 
    initial_top_k: int = 100,
    min_rating_threshold: float = 3.5,
    distance_weight: float = 0.4, 
    quality_weight: float = 0.6
) -> list:
    """
    Two-stage reranking with quality filter: 
    1. Filter products by minimum rating threshold
    2. Select top candidates by distance
    3. Rerank by quality + distance
    
    Args:
        products: List of product dicts with 'distance', 'average_rating', 'rating_number'
        initial_top_k: Number of top products to select by distance (default: 100)
        min_rating_threshold: Minimum average rating required (default: 3.5)
        distance_weight: Weight for semantic similarity (0.0 to 1.0, default 0.4)
        quality_weight: Weight for quality score (0.0 to 1.0, default 0.6)
    
    Returns:
        Reranked list of products sorted by combined score (descending)
    """
    reranking_time_start = time.time()
    
    if not products:
        return products
    
    total_products = len(products)
    
    # Stage 0: Filter by minimum rating threshold
    quality_filtered = [
        p for p in products 
        if p.get("average_rating", 0.0) >= min_rating_threshold
    ]
    
    filtered_count = total_products - len(quality_filtered)
    
    if filtered_count > 0:
        logger.info(
            f"🌟 Quality Filter: Removed {filtered_count} products below {min_rating_threshold} stars "
            f"({len(quality_filtered)} remaining)"
        )
    
    # If no products meet the threshold, return empty or fall back to original
    if not quality_filtered:
        logger.warning(
            f"⚠️ No products meet {min_rating_threshold}+ star threshold! "
            f"Returning top {min(initial_top_k, total_products)} products by distance only."
        )
        # Fallback: return top products by distance without quality filtering
        return sorted(products, key=lambda x: x.get("distance", float('inf')))[:initial_top_k]
    
    # Stage 1: Select top K candidates by distance (lower distance = better)
    # Sort by distance ascending (best matches first)
    top_candidates = sorted(
        quality_filtered, 
        key=lambda x: x.get("distance", float('inf'))
    )[:initial_top_k]
    
    logger.info(
        f"📊 Stage 1: Selected top {len(top_candidates)} products "
        f"from {len(quality_filtered)} quality-filtered products"
    )
    
    # Stage 2: Rerank top candidates by quality + distance
    # Calculate quality scores for top candidates only
    for product in top_candidates:
        product["quality_score"] = _calculate_quality_score(product)
    
    # Normalize distances within the top candidates
    distances = [p.get("distance", 0.0) for p in top_candidates]
    max_distance = max(distances) if distances else 1.0
    min_distance = min(distances) if distances else 0.0
    distance_range = max_distance - min_distance if max_distance > min_distance else 1.0
    
    for product in top_candidates:
        # Invert and normalize: lower distance = higher score
        raw_distance = product.get("distance", 0.0)
        normalized_distance = 1.0 - ((raw_distance - min_distance) / distance_range)
        product["normalized_distance"] = normalized_distance
        
        # Combined score
        product["combined_score"] = (
            (normalized_distance * distance_weight) + 
            (product["quality_score"] * quality_weight)
        )
    
    # Sort by combined score (descending)
    reranked = sorted(top_candidates, key=lambda x: x.get("combined_score", 0.0), reverse=True)
    
    reranking_time = time.time() - reranking_time_start
    
    logger.info(
        f"🔄 Stage 2: Reranked {len(reranked)} products by quality "
        f"(top score: {reranked[0].get('combined_score', 0):.4f}, "
        f"rating: {reranked[0].get('average_rating', 0):.1f}★)"
    )
    logger.info(f"✅ Total reranking took {reranking_time:.2f}s")
    
    return reranked


#############################################

### Using Cross Encoder Model for Reranking (Not used in L2 version) ###
# cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

#############################################


def _rerank_with_cross_encoder(
    query: str, 
    products: List[Dict[str, Any]], 
    top_k: int = None,
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Rerank products using cross-encoder for semantic relevance.
    
    Cross-encoders analyze query-document pairs together, providing more
    accurate relevance scores than distance-based ranking alone.
    
    Args:
        query: User search query
        products: List of product dicts with 'title', 'distance', etc.
        top_k: Return only top K results (None = return all)
        batch_size: Batch size for cross-encoder inference (default: 32)
    
    Returns:
        Reranked list of products sorted by cross-encoder score (descending)
    
    Performance:
        - ~1800 docs/sec on V100 GPU
        - ~200-400 docs/sec on CPU
        - Achieves 74.30 NDCG@10 on TREC DL 19
    """
    reranking_start = time.time()
    
    if not products:
        return products
    
    # Get cross-encoder model
    model = get_cross_encoder()
    
    # Prepare query-document pairs for cross-encoder
    # Combine title + main_category for better context
    pairs = []
    for product in products:
        # Build rich text representation
        doc_text = f"{product.get('title', '')}"
        
        # Add category if available
        if product.get('maincategory'):
            doc_text += f" | {product['maincategory']}"
        
        # Add style/occasion context if available
        if product.get('styles'):
            doc_text += f" | {product['styles']}"
        
        pairs.append([query, doc_text])
    
    # Get cross-encoder scores (higher = more relevant)
    logger.info(f"🔄 Cross-encoder reranking {len(products)} products...")
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    
    # Attach scores to products
    for i, product in enumerate(products):
        product['cross_encoder_score'] = float(scores[i])
    
    # Sort by cross-encoder score (descending)
    reranked = sorted(products, key=lambda x: x.get('cross_encoder_score', 0.0), reverse=True)
    
    # Optionally limit to top_k results
    if top_k is not None:
        reranked = reranked[:top_k]
    
    reranking_time = time.time() - reranking_start
    
    logger.info(
        f"✅ Cross-encoder reranking completed in {reranking_time:.2f}s "
        f"(top score: {reranked[0].get('cross_encoder_score', 0):.4f})"
    )
    
    return reranked


def _rerank_hybrid(
    query: str,
    products: List[Dict[str, Any]],
    distance_weight: float = 0.3,
    quality_weight: float = 0.2,
    cross_encoder_weight: float = 0.5,
    top_k: int = None
) -> List[Dict[str, Any]]:
    """
    Hybrid reranking combining cross-encoder, distance, and quality scores.
    
    This provides the best of both worlds:
    - Cross-encoder for semantic relevance
    - Distance for embedding similarity
    - Quality for product trustworthiness
    
    Args:
        query: User search query
        products: List of product dictionaries
        distance_weight: Weight for semantic distance (default: 0.3)
        quality_weight: Weight for quality score (default: 0.2)
        cross_encoder_weight: Weight for cross-encoder score (default: 0.5)
        top_k: Return only top K results
    
    Returns:
        Reranked products sorted by hybrid score (descending)
    """
    reranking_start = time.time()
    
    if not products:
        return products
    
    # Validate weights sum to 1.0
    total_weight = distance_weight + quality_weight + cross_encoder_weight
    if abs(total_weight - 1.0) > 0.01:
        logger.warning(
            f"Weights sum to {total_weight:.2f}, normalizing to 1.0"
        )
        distance_weight /= total_weight
        quality_weight /= total_weight
        cross_encoder_weight /= total_weight
    
    # 1. Calculate quality scores
    for product in products:
        product["quality_score"] = _calculate_quality_score(product)
    
    # 2. Normalize distances (lower L2 distance = better, so invert)
    distances = [p.get("distance", 0.0) for p in products]
    max_distance = max(distances) if distances else 1.0
    min_distance = min(distances) if distances else 0.0
    distance_range = max_distance - min_distance if max_distance > min_distance else 1.0
    
    for product in products:
        raw_distance = product.get("distance", 0.0)
        normalized_distance = 1.0 - ((raw_distance - min_distance) / distance_range)
        product["normalized_distance"] = normalized_distance
    
    # 3. Get cross-encoder scores and normalize to [0, 1]
    model = get_cross_encoder()
    pairs = [[query, product.get('title', '')] for product in products]
    cross_scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    
    # Normalize cross-encoder scores using sigmoid (logits → probabilities)
    import numpy as np
    normalized_cross_scores = 1 / (1 + np.exp(-cross_scores))
    
    for i, product in enumerate(products):
        product['cross_encoder_score'] = float(normalized_cross_scores[i])
    
    # 4. Calculate hybrid score
    for product in products:
        product["hybrid_score"] = (
            (product["normalized_distance"] * distance_weight) +
            (product["quality_score"] * quality_weight) +
            (product["cross_encoder_score"] * cross_encoder_weight)
        )
    
    # 5. Sort by hybrid score
    reranked = sorted(products, key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
    
    if top_k is not None:
        reranked = reranked[:top_k]
    
    reranking_time = time.time() - reranking_start
    
    logger.info(
        f"🔥 Hybrid reranking completed in {reranking_time:.2f}s "
        f"({len(products)} products, top score: {reranked[0].get('hybrid_score', 0):.4f})"
    )
    
    return reranked


def get_embedding(text: str):
    """Get embedding from Azure OpenAI"""
    response = azure_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


# ==========================================
# MAIN SEARCH FUNCTION (L2 VERSION)
# ==========================================


async def get_search_results(user_query: str, page: int, page_size: int) -> Dict[str, Any]:
    """
    Main search function with multi-layer Redis caching (L2 Distance Version)
    """
    start_time = time.time()
    logger.info(f"Processing search for query: '{user_query}' (page: {page}, size: {page_size})")
    
    # Validate and cap pagination
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    page = max(1, int(page))
    
    redis_client = get_redis_client() if CACHE_ENABLED else None
    
    # ============================================
    # CACHE LAYER 1: Exact Query Cache
    # ============================================
    exact_cache_key = _generate_cache_key(user_query, page, page_size)
    
    if redis_client:
        try:
            cached_result = _cache_get(exact_cache_key, redis_client)
            if cached_result:
                logger.info(f"✅ CACHE HIT (Exact): {exact_cache_key}")
                response_time = time.time() - start_time
                cached_result["status_code"] = 200
                cached_result["response_time"] = response_time
                return cached_result
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
    
    # ============================================
    # PROCESS QUERY
    # ============================================
    collection = get_collection()
    
    filter_cache_key = _generate_filter_cache_key(user_query)
    processed = None
    
    if redis_client:
        try:
            processed = _cache_get(filter_cache_key, redis_client)
            if processed:
                logger.info(f"✅ FILTER CACHE HIT: {filter_cache_key}")
        except Exception as e:
            logger.warning(f"Filter cache lookup failed: {e}")
    
    if not processed:
        processed = process_search_query(user_query)
        if redis_client:
            _cache_set(filter_cache_key, processed, redis_client)
            logger.info(f"💾 Cached filters: {filter_cache_key}")
    
    logger.info(f"Processed query: {processed}")
    
    # ============================================
    # CACHE LAYER 2: Embedding Cache
    # ============================================
    embedding_cache_key = _generate_embedding_cache_key(processed["improved_query"])
    query_embedding = None
    
    if redis_client:
        try:
            cached_embedding = _cache_get(embedding_cache_key, redis_client)
            if cached_embedding:
                query_embedding = cached_embedding
                logger.info(f"✅ EMBEDDING CACHE HIT: {embedding_cache_key}")
        except Exception as e:
            logger.warning(f"Embedding cache lookup failed: {e}")
    
    # ============================================
    # CACHE LAYER 3: Semantic Similarity Cache
    # ============================================
    semantic_match = None
    if redis_client and not query_embedding:
        embedding_start = time.time()
        query_embedding = get_embedding(processed["improved_query"])
        logger.info(f"Embedding generation took {time.time() - embedding_start:.2f}s")
        
        semantic_match = _check_semantic_cache(query_embedding, processed, redis_client)

        
        if semantic_match:
            logger.info(f"Semantic match found but proceeding with new search for accuracy")
    
    if not query_embedding:
        embedding_start = time.time()
        query_embedding = get_embedding(processed["improved_query"])
        logger.info(f"Embedding generation took {time.time() - embedding_start:.2f}s")
    
    if redis_client and query_embedding:
        _cache_set(embedding_cache_key, query_embedding, redis_client)
        logger.info(f"💾 Cached embedding: {embedding_cache_key}")
    
    # ============================================
    # MILVUS SEARCH (L2 VERSION)
    # ============================================
    filter_conditions = []
    
    if processed["gender"] != "unisex":
        if processed["gender"] == "kids":
            filter_conditions.append("for_underage == True")
        else:
            filter_conditions.append(f'genders == "{processed["gender"]}"')
    
    if processed["price"]["min"] is not None or processed["price"]["max"] is not None:
        # Always exclude products with price <= 0 when price filter is active
        filter_conditions.append("price > 0")
        if processed["price"]["min"] is not None:
            filter_conditions.append(f'price >= {processed["price"]["min"]}')
        if processed["price"]["max"] is not None:
            filter_conditions.append(f'price <= {processed["price"]["max"]}')
    
    filter_expr = " && ".join(filter_conditions) if filter_conditions else ""
    
    search_limit = MAX_TOTAL_RESULTS
    ef_value = max(search_limit, 100)
    
    search_params = {
        "metric_type": "L2",
        "params": {"ef": ef_value}
    }
    
    output_fields = [
        "product_id", "title", "price", "average_rating", 
        "rating_number", "image", "store_name", "main_category",
        "genders", "styles", "occasions", "materials", "for_underage"
    ]
    
    logger.info(f"Fetching {search_limit} results from Milvus with L2 distance")
    search_start = time.time()
    
    try:
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=search_limit,
            expr=filter_expr if filter_expr else None,
            output_fields=output_fields
        )
        logger.info(f"Milvus search took {time.time() - search_start:.2f}s")
    except Exception as e:
        logger.error(f"Milvus search failed: {e}")
        return {
            "status_code": 500,
            "error": "Search service unavailable",
            "response_time": time.time() - start_time
        }
    
    # ============================================
    # PROCESS RESULTS
    # ============================================
        # ============================================
    # PROCESS RESULTS WITH PRICE VALIDATION
    # ============================================
    all_hits = results[0] if results else []
    
    # Check if price filtering was requested
    price_filter_active = (processed["price"]["min"] is not None or 
                          processed["price"]["max"] is not None)
    
    all_products = []
    filtered_count = 0  # Track how many products were filtered out
    
    for hit in all_hits:
        product_price = hit.entity.get("price", 0.0)
        
        # Skip products with invalid prices when price filter is active
        if price_filter_active:
            # Filter out None, 0, or negative prices
            if product_price is None or product_price <= 0:
                filtered_count += 1
                logger.debug(f"Filtered product {hit.entity.get('product_id')}: invalid price {product_price}")
                continue
            
            # Double-check price range (Milvus filter should handle this, but belt-and-suspenders)
            if processed["price"]["min"] is not None and product_price < processed["price"]["min"]:
                filtered_count += 1
                continue
            if processed["price"]["max"] is not None and product_price > processed["price"]["max"]:
                filtered_count += 1
                continue
        
        all_products.append({
            "product_id": hit.entity.get("product_id"),
            "title": hit.entity.get("title"),
            "price": round(product_price, 2),
            "average_rating": round(hit.entity.get("average_rating", 0.0), 1),
            "rating_number": hit.entity.get("rating_number"),
            "image": hit.entity.get("image"),
            "store_name": hit.entity.get("store_name"),
            "main_category": hit.entity.get("main_category"),
            "genders": hit.entity.get("genders"),
            "styles": hit.entity.get("styles"),
            "occasions": hit.entity.get("occasions"),
            "materials": hit.entity.get("materials"),
            "for_underage": hit.entity.get("for_underage"),
            "distance": round(hit.score, 4)
        })
    
    if filtered_count > 0:
        logger.info(f"🔍 Filtered out {filtered_count} products with invalid/out-of-range prices")
    
    reranking_applied = False
    if processed.get("quality_keywords_detected", False):
        logger.info("🌟 Applying quality-based reranking...")
        all_products = _rerank_by_quality(
            all_products, 
            distance_weight=0.5,  # 50% weight to semantic similarity
            quality_weight=0.5    # 50% weight to quality score
        )
        reranking_applied = True
    
    # ============================================
    # PAGINATION
    # ============================================
    total_count = len(all_products)
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1
    has_next = page < total_pages
    has_prev = page > 1
    
    offset = (page - 1) * page_size
    start_idx = offset
    end_idx = offset + page_size
    paginated_products = all_products[start_idx:end_idx]
    
    response_time = time.time() - start_time
    
    response_data = {
        "status_code": 200,
        "response_time": response_time,
        "original_query": processed["original_query"],
        "suggested_query": processed["suggested_query"],  # Spell-checked before exclusion
        "improved_query": processed["improved_query"], # Cleaned after exclusion removal
        "reranking_applied": reranking_applied,  # ✅ ADDED
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
    
    # ============================================
    # CACHE CURRENT PAGE
    # ============================================
    if redis_client:
        try:
            cache_data = {
                "products": paginated_products,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_previous": has_prev,
                "original_query": processed["original_query"],
                "suggested_query": processed["suggested_query"],
                "improved_query": processed["improved_query"],
                "reranking_applied": reranking_applied,  # ✅ ADDED
                "filters_applied": response_data["filters_applied"]
            }
            _cache_set(exact_cache_key, cache_data, redis_client)
            logger.info(f"💾 Cached current page result: {exact_cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache current result: {e}")
    
    # ============================================
    # BACKGROUND CACHE WARMING
    # ============================================
    if redis_client and total_pages > page:
        try:
            warming_thread = threading.Thread(
                target=_background_cache_remaining_pages,
                args=(all_products, user_query, page_size, page, redis_client),
                daemon=True
            )
            warming_thread.start()
            logger.info(f"🔥 Started background cache warming for {total_pages - page} remaining pages")
        except Exception as e:
            logger.warning(f"Failed to start background cache warming: {e}")
    
    logger.info(f"Returning {len(paginated_products)} products for page {page}/{total_pages} (total: {total_count})")
    logger.info(f"Total search completed in {response_time:.2f}s")
    
    return response_data
