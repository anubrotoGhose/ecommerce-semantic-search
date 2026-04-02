from typing import Dict, Any, Optional
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


def _check_semantic_cache(embedding: list, redis_client) -> Optional[Dict]:
    """Check for semantically similar cached queries"""
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
            logger.info(f"🎯 Semantic cache HIT! Similarity: {best_similarity:.4f}")
            return {"similarity": best_similarity, "embedding_key": best_match}
        
        return None
        
    except Exception as e:
        logger.warning(f"Semantic cache check failed: {e}")
        return None


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
        MATERIAL_KEYWORDS, GENDER_MAP.keys(), GEO_DB.keys(), COLOR_KEYWORDS
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
              word_lower in GEO_DB or word_lower in COLOR_KEYWORDS):
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
                excluded_word in COLOR_KEYWORDS):
                
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
        
        semantic_match = _check_semantic_cache(query_embedding, redis_client)
        
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
    all_hits = results[0] if results else []
    
    all_products = []
    for hit in all_hits:
        all_products.append({
            "product_id": hit.entity.get("product_id"),
            "title": hit.entity.get("title"),
            "price": round(hit.entity.get("price", 0.0), 2),
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
        "improved_query": processed["improved_query"],    # Cleaned after exclusion removal
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
