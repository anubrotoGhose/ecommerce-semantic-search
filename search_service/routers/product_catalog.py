from fastapi import APIRouter, HTTPException, Depends, status
from typing import Optional, Dict, Any
import mysql.connector
from contextlib import contextmanager
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from .utils.schemas import ProductDetailsResponse, ProductCardResponse
# ==========================================
# ROUTER SETUP
# ==========================================

router = APIRouter(
    tags=["Products"],
    responses={404: {"description": "Not found"}}
)

# ==========================================
# DATABASE CONFIGURATION
# ==========================================

DB_CONFIG = {
    "host": os.getenv("mysql_db_host", "localhost"),
    "port": int(os.getenv("mysql_db_port", 3306)),
    "user": os.getenv("mysql_db_user"),
    "password": os.getenv("mysql_db_password"),
    "database": os.getenv("mysql_db_name"),
    "autocommit": True
}

# ==========================================
# DATABASE HELPER FUNCTIONS
# ==========================================

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        yield conn
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}"
        )
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_product_basic(product_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch basic product information from fashion_products table
    
    Args:
        product_id: UUID string of the product
        
    Returns:
        Dictionary with product basic info or None if not found
    """
    query = """
    SELECT 
        product_id,
        parent_asin,
        title,
        price,
        average_rating,
        rating_number,
        image,
        store_name,
        main_category
    FROM fashion_products
    WHERE product_id = %s
    LIMIT 1
    """
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (product_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                # Convert None price to 0.0
                if result['price'] is None:
                    result['price'] = 0.0
                
                return result
            return None
            
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(e)}"
        )


def get_best_image(image_field: Optional[str]) -> str:
    """
    Extract the best quality image URL from the image field
    
    Args:
        image_field: Image field from database (can be URL or JSON)
        
    Returns:
        Image URL string or empty string if not available
    """
    if not image_field or image_field.strip() == "":
        return ""
    
    # If it's already a clean URL
    if image_field.startswith("http"):
        return image_field.strip()
    
    # If it's wrapped in brackets or quotes, clean it
    image_field = image_field.strip('[]"\'')
    
    # Return the cleaned URL
    return image_field if image_field.startswith("http") else ""



# ==========================================
# API ENDPOINTS
# ==========================================

@router.get(
    "/card/{product_id}",
    response_model=ProductCardResponse,
    summary="Get Product Card Information",
    description="Lightweight response for product listing/card with essential display information",
    status_code=status.HTTP_200_OK
)
async def product_card(
    product_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Retrieve lightweight product card information for listing pages.
    
    **Authentication Required:** Yes (Bearer Token)
    
    **Returns:**
    - Product ID (UUID)
    - Parent ASIN
    - Title
    - Price
    - Average rating
    - Number of ratings
    - Primary image URL
    - Store name
    - Main category
    
    **Use Cases:**
    - Product listing grids
    - Search results
    - Category pages
    - Recommendation widgets
    """
    
    # Validate product_id format (basic UUID validation)
    if not product_id or len(product_id) != 36:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid product_id format. Expected UUID format (36 characters)"
        )
    
    # Fetch product data
    prod = get_product_basic(product_id)
    
    if not prod:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with id '{product_id}' not found"
        )
    
    # Extract best image
    primary_image = get_best_image(prod.get("image"))
    
    # Build response
    return ProductCardResponse(
        id=prod["product_id"],
        parent_asin=prod.get("parent_asin"),
        title=prod["title"],
        price=round(float(prod["price"] or 0.0), 2),
        average_rating=round(float(prod["average_rating"] or 0.0), 1),
        rating_number=int(prod["rating_number"] or 0),
        primary_image=primary_image,
        store_name=prod.get("store_name"),
        main_category=prod.get("main_category")
    )


@router.get(
    "/details/{product_id}",
    response_model=ProductDetailsResponse,
    summary="Get Full Product Details",
    description="Complete product information including all metadata",
    status_code=status.HTTP_200_OK
)
async def product_details(
    product_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get comprehensive product information including styles, occasions, materials, etc.
    
    **Authentication Required:** Yes (Bearer Token)
    """
    
    if not product_id or len(product_id) != 36:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid product_id format"
        )
    
    query = """
    SELECT 
        product_id,
        parent_asin,
        title,
        description,
        features,
        store_name,
        main_category,
        price,
        average_rating,
        rating_number,
        image,
        seasons,
        styles,
        occasions,
        genders,
        ages,
        article_types,
        materials,
        colors,
        sizes,
        vibe,
        for_underage,
        created_at
    FROM fashion_products
    WHERE product_id = %s
    LIMIT 1
    """
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (product_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Product with id '{product_id}' not found"
                )
            
            # Convert comma-separated strings to lists
            result['styles'] = [s.strip() for s in (result.get('styles') or '').split(',') if s.strip()]
            result['occasions'] = [o.strip() for o in (result.get('occasions') or '').split(',') if o.strip()]
            result['materials'] = [m.strip() for m in (result.get('materials') or '').split(',') if m.strip()]
            result['colors'] = [c.strip() for c in (result.get('colors') or '').split(',') if c.strip()]
            result['sizes'] = [s.strip() for s in (result.get('sizes') or '').split(',') if s.strip()]
            
            # Clean up image
            result['image'] = get_best_image(result.get('image'))
            
            # Round numeric fields
            result['price'] = round(float(result['price'] or 0.0), 2)
            result['average_rating'] = round(float(result['average_rating'] or 0.0), 1)
            
            # Convert created_at to string
            if result.get('created_at'):
                result['created_at'] = str(result['created_at'])
            
            return ProductDetailsResponse(**result)
            
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


@router.get(
    "/store/{store_name}",
    summary="Get Products by Store",
    description="Retrieve all products from a specific store/brand",
    status_code=status.HTTP_200_OK
)
async def products_by_store(
    store_name: str,
    page: int = 1,
    page_size: int = 20,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get paginated list of products from a specific store.
    
    **Authentication Required:** Yes (Bearer Token)
    
    **Query Parameters:**
    - page: Page number (default: 1)
    - page_size: Items per page (default: 20, max: 100)
    """
    
    if page < 1 or page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid pagination parameters. Page >= 1, PageSize between 1-100"
        )
    
    offset = (page - 1) * page_size
    
    count_query = """
    SELECT COUNT(*) as total
    FROM fashion_products
    WHERE store_name = %s
    """
    
    data_query = """
    SELECT 
        product_id,
        parent_asin,
        title,
        price,
        average_rating,
        rating_number,
        image,
        store_name,
        main_category
    FROM fashion_products
    WHERE store_name = %s
    ORDER BY average_rating DESC, rating_number DESC
    LIMIT %s OFFSET %s
    """
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # Get total count
            cursor.execute(count_query, (store_name,))
            total_count = cursor.fetchone()['total']
            
            if total_count == 0:
                return {
                    "store_name": store_name,
                    "total_count": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                    "has_next": False,
                    "has_previous": False,
                    "products": []
                }
            
            # Get products
            cursor.execute(data_query, (store_name, page_size, offset))
            products = cursor.fetchall()
            cursor.close()
            
            # Clean up products
            for prod in products:
                prod['image'] = get_best_image(prod.get('image'))
                prod['price'] = round(float(prod['price'] or 0.0), 2)
                prod['average_rating'] = round(float(prod['average_rating'] or 0.0), 1)
            
            total_pages = (total_count + page_size - 1) // page_size
            
            return {
                "store_name": store_name,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1,
                "products": products
            }
            
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


@router.get(
    "/category/{category_name}",
    summary="Get Products by Category",
    description="Retrieve all products from a specific category",
    status_code=status.HTTP_200_OK
)
async def products_by_category(
    category_name: str,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "rating",  # rating, price_asc, price_desc
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get paginated list of products from a specific category.
    
    **Authentication Required:** Yes (Bearer Token)
    
    **Query Parameters:**
    - page: Page number (default: 1)
    - page_size: Items per page (default: 20, max: 100)
    - sort_by: Sort order (rating, price_asc, price_desc)
    """
    
    if page < 1 or page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid pagination parameters"
        )
    
    # Determine sort clause
    sort_clauses = {
        "rating": "average_rating DESC, rating_number DESC",
        "price_asc": "price ASC",
        "price_desc": "price DESC"
    }
    
    if sort_by not in sort_clauses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sort_by value. Must be one of: {', '.join(sort_clauses.keys())}"
        )
    
    offset = (page - 1) * page_size
    
    count_query = """
    SELECT COUNT(*) as total
    FROM fashion_products
    WHERE main_category = %s
    """
    
    data_query = f"""
    SELECT 
        product_id,
        parent_asin,
        title,
        price,
        average_rating,
        rating_number,
        image,
        store_name,
        main_category
    FROM fashion_products
    WHERE main_category = %s
    ORDER BY {sort_clauses[sort_by]}
    LIMIT %s OFFSET %s
    """
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute(count_query, (category_name,))
            total_count = cursor.fetchone()['total']
            
            if total_count == 0:
                return {
                    "category_name": category_name,
                    "total_count": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                    "has_next": False,
                    "has_previous": False,
                    "products": []
                }
            
            cursor.execute(data_query, (category_name, page_size, offset))
            products = cursor.fetchall()
            cursor.close()
            
            for prod in products:
                prod['image'] = get_best_image(prod.get('image'))
                prod['price'] = round(float(prod['price'] or 0.0), 2)
                prod['average_rating'] = round(float(prod['average_rating'] or 0.0), 1)
            
            total_pages = (total_count + page_size - 1) // page_size
            
            return {
                "category_name": category_name,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "sort_by": sort_by,
                "has_next": page < total_pages,
                "has_previous": page > 1,
                "products": products
            }
            
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
