from typing import Optional
from pydantic import BaseModel, Field

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

class ProductDetailsResponse(BaseModel):
    """Response model for full product details"""
    product_id: str
    parent_asin: Optional[str]
    title: str
    description: Optional[str]
    features: Optional[str]
    store_name: Optional[str]
    main_category: Optional[str]
    price: float
    average_rating: float
    rating_number: int
    image: str
    seasons: Optional[str]
    styles: list[str]
    occasions: list[str]
    genders: Optional[str]
    ages: Optional[str]
    article_types: Optional[str]
    materials: list[str]
    colors: list[str]
    sizes: list[str]
    vibe: Optional[str]
    for_underage: bool
    created_at: Optional[str]

class ProductCardResponse(BaseModel):
    """Response model for product card endpoint"""
    id: str = Field(..., description="Product UUID")
    parent_asin: Optional[str] = Field(None, description="Parent ASIN identifier")
    title: str = Field(..., description="Product title")
    price: float = Field(..., description="Product price")
    average_rating: float = Field(..., description="Average rating (0-5)")
    rating_number: int = Field(..., description="Number of ratings")
    primary_image: str = Field(..., description="Primary product image URL")
    store_name: Optional[str] = Field(None, description="Store/Brand name")
    main_category: Optional[str] = Field(None, description="Product category")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "00043af7-7abc-4ba3-8cb3-c0a66370675c",
                "parent_asin": "B07GL3D1PK",
                "title": "Three Layer Women's Winter Headband for Yoga Running",
                "price": 0.0,
                "average_rating": 4.5,
                "rating_number": 15,
                "primary_image": "https://m.media-amazon.com/images/I/91+uvb4xcOL._AC_UL1500_.jpg",
                "store_name": "Maven Thread",
                "main_category": "AMAZON FASHION"
            }
        }