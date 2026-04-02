from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
import logging
import time
from .utils.fetch_search_results_l2 import get_search_results
from .utils.auth import get_current_active_user, log_user_query
from .utils.schemas import SearchRequest
from opentelemetry import trace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/search", tags=["search"])
async def search_main(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user)
):
    try:
        current_span = trace.get_current_span()
        
        # Validate query
        if not request.user_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Add request attributes
        current_span.set_attribute("user.id", current_user["USER_ID"])
        current_span.set_attribute("user.username", current_user.get("USERNAME", "unknown"))
        current_span.set_attribute("search.query", request.user_query)
        current_span.set_attribute("search.page", request.page)
        current_span.set_attribute("search.page_size", request.page_size)
        
        # Log user query in background
        background_tasks.add_task(
            log_user_query, 
            current_user["USER_ID"], 
            request.user_query
        )

        # Execute search
        result = await get_search_results(
            request.user_query.strip(), 
            request.page, 
            request.page_size
        )
        
        # Add result metrics (primitive types only)
        current_span.set_attribute("search.results_count", result.get("totalcount", 0))
        current_span.set_attribute("search.total_pages", result.get("totalpages", 0))
        current_span.set_attribute("search.response_time_ms", result.get("responsetime", 0) * 1000)
        current_span.set_attribute("search.has_next", result.get("hasnext", False))
        current_span.set_attribute("search.has_previous", result.get("hasprevious", False))
        
        # Optionally: add JSON string of filters applied (if needed)
        if "filtersapplied" in result:
            current_span.set_attribute("search.filters_json", json.dumps(result["filtersapplied"]))
        
        # For full output logging, use OpenTelemetry events instead
        current_span.add_event(
            name="search.completed",
            attributes={
                "query.original": result.get("originalquery", ""),
                "query.improved": result.get("improvedquery", ""),
                "products.returned": len(result.get("products", []))
            }
        )
        
        return result
        
    except Exception as e:
        current_span.set_attribute("error", True)
        current_span.set_attribute("error.message", str(e))
        current_span.record_exception(e)
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.post("/search/public",tags=["search"])
async def search_main_public_debug(
    request: SearchRequest,
    background_tasks: BackgroundTasks
    ):
    try:
        if not request.user_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        return await get_search_results(
            request.user_query.strip(), 
            request.page, 
            request.page_size
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.get("/health", tags=["health"])
async def health_check():
    """Health check with Milvus status"""
    try:
        from .utils.fetch_search_results_l2 import get_collection
        collection = get_collection()
        return {
            "status": "healthy",
            "milvus_loaded": True,
            "total_entities": collection.num_entities
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
