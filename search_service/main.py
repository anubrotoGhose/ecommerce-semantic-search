from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routers import search_api, auth
from phoenix.otel import register
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


load_dotenv()


tracer_provider = register(
    project_name="fab-search-api",
    endpoint="http://10.169.101.75:6006/v1/traces",  # Your Phoenix instance
    auto_instrument=True,  # Auto-instruments OpenAI and FastAPI
    batch=False  # Send spans immediately (good for development)
)

app = FastAPI(title="Fashion Fab Search API")

# IMPORTANT: Manually instrument FastAPI AFTER app creation
FastAPIInstrumentor.instrument_app(app)

# DEVELOPMENT ONLY: Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_api.router)
app.include_router(auth.router)

@app.get("/")
def read_root():
    return {"message": "Fashion Search API is running", "status": "healthy"}
