from fastapi import FastAPI
from app.api.routes import health
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(health.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
