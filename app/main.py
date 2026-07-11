import logging
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.config import get_settings
from app.routes import router


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    app = FastAPI(
        title="Flywheel Gateway",
        version="0.1.0",
        description="Self-improving hybrid inference gateway - local AMD GPU tier + cloud escalation.",
    )
    app.include_router(router)

    @app.get("/dashboard")
    async def dashboard():
        return FileResponse("app/static/dashboard.html")

    @app.get("/health")
    async def health():
        return {"status": "ok", "env": settings.env, "version": app.version}

    return app


app = create_app()