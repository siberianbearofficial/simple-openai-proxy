from fastapi import FastAPI

from openai_proxy import routers
from openai_proxy.exception_handler import endpoints_exception_handler


def create_app() -> FastAPI:
    app = FastAPI(
        title="Simple OpenAI Proxy API",
        description="Proxying requests to OpenAI",
        version="0.1.0",
        contact={
            "name": "OpenAI Proxy Support",
            "url": "https://github.com/siberianbearofficial/simple-openai-proxy",
            "email": "contact@aleksei-orlov.ru",
        },
    )

    app.include_router(routers.openai_router)

    app.exception_handler(Exception)(endpoints_exception_handler)

    return app
