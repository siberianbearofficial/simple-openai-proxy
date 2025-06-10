import traceback

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger


async def endpoints_exception_handler(_: Request, ex: Exception) -> JSONResponse:
    if isinstance(ex, ValueError):
        logger.error(ex)
        return JSONResponse(status_code=400, content={"detail": str(ex)})
    logger.exception(ex)
    return JSONResponse(status_code=500, content={"detail": traceback.format_exc()})
