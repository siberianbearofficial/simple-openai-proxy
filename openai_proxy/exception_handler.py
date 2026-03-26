import traceback

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger

from openai_proxy.services.polza_cost_control import CostLimitExceededError


async def endpoints_exception_handler(_: Request, ex: Exception) -> JSONResponse:
    if isinstance(ex, CostLimitExceededError):
        logger.warning(ex)
        return JSONResponse(status_code=429, content={"detail": str(ex)})
    if isinstance(ex, ValueError):
        logger.error(ex)
        return JSONResponse(status_code=400, content={"detail": str(ex)})
    logger.exception(ex)
    return JSONResponse(status_code=500, content={"detail": traceback.format_exc()})
