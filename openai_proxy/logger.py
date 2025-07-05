from loguru import logger
import asyncio
import time
from functools import wraps


logger.add(
    "time.log",
    rotation="1 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    enqueue=True
)


def measure(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.success(f"Finished {func.__name__} in {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Function {func.__name__} end in {duration:.4f}s with error: {str(e)}")
            raise e
    return wrapper
