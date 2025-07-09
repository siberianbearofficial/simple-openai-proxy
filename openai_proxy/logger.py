from loguru import logger
import sys
import time
from functools import wraps

from openai_proxy.schemas import OpenAIRequest


logger.add(
    sys.stdout,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    enqueue=True
)


def measure(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}.")
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
