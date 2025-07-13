import datetime
import sys
import uuid
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Optional, Type, Union, get_args, get_origin

BUILTIN_TYPES_MAP: dict[Type[Any], str] = {
    int: "int",
    bool: "bool",
    float: "float",
    str: "string",
    uuid.UUID: "uuid",
    datetime.datetime: "datetime",
    datetime.date: "date",
    datetime.time: "time",
    datetime.timedelta: "duration",
}


def builtin_name(t: Type[Any]) -> Optional[str]:
    return BUILTIN_TYPES_MAP.get(t)


def get_openapi_type(t: Type[Any]) -> str:
    match builtin_name(t):
        case "int":
            return "int"
        case "float":
            return "float"
        case "bool":
            return "bool"
        case None:
            return "object"
        case _:
            return "string"


def get_openapi_format(t: Type[Any]) -> str:
    match builtin_name(t):
        case "uuid":
            return "uuid"
        case "datetime":
            return "datetime"
        case "date":
            return "date"
        case "time":
            return "time"
        case "string":
            return "string"
        case _:
            return ""


def parse_annotation(annotation: Type[Any]) -> tuple[Type[Any], bool]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Если тип Annotated, извлекаем вложенный тип
    if origin is Annotated:
        annotation = args[0]
        origin = get_origin(annotation)
        args = get_args(annotation)

    # Проверка на Optional (Union с None)
    is_optional = False
    base_type = annotation

    if origin is Union:
        args = tuple(arg for arg in args if arg is not type(None))
        if len(args) == 1:
            base_type = args[0]
            is_optional = True
        else:
            base_type = Union[args]
            is_optional = type(None) in args
    elif sys.version_info >= (3, 10) and origin is UnionType:  # поддержка |
        args = tuple(arg for arg in args if arg is not type(None))
        if len(args) == 1:
            base_type = args[0]
            is_optional = True

    return base_type, is_optional


def read_prompts(*paths: Path) -> list[str]:
    return [p.read_text().strip() for p in paths]


def ensure_prompts(
    system_prompts: Optional[list[str]] = None,
    system_prompt_paths: Optional[list[Path]] = None,
) -> list[str]:
    if system_prompts is None and system_prompt_paths is None:
        err = "Either system_prompts or system_prompt_paths must be provided"
        raise ValueError(err)
    if system_prompts is not None and system_prompt_paths is not None:
        err = "Only one of system_prompts or system_prompt_paths must be provided"
        raise ValueError(err)
    if system_prompts is not None:
        return system_prompts
    if system_prompt_paths is not None:
        return read_prompts(*system_prompt_paths)

    err = "Unreachable code"
    raise RuntimeError(err)
