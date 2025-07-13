import json
from typing import Type, TypeVar

from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class CodeBlocksParser:
    def __init__(self, doc: str) -> None:
        self._doc = doc

    def find_json_blocks(self) -> list[str]:
        try:
            json.loads(self._doc)
        except json.JSONDecodeError:
            # ладно, в лоб не получилось, попробуем найти блоки json в тексте
            pass
        else:
            return [self._doc]

        json_blocks = self._extract_code_blocks("json")
        if not json_blocks:
            return []

        try:
            for b in json_blocks:
                json.loads(b)
        except json.JSONDecodeError as e:
            err = f"Invalid json in doc: {json_blocks}"
            raise ValueError(err) from e

        return json_blocks

    def find_json_block(
        self,
        error_if_not_found: bool = False,
        error_if_multiple_found: bool = False,
    ) -> str:
        json_blocks = self.find_json_blocks()

        if len(json_blocks) < 1:
            err = f"Json blocks not found: {self._doc}"
            if error_if_not_found:
                raise RuntimeError(err)

            logger.warning(err)
            return ""

        if len(json_blocks) > 1:
            err = f"Multiple json blocks found: {self._doc}, using first one"
            if error_if_multiple_found:
                raise RuntimeError(err)

            logger.warning(err)

        return json_blocks[0]

    def find_and_validate_json_block(
        self,
        model: Type[T],
        error_if_not_found: bool = False,
        error_if_multiple_found: bool = False,
    ) -> T:
        json_block = self.find_json_block(error_if_not_found, error_if_multiple_found)
        return model.model_validate_json(json_block)

    def _extract_code_blocks(self, language: str = "") -> list[str]:
        code_blocks: list[str] = []
        code_block_lines: list[str] = []
        is_code_block: bool = False
        for line in self._doc.split("\n"):
            if is_code_block and "```" in line:
                is_code_block = False
            elif is_code_block:
                code_block_lines.append(line.strip())
            elif f"```{language}" in line and code_block_lines:
                is_code_block = True
                code_blocks.append("".join(code_block_lines))
                code_block_lines.clear()
            elif f"```{language}" in line:
                is_code_block = True
        if code_block_lines:
            code_blocks.append("".join(code_block_lines))

        return code_blocks
