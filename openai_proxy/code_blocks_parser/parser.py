import json
from typing import Any


class CodeBlocksParser:
    def __init__(self, doc: str) -> None:
        self._doc = doc

    def find_json_blocks(self) -> list[dict[str, Any]]:
        try:
            return [json.loads(self._doc)]
        except json.JSONDecodeError:
            # ладно, в лоб не получилось, попробуем найти блоки json в тексте
            pass

        json_blocks: list[str] = []
        json_block_lines: list[str] = []
        is_json_block: bool = False
        for line in self._doc.split("\n"):
            if is_json_block and "```" in line:
                is_json_block = False
            elif is_json_block:
                json_block_lines.append(line.strip())
            elif "```json" in line and json_block_lines:
                is_json_block = True
                json_blocks.append("".join(json_block_lines))
                json_block_lines.clear()
            elif "```json" in line:
                is_json_block = True
        if json_block_lines:
            json_blocks.append("".join(json_block_lines))

        if not json_blocks:
            err = f"Json not found in doc: {self._doc}"
            raise ValueError(err)

        try:
            return [json.loads(b) for b in json_blocks]
        except json.JSONDecodeError as e:
            err = f"Invalid json in doc: {json_blocks}"
            raise ValueError(err) from e
