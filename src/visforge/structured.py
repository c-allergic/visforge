"""Prompt and validate the JSON replies used as visual evidence."""

import json
import math
import re
from dataclasses import dataclass

from .local_model import final_visible_text


class OutputError(ValueError):
    def __init__(self, subtype: str, message: str):
        super().__init__(message)
        self.subtype = subtype


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise OutputError("ambiguous_json", "JSON contains a duplicate object key")
    return result


def _finite_float(token):
    number = float(token)
    if not math.isfinite(number):
        raise OutputError("invalid_json", "JSON contains a non-finite number")
    return number


_DECODER = json.JSONDecoder(object_pairs_hook=_unique_object, parse_float=_finite_float, parse_constant=_finite_float)
# Containers can be introduced inline by prose. Scalars on their own line also
# count as replies, including inside a Markdown fence. Consume complete quoted
# strings so brackets inside a string never become evidence.
_TOKEN = re.compile(
    r'[{}\[\]"]|(?<![\w.])(?:true|false|null|NaN|-?Infinity|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    r'(?=[ \t]*(?:$|[\[{]))', re.MULTILINE,
)
# A bracketed phrase embedded in a sentence (e.g. "Answer [see below]")
# contains no nested JSON values. All other failed container starts are fatal.
_PROSE_NOTE = re.compile(r'(?<=\w )\[(?!true\b|false\b|null\b)[^\W\d_]+(?:[ \t]+[^\W\d_]+)*\]')


def _single_json(text):
    visible = final_visible_text(text).replace("\r\n", "\n")
    found = False
    pos = last_value_end = 0
    while match := _TOKEN.search(visible, pos):
        start = match.start()
        try:
            value, end = _DECODER.raw_decode(visible, start)
        except OutputError:
            raise
        except (ValueError, RecursionError):
            if note := _PROSE_NOTE.match(visible, start):
                pos = note.end()
                continue
            # Never scan inside a malformed outer container for a usable row.
            raise OutputError("invalid_json", "Response contains incomplete or invalid JSON") from None
        pos = end
        if not isinstance(value, (dict, list)):
            line_start = max(last_value_end, visible.rfind("\n", last_value_end, start) + 1)
            suffix = visible[end:].lstrip(" \t")
            if isinstance(value, str) and suffix.startswith(":"):
                raise OutputError("invalid_json", "Object member outside a JSON object")
            if visible[line_start:start].strip() or (suffix and suffix[0] != "\n" and not _TOKEN.match(suffix)):
                continue
        if found:
            raise OutputError("ambiguous_json", "Response contains multiple JSON values")
        found = True
        result = value
        last_value_end = end
    if not found:
        raise OutputError("invalid_json", "No complete JSON value in the visible response")
    return result


@dataclass(frozen=True)
class OutputContract:
    """Declare an example, required object field types, and any collection envelope.

    By default an object requires the top-level types in its example. An explicit
    fields mapping overrides those requirements for responses with optional fields.
    list_key permits a bare array and its named envelope, both returned as a list.
    """

    example: dict | list
    fields: dict[str, type] | None = None
    list_key: str | None = None

    @property
    def instruction(self) -> str:
        kind = "a JSON array" if isinstance(self.example, list) else "a JSON object"
        return f"Output only {kind} matching this example: " + json.dumps(self.example, ensure_ascii=False, allow_nan=False)

    def parse(self, text: str) -> dict | list:
        value = _single_json(text)
        if isinstance(value, dict):
            if self.list_key is None and not isinstance(self.example, dict):
                raise OutputError("wrong_schema", "Expected a JSON array")
            fields = self.fields
            if fields is None:
                fields = {k: type(v) for k, v in self.example.items()} if isinstance(self.example, dict) else {}
            for key, required_type in fields.items():
                if key not in value or type(value[key]) is not required_type:
                    raise OutputError("wrong_schema", f"Expected a {required_type.__name__} field named {key}")
            if self.list_key is None:
                return value
            if not isinstance(value.get(self.list_key), list):
                raise OutputError("wrong_schema", f"Expected an array field named {self.list_key}")
            if any(isinstance(v, (list, dict)) for k, v in value.items() if k != self.list_key):
                raise OutputError("ambiguous_json", "Collection envelope has competing structured fields")
            return value[self.list_key]
        if isinstance(value, list) and (self.list_key is not None or isinstance(self.example, list)):
            return value
        raise OutputError("wrong_schema", "Unexpected top-level JSON type")


READ_OUTPUT = OutputContract({"text": "...", "legible": True}, fields={"legible": bool})


def readable_text(data: dict) -> str | None:
    """Preserve a legible transcription or return None for explicit abstention."""
    if data.get("legible") is False:
        return None
    text = data.get("text")
    if data.get("legible") is not True or not isinstance(text, str) or not text.strip():
        raise OutputError("wrong_schema", "A legible reading requires nonempty text")
    return text
