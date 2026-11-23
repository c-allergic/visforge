"""模型入口：OpenAI 兼容的 /chat/completions 和本地视觉模型。"""

import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

from .local_model import final_visible_text, make_local_model


def default_settings():
    """环境变量给出的默认模型设置。"""
    return {
        "base_url": os.environ.get("VISFORGE_API_BASE", ""),
        "api_key": os.environ.get("VISFORGE_API_KEY", ""),
        "model": os.environ.get("VISFORGE_MODEL", ""),
    }


def _resolve_settings(settings):
    """合并显式设置与环境变量默认值；显式的非空值优先。"""
    merged = default_settings()
    for key, value in dict(settings or {}).items():
        if value is not None and value != "":
            merged[key] = value
    if not merged.get("base_url"):
        raise ValueError("missing base_url: set the VISFORGE_API_BASE environment variable or pass base_url in settings")
    if not merged.get("model"):
        raise ValueError("missing model: set the VISFORGE_MODEL environment variable or pass model in settings")
    merged.setdefault("api_key", "")
    merged.setdefault("temperature", 0.2)
    merged.setdefault("max_tokens", 1024)
    merged.setdefault("timeout", 120)
    return merged


def _encode_image(image):
    """把图片路径或 PIL 图片变成 base64 data URL。"""
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    path = Path(image)
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return "data:{};base64,{}".format(mime, base64.b64encode(path.read_bytes()).decode())


def respond(request, settings=None):
    """把图和文字发给指定模型，返回 (回答文本, 元信息含模型名)。

    request：{"system": 可选系统提示, "text": 正文, "images": [路径或 PIL 图片, ...]}。
    """
    return _chat_completion(request, _resolve_settings(settings))


def _chat_completion(request, s, *, private_errors=False):
    content = [{"type": "text", "text": request.get("text", "")}]
    for image in request.get("images") or []:
        content.append({"type": "image_url", "image_url": {"url": _encode_image(image)}})
    messages = []
    if request.get("system"):
        messages.append({"role": "system", "content": request["system"]})
    messages.append({"role": "user", "content": content})
    body = {
        "model": s["model"],
        "messages": messages,
        s.get("token_field", "max_tokens"): s["max_tokens"],
    }
    if s.get("temperature") is not None:
        body["temperature"] = s["temperature"]
    req = urllib.request.Request(
        s["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    req.add_unredirected_header("Authorization", "Bearer " + s.get("api_key", ""))
    try:
        with urllib.request.urlopen(req, timeout=s["timeout"]) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if private_errors:
            exc.close()
            raise RuntimeError(f"verifier API returned HTTP {exc.code}") from None
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"model API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        if private_errors:
            raise RuntimeError("verifier API unreachable") from None
        raise RuntimeError(f"model API unreachable: {exc.reason}") from exc
    text = _verifier_text(payload) if private_errors else payload["choices"][0]["message"]["content"]
    return text, {"model": payload.get("model", s["model"]), "usage": payload.get("usage", {})}


def _verifier_text(payload):
    try:
        choice = payload["choices"][0]
        text = choice["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise RuntimeError("verifier API returned an invalid chat completion") from None
    if isinstance(text, list):
        text = "".join(
            part["text"] for part in text if isinstance(part, dict)
            and part.get("type") == "text" and isinstance(part.get("text"), str)
        )
    if isinstance(text, str) and text.strip():
        return text
    if choice.get("finish_reason") == "length":
        raise RuntimeError("verifier hit its token limit before returning final text; increase VISFORGE_VERIFIER_MAX_TOKENS")
    raise RuntimeError("verifier returned no final text")


def make_model(settings=None):
    """把模型设置固定成一个 model(request) 函数，供各模块当 model 参数传入。"""
    resolved = _resolve_settings(settings)

    def model(request):
        return respond(request, resolved)

    return model


def _validate_verifier_endpoint(base_url):
    try:
        url = urllib.parse.urlsplit(base_url)
        valid = (
            url.scheme == "https" and bool(url.hostname) and url.username is None
            and url.port != 0  # Reading .port validates its syntax and range.
            and not re.search(r"[\x00-\x20\x7f\\?#]", base_url)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Use a valid HTTPS verifier base URL without userinfo, query, or fragment.")


def build_cycle_models(
    local_model_path, *, verifier_base_url=None, verifier_model=None,
    verifier_api_key_env="VISFORGE_VERIFIER_API_KEY", local_max_new_tokens=512,
    local_temperature=0.0, verifier_max_tokens=None, verifier_token_field=None, verifier_timeout=120,
):
    """首轮三个本地角色共用一份权重。独立 verifier 仅读取自己的配置。"""
    base_url = verifier_base_url or os.environ.get("VISFORGE_VERIFIER_API_BASE")
    model_id = verifier_model or os.environ.get("VISFORGE_VERIFIER_MODEL")
    if not base_url:
        raise ValueError("set verifier_base_url or VISFORGE_VERIFIER_API_BASE")
    _validate_verifier_endpoint(base_url)
    if not model_id:
        raise ValueError("set verifier_model or VISFORGE_VERIFIER_MODEL")
    key = os.environ.get(verifier_api_key_env, "").strip()
    if not key:
        if verifier_api_key_env == "VISFORGE_VERIFIER_API_KEY":
            raise ValueError("set VISFORGE_VERIFIER_API_KEY")
        raise ValueError("the configured verifier key environment variable is empty")
    if any(not 33 <= ord(c) <= 126 for c in key):
        raise ValueError("invalid verifier credential in the configured environment variable")
    token_field = verifier_token_field or os.environ.get("VISFORGE_VERIFIER_TOKEN_FIELD", "max_completion_tokens")
    if token_field not in ("max_tokens", "max_completion_tokens"):
        raise ValueError("VISFORGE_VERIFIER_TOKEN_FIELD must be max_tokens or max_completion_tokens")
    if verifier_max_tokens is None:
        verifier_max_tokens = os.environ.get("VISFORGE_VERIFIER_MAX_TOKENS", "1024")
    try:
        max_tokens = int(verifier_max_tokens)
    except (TypeError, ValueError):
        raise ValueError("VISFORGE_VERIFIER_MAX_TOKENS must be a positive integer") from None
    if max_tokens < 1:
        raise ValueError("VISFORGE_VERIFIER_MAX_TOKENS must be a positive integer")
    settings = {
        "base_url": base_url, "model": model_id, "api_key": key,
        "token_field": token_field, "max_tokens": max_tokens, "timeout": verifier_timeout,
    }

    def verifier(request):
        return _chat_completion(request, settings, private_errors=True)

    local = make_local_model(local_model_path, max_new_tokens=local_max_new_tokens, temperature=local_temperature)
    return {"synthesis": local, "target": local, "observer": local, "verifier": verifier}


def response_excerpt(text):
    """Bound and redact a diagnostic excerpt, hashing the full sanitized visible text."""
    visible = final_visible_text(text)
    visible = re.sub(r"data:[\w.+/-]+;base64,[A-Za-z0-9+/=]+", "[media removed]", visible, flags=re.I)
    visible = re.sub(r"\bBearer\s+[^\s,\x22']+|\bsk-[\w-]+", "[redacted]", visible, flags=re.I)
    visible = re.sub(
        r"""(\b[\w.-]*(?:api[_-]?key|access[_-]?token|secret[_-]?key)["']?\s*[:=]\s*)"""
        r"""(?:"[^"]*"|'[^']*'|[^\s,;}]+)""", r"\1[redacted]", visible, flags=re.I,
    )
    visible = " ".join("".join(ch if ch.isprintable() else " " for ch in visible).split())
    clipped = len(visible) > 1000
    return {
        "excerpt_truncated": clipped,
        "final_text_excerpt": visible[:495] + " […] " + visible[-495:] if clipped else visible,
        "final_text_sha256": hashlib.sha256(visible.encode()).hexdigest(),
    }


def extract_json(text, *, allow_array=False):
    """Decode the first top-level container, including a fenced or prose-wrapped reply."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text.startswith(("{", "[")):
        fence = re.search(r"^[ \t]*```(?:json)?[ \t]*\r?\n", text, flags=re.I | re.M)
        if fence:
            text = text[fence.end():].lstrip()
        else:
            start = re.search(r"[\[{]", text)
            if start is None:
                return None
            text = text[start.start():]
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) or (allow_array and isinstance(value, list)) else None
