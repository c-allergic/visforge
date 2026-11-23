"""工具：裁剪、放大、读字、计算。返回值带 provenance（bbox、所用参数）。"""

import ast

from PIL import Image

from .structured import READ_OUTPUT, OutputError, readable_text


def _load(image):
    """路径或 PIL 图片统一成 PIL 图片。"""
    if isinstance(image, Image.Image):
        return image
    return Image.open(image)


def _clamp(bbox, size):
    """把 bbox 夹到图片范围内；夹完为空就报错。"""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    width, height = size
    x1, y1 = max(0, min(x1, width)), max(0, min(y1, height))
    x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox outside the image or invalid: {bbox}")
    return (x1, y1, x2, y2)


def _crop(image, bbox):
    img = _load(image)
    box = _clamp(bbox, img.size)
    return img.crop(box), box


_SAFE_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)


def _safe_eval(expression):
    """compute 只做纯数值运算：白名单节点校验后再求值。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"compute cannot parse the expression: {expression}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, _SAFE_BINOPS + (ast.USub, ast.UAdd)):
            continue  # 运算符节点本身
        if isinstance(node, ast.BinOp) and isinstance(node.op, _SAFE_BINOPS):
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            continue
        raise ValueError(f"compute supports only pure numeric expressions: {expression}")
    return eval(compile(tree, "<compute>", "eval"), {"__builtins__": {}}, {})


def run_tool(name, image, params):
    """执行一个工具。image 是路径或 PIL 图片，params 是参数 dict。

    crop    -> {"image": 裁剪图, "bbox", "provenance"}
    zoom    -> {"image": 放大图, "bbox", "scale", "provenance"}
    read_text -> {"text", "legible", "provenance"}（需要 params["model"]）
    compute -> {"value", "provenance"}
    """
    params = dict(params or {})
    if name == "crop":
        crop, box = _crop(image, params["bbox"])
        return {
            "image": crop,
            "bbox": list(box),
            "provenance": {"tool": "crop", "params": {"bbox": list(params["bbox"])}, "bbox": list(box)},
        }
    if name == "zoom":
        scale = float(params.get("scale", 2))
        crop, box = _crop(image, params["bbox"])
        big = crop.resize(
            (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
            Image.LANCZOS,
        )
        return {
            "image": big,
            "bbox": list(box),
            "scale": scale,
            "provenance": {
                "tool": "zoom",
                "params": {"bbox": list(params["bbox"]), "scale": scale},
                "bbox": list(box),
            },
        }
    if name == "read_text":
        model = params.get("model")
        if model is None:
            raise ValueError("read_text requires the model parameter")
        region = params.get("region")
        target = _load(image)
        provenance = {"tool": "read_text", "region": None, "model": None}
        if region:
            target, box = _crop(image, region)
            provenance["region"] = list(box)
        prompt = (
            "Read the text or number in the image."
            + (params.get("hint") or "")
            + "\nSet legible to false if the text cannot be read.\n" + READ_OUTPUT.instruction
        )
        text, meta = model({"text": prompt, "images": [target]})
        provenance["model"] = (meta or {}).get("model")
        try:
            result = readable_text(READ_OUTPUT.parse(text))
        except OutputError as error:
            provenance["response_error"] = error.subtype
            result = None
        return {"text": result or "", "legible": result is not None, "provenance": provenance}
    if name == "compute":
        expression = str(params.get("expression", ""))
        return {
            "value": _safe_eval(expression),
            "provenance": {"tool": "compute", "expression": expression},
        }
    raise ValueError(f"unknown tool: {name}")
