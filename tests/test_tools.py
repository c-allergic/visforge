"""工具：裁剪、放大、读字、计算，provenance 完整。"""

import json

import pytest
from PIL import Image

from fakes import make_scene
from visforge.tools import run_tool


def test_crop_returns_image_and_bbox_provenance(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    out = run_tool("crop", str(scene), {"bbox": [10, 20, 110, 70]})
    assert out["image"].size == (100, 50)
    assert out["bbox"] == [10, 20, 110, 70]
    assert out["provenance"]["tool"] == "crop"
    assert out["provenance"]["params"] == {"bbox": [10, 20, 110, 70]}


def test_zoom_enlarges_crop_and_records_scale(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    out = run_tool("zoom", str(scene), {"bbox": [60, 260, 300, 440], "scale": 2})
    assert out["image"].size == (480, 360)
    assert out["bbox"] == [60, 260, 300, 440]
    assert out["provenance"]["params"] == {"bbox": [60, 260, 300, 440], "scale": 2.0}


def test_read_text_uses_model_and_records_provenance(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    model_calls = []

    def model(request):
        model_calls.append(request)
        return json.dumps({"text": "TEA", "legible": True}), {"model": "fake"}

    out = run_tool("read_text", str(scene), {"region": [60, 260, 300, 440], "model": model})
    assert out["text"] == "TEA" and out["legible"] is True
    assert out["provenance"]["region"] == [60, 260, 300, 440]
    assert out["provenance"]["model"] == "fake"
    # 交给模型的是裁剪出来的区域图
    assert isinstance(model_calls[0]["images"][0], Image.Image)


@pytest.mark.parametrize("reply", [
    '{"legible":"false","text":"guessed"}',
    '{"legible":true,"text":null}',
    '{"legible":true,"text":"  "}',
    '<think>{"legible":true,"text":"draft"}</think>{"legible":false}',
])
def test_read_text_never_promotes_invalid_or_unreadable_text(reply):
    image = Image.new("RGB", (20, 20))
    result = run_tool("read_text", image, {"model": lambda request: (reply, {})})
    assert result["legible"] is False
    assert result["text"] == ""


def test_compute_safe_arithmetic():
    assert run_tool("compute", None, {"expression": "2*(3+4)"})["value"] == 14
    assert run_tool("compute", None, {"expression": "-3**2 + 1"})["value"] == -8
    with pytest.raises(ValueError):
        run_tool("compute", None, {"expression": "__import__('os').system('x')"})


def test_crop_out_of_range_raises(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    with pytest.raises(ValueError):
        run_tool("crop", str(scene), {"bbox": [600, 500, 700, 600]})
    with pytest.raises(ValueError):
        run_tool("nope", str(scene), {})
