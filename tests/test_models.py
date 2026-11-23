"""模型入口：环境变量默认值、图片编码；不联网。"""

import io
import json
import sys
import traceback
import urllib.error
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from visforge import models
from visforge.models import _encode_image, default_settings, extract_json, make_model


def test_default_settings_from_env(monkeypatch):
    monkeypatch.setenv("VISFORGE_API_BASE", "http://x/v1")
    monkeypatch.setenv("VISFORGE_API_KEY", "k")
    monkeypatch.setenv("VISFORGE_MODEL", "m1")
    assert default_settings() == {"base_url": "http://x/v1", "api_key": "k", "model": "m1"}


def test_explicit_settings_override_env(monkeypatch):
    monkeypatch.setenv("VISFORGE_API_BASE", "http://x/v1")
    monkeypatch.setenv("VISFORGE_MODEL", "env-model")
    from visforge.models import _resolve_settings

    s = _resolve_settings({"model": "explicit"})
    assert s["model"] == "explicit"  # 显式设置优先于环境变量
    assert s["base_url"] == "http://x/v1"  # 其余取环境变量默认值


def test_make_model_requires_base_url(monkeypatch):
    monkeypatch.delenv("VISFORGE_API_BASE", raising=False)
    with pytest.raises(ValueError):
        make_model({"model": "m"})


def test_encode_image_pil_and_path(tmp_path):
    url = _encode_image(Image.new("RGB", (4, 4)))
    assert url.startswith("data:image/png;base64,")
    p = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(p)
    assert _encode_image(str(p)).startswith("data:image/jpeg;base64,")


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('回答如下：{"a": {"b": 2}} 完毕') == {"a": {"b": 2}}
    assert extract_json("没有 JSON") is None
    assert extract_json('{"坏掉": ') is None


def test_extract_json_parses_whole_arrays_and_quoted_braces():
    assert extract_json('{"label":"a}b["}') == {"label": "a}b["}
    text = 'Detected:\n```json\n[{"label":"a}b"}]\n```'
    assert extract_json(text, allow_array=True) == [{"label": "a}b"}]
    assert extract_json('Answer [see below]:\n```json\n{"x":"y"}\n```') == {"x": "y"}
    assert extract_json(text) is None
    assert extract_json('[{"label":"complete inner object"}', allow_array=True) is None
    assert extract_json('{"nested":{"ok":true}', allow_array=True) is None


@pytest.fixture
def verifier_env(monkeypatch):
    monkeypatch.setenv("VISFORGE_VERIFIER_API_BASE", "https://verifier.invalid/v1")
    monkeypatch.setenv("VISFORGE_VERIFIER_MODEL", "judge-v1")
    monkeypatch.setenv("VISFORGE_VERIFIER_API_KEY", "test-verifier-only")
    monkeypatch.setenv("VISFORGE_API_BASE", "https://legacy.invalid/v1")
    monkeypatch.setenv("VISFORGE_MODEL", "legacy")
    monkeypatch.setenv("VISFORGE_API_KEY", "test-legacy-only")


@pytest.mark.parametrize(
    "missing", ["VISFORGE_VERIFIER_API_BASE", "VISFORGE_VERIFIER_MODEL", "VISFORGE_VERIFIER_API_KEY"]
)
def test_cycle_rejects_incomplete_verifier_settings(monkeypatch, verifier_env, tmp_path, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match=missing):
        models.build_cycle_models(tmp_path)


@pytest.mark.parametrize("url", [
    "http://example.invalid/v1",
    "https:///v1",
    "https://name:test-url-secret@example.invalid/v1",
    "https://example.invalid/v1?token=test-url-secret",
    "https://example.invalid/v1#test-url-secret",
    "https://example.invalid:test-url-secret/v1",
    "https://example.invalid/\nAuthorization:test-url-secret",
])
def test_cycle_rejects_unsafe_endpoint_without_echoing_it(verifier_env, tmp_path, url):
    with pytest.raises(ValueError, match="HTTPS") as caught:
        models.build_cycle_models(tmp_path, verifier_base_url=url)
    assert "test-url-secret" not in "".join(traceback.format_exception(caught.value))


def test_cycle_verifier_uses_only_its_endpoint_key_and_model(monkeypatch, verifier_env, tmp_path):
    monkeypatch.setenv("MY_VERIFIER_KEY", "test-override-only")
    sent = []

    def urlopen(request, timeout):
        sent.append(request)
        assert timeout == 17
        return io.BytesIO(b'{"model":"judge-v2","choices":[{"message":{"content":"yes"}}]}')

    monkeypatch.setattr(models.urllib.request, "urlopen", urlopen)
    roles = models.build_cycle_models(
        tmp_path, verifier_api_key_env="MY_VERIFIER_KEY", verifier_timeout=17,
        verifier_model="judge-v2", verifier_base_url="https://override.invalid/api/v1/",
    )
    answer, metadata = roles["verifier"]({"text": "evaluate", "images": [Image.new("RGB", (1, 1))]})
    request, = sent
    assert request.full_url == "https://override.invalid/api/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-override-only"
    assert "Authorization" not in request.headers  # urllib must drop this header on a redirect
    assert b"test-override-only" not in request.data
    body = json.loads(request.data)
    assert body["model"] == "judge-v2"
    assert body["max_completion_tokens"] == 1024
    assert "max_tokens" not in body and "temperature" not in body
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert (answer, metadata) == ("yes", {"model": "judge-v2", "usage": {}})


def test_cycle_verifier_omits_service_error_details(monkeypatch, verifier_env, tmp_path):
    def urlopen(request, timeout):
        echoed = request.get_header("Authorization")
        raise urllib.error.HTTPError(request.full_url, 401, echoed, {}, io.BytesIO(echoed.encode()))

    monkeypatch.setattr(models.urllib.request, "urlopen", urlopen)
    verifier = models.build_cycle_models(tmp_path)["verifier"]
    with pytest.raises(RuntimeError, match="401") as caught:
        verifier({"text": "check"})
    assert "test-verifier-only" not in "".join(traceback.format_exception(caught.value))


def test_verifier_legacy_token_limit_does_not_forward_reasoning(monkeypatch, verifier_env, tmp_path):
    monkeypatch.setenv("VISFORGE_VERIFIER_TOKEN_FIELD", "max_tokens")
    monkeypatch.setenv("VISFORGE_VERIFIER_MAX_TOKENS", "384")
    sent = []

    def urlopen(request, timeout):
        sent.append(json.loads(request.data))
        response = {"choices": [{"finish_reason": "length", "message": {
            "reasoning_content": "private internal analysis", "content": None,
        }}]}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(models.urllib.request, "urlopen", urlopen)
    verifier = models.build_cycle_models(tmp_path)["verifier"]
    with pytest.raises(RuntimeError, match="token limit") as caught:
        verifier({"text": "check"})
    assert "private internal analysis" not in str(caught.value)
    assert len(sent) == 1
    assert sent[0]["max_tokens"] == 384
    assert "max_completion_tokens" not in sent[0] and "temperature" not in sent[0]


def test_verifier_reads_only_final_text_blocks(monkeypatch, verifier_env, tmp_path):
    def urlopen(request, timeout):
        body = json.loads(request.data)
        assert body["max_completion_tokens"] == 900
        response = {"choices": [{"finish_reason": "stop", "message": {"content": [
            {"type": "reasoning", "text": "wrong answer"}, {"type": "text", "text": "yes"},
        ]}}]}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(models.urllib.request, "urlopen", urlopen)
    verifier = models.build_cycle_models(tmp_path, verifier_token_field="max_completion_tokens",
                                         verifier_max_tokens=900)["verifier"]
    assert verifier({"text": "check"})[0] == "yes"


@pytest.mark.parametrize(
    "free_gib,total_gib,job_gib,expected_limits",
    [([79], [80], 96, None),
     ([44], [45], 96, {0: 38, "cpu": 72}),
     ([36], [45], 32, {0: 30, "cpu": 24}),
     ([79, 78], [80, 80], 96, {0: 73, 1: 72, "cpu": 72}),
     ([44, 44, 44], [45, 45, 45], 96, {0: 38, 1: 38, "cpu": 72})],
)
def test_cycle_shares_one_lazy_gpu_model_for_paths_pil_and_text(
    monkeypatch, verifier_env, tmp_path, free_gib, total_gib, job_gib, expected_limits,
):
    # Token IDs 9, 8 stand for the prompt, 7 for the answer generated by the checkpoint.
    class Inputs(dict):
        def to(self, device):
            assert device == "cuda:0"
            return self

    processor = Mock()
    processor.apply_chat_template.return_value = Inputs(
        input_ids=SimpleNamespace(shape=(1, 2)), image_grid_thw=SimpleNamespace(tolist=lambda: [[1, 6, 4]]),
    )
    processor.decode.side_effect = lambda ids, **kwargs: " ".join(str(token) for token in ids)
    # For a dispatched model its first parameter may be on a meta device.
    model = Mock(device="meta" if expected_limits else "cuda:0")
    model.generation_config.eos_token_id = [7, 17]
    model.eval.return_value = model
    model.generate.return_value = [[9, 8, 7]]
    transformers = SimpleNamespace(AutoProcessor=Mock(), AutoModelForImageTextToText=Mock())
    transformers.AutoProcessor.from_pretrained.return_value = processor
    transformers.AutoModelForImageTextToText.from_pretrained.return_value = model
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    gib = 1024 ** 3
    def mem_get_info(index):
        assert index < 2
        return free_gib[index] * gib, total_gib[index] * gib

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: len(free_gib), mem_get_info=mem_get_info),
        inference_mode=nullcontext,
    ))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=128 * gib),
    ))
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(job_gib * 1024))
    roles = models.build_cycle_models(tmp_path, local_max_new_tokens=32)
    assert roles["synthesis"] is roles["target"] is roles["observer"]
    assert roles["synthesis"] is not roles["verifier"]
    transformers.AutoModelForImageTextToText.from_pretrained.assert_not_called()
    path = tmp_path / "seed.png"
    crop = Image.new("RGB", (2, 3))
    crop.save(path)
    for role, images in [("synthesis", [path]), ("target", [crop]), ("observer", [])]:
        answer, meta = roles[role]({"system": "test system", "text": "test prompt", "images": images,
                                   "generation_diagnostics": True})
        assert answer == "7"
        assert meta["model"]
        assert meta["generation"] == {
            "input_tokens": 2, "generated_tokens": 1, "max_new_tokens": 32, "last_token_id": 7,
            "finish_reason": "eos", "image_grid_thw": [[1, 6, 4]],
        }
        args, kwargs = processor.apply_chat_template.call_args
        assert kwargs["enable_thinking"] is False
        system, user = args[0]
        assert system["role"] == "system"
        assert user["content"][-1] == {"type": "text", "text": "test prompt"}
        if images:
            assert user["content"][0]["image"].size == (2, 3)
        else:
            assert len(user["content"]) == 1
    assert transformers.AutoProcessor.from_pretrained.call_count == 1
    load = transformers.AutoModelForImageTextToText.from_pretrained
    assert load.call_count == 1
    assert load.call_args.kwargs["local_files_only"] is True
    if expected_limits:
        assert load.call_args.kwargs["device_map"] == ("balanced" if len(free_gib) >= 2 else "auto")
        assert load.call_args.kwargs["max_memory"] == {device: size * gib for device, size in expected_limits.items()}
    else:
        assert load.call_args.kwargs["device_map"] == {"": "cuda:0"}
    assert model.generate.call_args.kwargs["max_new_tokens"] == 32
    assert model.generate.call_args.kwargs["do_sample"] is False
    # A multi-rule visual plan may need more output than a short VQA answer.
    _, plan_meta = roles["synthesis"]({"text": "plan", "max_new_tokens": 1024,
                                        "generation_diagnostics": True})
    assert model.generate.call_args.kwargs["max_new_tokens"] == 1024
    assert plan_meta["generation"]["max_new_tokens"] == 1024
    roles["target"]({"text": "short answer"})
    assert model.generate.call_args.kwargs["max_new_tokens"] == 32


def test_generation_metrics_distinguish_token_limit_from_eos():
    from visforge.local_model import _generation_info

    inputs = {"input_ids": SimpleNamespace(shape=(1, 23))}
    config = SimpleNamespace(eos_token_id=9)
    assert _generation_info(inputs, [6, 7], config, 2)["finish_reason"] == "length"
    assert _generation_info(inputs, [6, 9], config, 2)["finish_reason"] == "eos"
    assert _generation_info(inputs, [7], config, 2)["finish_reason"] == "other"
