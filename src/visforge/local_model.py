"""Transformers 推理：在一或两张可见 GPU 上按需加载一份本地 checkpoint。"""

import os
import re
from pathlib import Path
from threading import Lock

from PIL import Image


def final_visible_text(text):
    """Keep the final answer if the model emitted explicit thinking boundaries."""
    if not isinstance(text, str):
        return ""
    tail = re.split(r"</think\s*>", text, flags=re.IGNORECASE)[-1]
    return re.split(r"<think\s*>", tail, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _generation_info(inputs, tokens, config, limit):
    generated = len(tokens)
    last = int(tokens[-1]) if generated else None
    eos = config.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    finish = "eos" if last is not None and last in (eos or []) else "other"
    if finish != "eos" and generated >= limit:
        finish = "length"
    grid = inputs.get("image_grid_thw")
    return {
        "input_tokens": int(inputs["input_ids"].shape[-1]), "generated_tokens": generated,
        "max_new_tokens": limit, "last_token_id": last, "finish_reason": finish,
        "image_grid_thw": grid.tolist() if grid is not None else [],
    }


def _placement(cuda):
    gib = 1024 ** 3
    device_count = min(2, cuda.device_count())
    memory = [cuda.mem_get_info(index) for index in range(device_count)]
    if device_count == 1 and memory[0][0] >= 70 * gib:
        return {"device_map": {"": "cuda:0"}}

    # Leave room for image activations, the KV cache and the next offloaded layer.
    limits = {}
    for index, (free, total) in enumerate(memory):
        gpu_budget = free - 6 * gib
        if total <= 48 * gib:
            gpu_budget = min(gpu_budget, 38 * gib)
        if gpu_budget <= 0:
            raise RuntimeError("Qwen inference requires over 6 GiB free on each allocated GPU")
        limits[index] = gpu_budget

    import psutil  # Already required by Accelerate's device-map loader.

    available = psutil.virtual_memory().available
    slurm_mib = int(os.environ.get("SLURM_MEM_PER_NODE") or 0)
    if not slurm_mib:
        slurm_mib = int(os.environ.get("SLURM_MEM_PER_GPU", 0)) * device_count
    if not slurm_mib:
        slurm_mib = int(os.environ.get("SLURM_MEM_PER_CPU", 0)) * int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    if slurm_mib:
        available = min(available, slurm_mib * 1024 ** 2)
    limits["cpu"] = min(72 * gib, available * 3 // 4)
    return {"device_map": "balanced" if device_count == 2 else "auto", "max_memory": limits}


def _messages(request):
    content = []
    for source in request.get("images") or []:
        if isinstance(source, Image.Image):
            image = source.convert("RGB")
        else:
            with Image.open(source) as opened:
                image = opened.convert("RGB")
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": request.get("text", "")})
    messages = []
    if request.get("system"):
        messages.append({"role": "system", "content": request["system"]})
    return [*messages, {"role": "user", "content": content}]


def make_local_model(local_model_path, *, max_new_tokens=512, temperature=0.0):
    """返回 model(request)；第三方推理库和权重只在第一次调用时加载。"""
    path = Path(local_model_path).expanduser()
    if not path.is_dir():
        raise ValueError(f"local model directory does not exist: {path}")
    if max_new_tokens < 1 or temperature < 0:
        raise ValueError("max_new_tokens must be positive and temperature must be nonnegative")
    lock = Lock()
    model = processor = None
    generation = {"do_sample": temperature > 0}
    if temperature > 0:
        generation["temperature"] = temperature

    def respond(request):
        nonlocal model, processor
        import torch

        limit = request.get("max_new_tokens", max_new_tokens)
        if type(limit) is not int or limit < 1:
            raise ValueError("request max_new_tokens must be a positive integer")
        with lock, torch.inference_mode():
            if model is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("local Qwen inference requires an allocated CUDA GPU")
                from transformers import AutoModelForImageTextToText, AutoProcessor

                processor = AutoProcessor.from_pretrained(path, local_files_only=True)
                model = AutoModelForImageTextToText.from_pretrained(
                    path, local_files_only=True, dtype="auto", **_placement(torch.cuda),
                ).eval()

            # Qwen interprets the two size limits as pixel areas. Cap full images
            # while allowing crop-based operators to see detail from the original.
            inputs = processor.apply_chat_template(
                _messages(request), tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, enable_thinking=False,
                processor_kwargs={"images_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 2097152}}},
            ).to("cuda:0")  # Accelerate handles the following layer placements and any CPU offload.
            output = model.generate(**inputs, **generation, max_new_tokens=limit)
            tokens = output[0][inputs["input_ids"].shape[-1]:]
            meta = {"model": path.name, "backend": "transformers"}
            if request.get("generation_diagnostics"):
                # Keep tag boundaries until hidden reasoning has been removed.
                visible = final_visible_text(processor.decode(tokens, skip_special_tokens=False))
                text = re.sub(r"<\|[\w]+\|>", "", visible).strip()
                meta["generation"] = _generation_info(inputs, tokens, model.generation_config, limit)
            else:
                text = processor.decode(tokens, skip_special_tokens=True).strip()
            return text, meta

    return respond
