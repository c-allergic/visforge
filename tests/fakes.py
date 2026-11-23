"""无网络测试用的假对象与合成图。"""

import json as _json
from pathlib import Path

from PIL import Image, ImageDraw

# 合成场景里各物体的像素位置（测试断言用）
BLUE_BOX = [140, 180, 220, 260]  # 托盘 A 上的蓝方块
BLUE_BOX_2 = [420, 140, 470, 190]  # 右上角另一个蓝方块（更小）
TRAY_A = [60, 260, 300, 440]
TRAY_B = [340, 260, 580, 440]
LABEL_A = "TEA"
LABEL_B = "DISH"


def make_scene(path):
    """合成图：两个托盘色块，蓝方块坐在托盘 A 上，各带文字标签。"""
    img = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(img)
    d.rectangle(TRAY_A, fill=(180, 180, 190), outline="black")
    d.rectangle(TRAY_B, fill=(180, 180, 190), outline="black")
    d.rectangle(BLUE_BOX, fill=(40, 80, 200), outline="black")
    d.rectangle(BLUE_BOX_2, fill=(40, 80, 200), outline="black")
    d.text((150, 400), LABEL_A, fill="black")
    d.text((430, 400), LABEL_B, fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


class FakeModel:
    """按关键语匹配返回预设回答的可编程假模型；记录每次调用。"""

    def __init__(self, rules=None, default=None):
        self._rules = list((rules or {}).items())
        self.default = default
        self.calls = []

    def add(self, matcher, response):
        """追加规则，后加的优先（插到最前）；matcher 是子串或可调用对象 (request)->bool。"""
        self._rules.insert(0, (matcher, response))

    def __call__(self, request):
        instructions = request.get("text", "")
        if instructions.startswith("You are writing questions") and "\nOperator definitions:\n" in instructions:
            # An operator prompt quoted as library data is not an invocation of that operator.
            header, definitions = instructions.split("\nOperator definitions:\n", 1)
            instructions = header + "\n" + definitions.partition("\nTemplate library:\n")[2]
        hay = instructions + "\n" + " ".join(str(i) for i in request.get("images") or [])
        self.calls.append(request)
        for matcher, response in self._rules:
            hit = matcher(request) if callable(matcher) else (matcher in hay)
            if hit:
                return self._emit(response, request)
        if self.default is not None:
            return self._emit(self.default, request)
        raise AssertionError(f"FakeModel has no scripted answer for: {hay[:200]!r}")

    def _emit(self, response, request):
        text = response(request) if callable(response) else response
        return text, {"model": "fake"}


class FakeJudge:
    """判官假对象：按回答文本查表判等效，未命中返回 default。"""

    def __init__(self, table=None, default=False):
        self.table = {str(k): bool(v) for k, v in (table or {}).items()}
        self.default = default
        self.calls = []

    def __call__(self, answer, reference):
        self.calls.append((str(answer), str(reference)))
        return self.table.get(str(answer), self.default)


def generation_model(invisible_on=None, verdict="accept"):
    """出题/验题假模型：能走通全部 6 条模板链。

    invisible_on 是图片路径子串，命中时找不到承托接触证据；None 表示总能找到。
    """
    m = FakeModel()

    def plan_scene(request):
        templates = _json.loads(request["text"].split("Template library:\n")[1])
        values = {"target": "the left blue square", "targets": "blue squares",
                  "a_desc": "blue squares", "b_desc": "trays", "entity_desc": "left blue square",
                  "anchor_desc": "right tray", "candidates_desc": "blue squares"}
        # Script the visual planner's preferences for this specific synthetic image.
        ranked = ["read-one-object", "support-read-label", "range-count", "extreme", "count-compare",
                  "locate", "same-page-match"]
        cues = {"support-read-label": "The left blue square touches a tray with a printed label.",
                "range-count": "Two separate blue squares are visible.",
                "extreme": "The square on the left is visibly larger.",
                "count-compare": "Both categories can be located in the scene.",
                "locate": "The left square and right tray are distinct.",
                "same-page-match": "The named square can be identified among the blue squares."}
        def priority(tpl):
            return ranked.index(tpl["name"]) if tpl["name"] in ranked else len(ranked)
        choices = [{"template": t["name"], "slots": {key: values[key] for key in t["slots"]},
                    "evidence": cues.get(t["name"], "There is a visible label under the blue square.")}
                   for t in sorted(templates, key=priority) if all(key in values for key in t["slots"])]
        return _json.dumps({"choices": choices})

    def list_objects(request):
        text = request["text"]
        if "left blue square" in text:
            objects = [{"name": "blue square", "bbox": BLUE_BOX}]
        elif "blue squares" in text:
            objects = [
                {"name": "blue square", "bbox": BLUE_BOX},
                {"name": "blue square", "bbox": BLUE_BOX_2},
            ]
        elif "right tray" in text:
            objects = [{"name": "tray", "bbox": TRAY_B}]
        elif "trays" in text:
            objects = [{"name": "tray", "bbox": TRAY_A}]
        else:
            objects = [
                {"name": "blue square", "bbox": BLUE_BOX},
                {"name": "blue square", "bbox": BLUE_BOX_2},
            ]
        return _json.dumps({"objects": objects})

    def find_partner(request):
        images = " ".join(str(i) for i in request.get("images") or [])
        if invisible_on is not None and invisible_on in images:
            return _json.dumps({"visible": False, "reason": "the bottom contact of the cup is not visible"})
        bbox = TRAY_B if "[420," in request["text"] else TRAY_A
        contact = [bbox[0] + 80, bbox[1] - 8, bbox[2] - 80, bbox[1] + 8]
        return _json.dumps({"partner": {"name": "tray", "bbox": bbox}, "contact": contact})

    def read_sizes(request):
        # Extreme：按 Select 返回顺序读 size
        return _json.dumps([
            {"name": "blue square", "bbox": BLUE_BOX, "value": "80 pixels"},
            {"name": "blue square", "bbox": BLUE_BOX_2, "value": "50 pixels"},
        ])

    def match_object(request):
        return _json.dumps({"match_index": 0, "cue": "same printed label", "distinct": True})

    def read_label(request):
        return _json.dumps({"text": LABEL_A, "legible": True})

    def verify(request):
        return _json.dumps({"verdict": verdict, "reason": "evidence is sufficient and the answer is correct"})

    m.add("You are writing questions", plan_scene)
    m.add("list all objects matching", list_objects)
    m.add("Target object:", find_partner)
    m.add("Output only a JSON array", read_sizes)
    m.add("match_index", match_object)
    m.add("What to read:", read_label)
    m.add("You are a question auditor", verify)
    return m


def solver_model():
    """答题假模型：计数题答对，读标签题故意答错（让题“更难”）。"""
    m = FakeModel()
    m.add("How many", "2")
    m.default = "no idea"
    return m
