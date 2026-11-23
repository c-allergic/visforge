"""共用记录：Seed（原图出处）、QuestionView（盲答视图）、Candidate（待检查的题）。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Seed:
    """一张选好的原图。只留出处信息，缺什么字段迭代中再补。"""

    path: str
    group: str
    page: int | None = None


@dataclass
class QuestionView:
    """给答题者看的内容：只有图和问题，绝不带答案（盲答分离）。"""

    image: str
    question: str


@dataclass
class Candidate:
    """一道待检查的候选题：证据和计划轨迹都留着，事后可查构造记录。"""

    seed: Seed
    question: str
    answer: Any
    evidence: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    status: str = "pending"  # pending / accepted / revised / rejected
