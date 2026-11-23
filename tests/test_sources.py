"""选图清单：CSV 过滤、页序、可读性校验。"""

import pytest

from fakes import make_scene
from visforge.records import Seed
from visforge.sources import load_seeds


def test_load_seeds_csv_filters_by_purpose(tmp_path):
    a = make_scene(tmp_path / "a" / "one.png")
    b = make_scene(tmp_path / "b" / "two.png")
    manifest = tmp_path / "list.csv"
    manifest.write_text(
        "path,group,page,purpose\n"
        f"{a},r1,2,synthesize\n"
        f"{b},r1,1,holdout\n",
        encoding="utf-8",
    )
    seeds = load_seeds(manifest, "synthesize")
    assert len(seeds) == 1
    assert seeds[0] == Seed(str(a), "r1", 2)


def test_load_seeds_keeps_page_order_within_group(tmp_path):
    a = make_scene(tmp_path / "p3.png")
    b = make_scene(tmp_path / "p1.png")
    c = make_scene(tmp_path / "p2.png")
    manifest = tmp_path / "list.csv"
    manifest.write_text(
        "path,group,page\n"
        f"{a},r1,3\n{b},r1,1\n{c},r1,2\n",
        encoding="utf-8",
    )
    seeds = load_seeds(manifest)
    assert [s.page for s in seeds] == [1, 2, 3]


def test_load_seeds_txt_uses_parent_dir_as_group(tmp_path):
    a = make_scene(tmp_path / "docs" / "one.png")
    manifest = tmp_path / "list.txt"
    manifest.write_text(f"# 注释行\n{a}\n", encoding="utf-8")
    seeds = load_seeds(manifest)
    assert len(seeds) == 1 and seeds[0].group == "docs" and seeds[0].page is None


def test_load_seeds_rejects_unreadable_image(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    manifest = tmp_path / "list.txt"
    manifest.write_text(str(bad), encoding="utf-8")
    with pytest.raises(Exception):
        load_seeds(manifest)
