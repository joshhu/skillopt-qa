from skillopt.data import _flatten_drcd, _format_context, _to_item, load_split


def test_format_context_dict_form():
    ctx = {"title": ["A", "B"], "sentences": [["s1.", "s2."], ["t1."]]}
    out = _format_context(ctx)
    assert "## A" in out and "## B" in out
    assert "s1. s2." in out and "t1." in out


def test_format_context_list_form():
    ctx = [["A", ["s1."]], ["B", ["t1.", "t2."]]]
    out = _format_context(ctx)
    assert "## A" in out and "t1. t2." in out


def test_to_item_shape():
    row = {
        "id": "abc",
        "question": " who? ",
        "answer": "him",
        "context": {"title": ["T"], "sentences": [["x."]]},
    }
    item = _to_item(row, 0)
    assert item["id"] == "abc"
    assert item["question"] == "who?"
    assert item["answers"] == ["him"]
    assert "## T" in item["context"]


def test_flatten_drcd_shape_and_dedup():
    rows = [{
        "title": "臺北市",
        "id": "0",
        "paragraphs": [{
            "context": "臺北市是臺灣的首都。",
            "id": "p0",
            "qas": [
                {"id": "q0", "question": "臺灣的首都是哪裡?",
                 "answers": [{"text": "臺北市", "answer_start": 0},
                             {"text": "臺北市", "answer_start": 0}]},
                {"id": "q1", "question": "沒有答案?", "answers": []},
            ],
        }],
    }]
    items = _flatten_drcd(rows)
    assert len(items) == 1                    # 空 answers 的 qa 被略過
    item = items[0]
    assert item["id"] == "q0"
    assert item["question"] == "臺灣的首都是哪裡?"
    assert item["answers"] == ["臺北市"]      # 去重
    assert item["context"].startswith("## 臺北市\n")
    assert "臺北市是臺灣的首都。" in item["context"]


def test_load_split_missing_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "train")
