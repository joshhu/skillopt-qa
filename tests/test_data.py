from skillopt.data import _format_context, _to_item, load_split


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


def test_load_split_missing_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "train")
