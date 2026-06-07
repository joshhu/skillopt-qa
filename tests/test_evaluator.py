from skillopt.agent import Trajectory
from skillopt.evaluator import _tokenize, evaluate, exact_match, f1, normalize


def test_normalize_strips_articles_punct_case():
    assert normalize("The  Eiffel Tower!") == "eiffel tower"
    assert normalize("A dog.") == "dog"


def test_normalize_strips_cjk_punctuation():
    assert normalize("臺北市。") == "臺北市"
    assert normalize("「臺北」,首都") == "臺北首都"


def test_tokenize_mixed_segmentation():
    # CJK 逐字、Latin 連續字母/數字為一詞
    assert _tokenize("臺北市") == ["臺", "北", "市"]
    assert _tokenize("New York City") == ["new", "york", "city"]
    assert _tokenize("iPhone 15") == ["iphone", "15"]


def test_f1_chinese_partial_overlap():
    # 中文部分重疊:不再退化成全有/全無
    score = f1("臺北市", ["臺北"])
    assert 0.0 < score < 1.0
    assert f1("臺北", ["臺北"]) == 1.0
    assert f1("高雄", ["臺北"]) == 0.0


def test_exact_match_chinese_with_punctuation():
    assert exact_match("臺北市。", ["臺北市"]) == 1.0


def test_exact_match_after_normalization():
    assert exact_match("the answer", ["Answer"]) == 1.0
    assert exact_match("nope", ["Answer"]) == 0.0


def test_f1_partial_overlap():
    score = f1("New York City", ["New York"])
    assert 0.0 < score < 1.0
    assert f1("New York", ["New York"]) == 1.0
    assert f1("paris", ["london"]) == 0.0


def test_evaluate_sets_correct_and_aggregates():
    trajs = [
        Trajectory("1", "q1", ["yes"], "yes", "yes"),
        Trajectory("2", "q2", ["Paris"], "London", "London"),
    ]
    res = evaluate(trajs)
    assert res.n == 2
    assert res.em == 0.5
    assert trajs[0].correct is True
    assert trajs[1].correct is False
    assert res.metric("em") == 0.5
