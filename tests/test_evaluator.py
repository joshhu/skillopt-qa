from skillopt.agent import Trajectory
from skillopt.evaluator import evaluate, exact_match, f1, normalize


def test_normalize_strips_articles_punct_case():
    assert normalize("The  Eiffel Tower!") == "eiffel tower"
    assert normalize("A dog.") == "dog"


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
