from pathlib import Path

import pytest

from lmms_eval.tasks.ai2d.utils import StrictDirectAnswerFilter
from lmms_eval.utils import load_yaml_config


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("A", "A"),
        ("(A)", "__invalid__"),
        ("a", "__invalid__"),
        ("B.", "__invalid__"),
        ("(c).", "__invalid__"),
        (" D ", "D"),
        ("A,", "__invalid__"),
        ("E", "__invalid__"),
        ("Answer: AD", "__invalid__"),
        ("A or B", "__invalid__"),
        ("not A", "__invalid__"),
        ("I reject option D", "__invalid__"),
        ("The final answer is B, not option A.", "__invalid__"),
    ],
)
def test_strict_direct_answer_filter(response, expected):
    assert StrictDirectAnswerFilter().apply([[response]], [{}]) == [expected]


def test_strict_direct_task_is_separate_and_versioned():
    task_path = Path(__file__).parents[2] / "lmms_eval" / "tasks" / "ai2d" / "ai2d_direct_strict_v1.yaml"

    config = load_yaml_config(task_path, mode="simple")

    assert config["task"] == "ai2d_direct_strict_v1"
    assert config["metadata"]["version"] == 1.0
    assert config["metadata"]["protocol"] == "local_custom_strict_direct_v1"
    assert config["filter_list"][0]["name"] == "strict-direct-extract-v1"
    assert config["metric_list"][0]["ignore_case"] is False
    assert config["metric_list"][0]["ignore_punctuation"] is False


def test_historical_ai2d_direct_task_version_is_unchanged():
    task_path = Path(__file__).parents[2] / "lmms_eval" / "tasks" / "ai2d" / "ai2d_direct.yaml"

    config = load_yaml_config(task_path, mode="simple")

    assert config["task"] == "ai2d_direct"
    assert config["metadata"][0]["version"] == 0.0
