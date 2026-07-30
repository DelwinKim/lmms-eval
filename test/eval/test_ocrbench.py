from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from lmms_eval.api.task import TaskConfig
from lmms_eval.evaluator_utils import TaskOutput
from lmms_eval.tasks.ocrbench import utils
from lmms_eval.utils import load_yaml_config


def _results(size, correct, category="Regular Text Recognition"):
    return [{"question_type": category, "score": int(index < correct)} for index in range(size)]


@pytest.mark.parametrize(
    ("size", "correct", "expected_accuracy"),
    [
        (1, 1, 1.0),
        (500, 422, 0.844),
        (1000, 714, 0.714),
    ],
)
def test_ocrbench_accuracy_is_limit_aware(size, correct, expected_accuracy):
    assert utils.ocrbench_aggregate_accuracy(_results(size, correct), None) == pytest.approx(expected_accuracy)


def test_ocrbench_full_raw_score_preserves_canonical_semantics():
    report = mock_open()
    with patch.object(utils, "generate_submission_file", return_value="ocrbench_results.txt"), patch("builtins.open", report):
        raw_score = utils.ocrbench_aggregate_raw_score(_results(1000, 714), None)

    assert raw_score == 714
    report_text = "".join(call.args[0] for call in report().write.call_args_list)
    assert "Evaluated rows: 1000/1000" in report_text
    assert "Raw score: 714 (canonical full-benchmark raw score)" in report_text


def test_ocrbench_partial_raw_score_is_labeled_and_warned():
    report = mock_open()
    with (
        patch.object(utils, "generate_submission_file", return_value="ocrbench_results.txt"),
        patch.object(utils.logger, "warning") as warning,
        patch("builtins.open", report),
    ):
        raw_score = utils.ocrbench_aggregate_raw_score(_results(500, 422), None)

    assert raw_score == 422
    warning.assert_called_once()
    report_text = "".join(call.args[0] for call in report().write.call_args_list)
    assert "Evaluated rows: 500/1000" in report_text
    assert "partial raw score; not comparable" in report_text


def test_ocrbench_aggregators_are_stateless_across_repeated_calls():
    assert utils.ocrbench_aggregate_accuracy(_results(1, 1), None) == 1.0
    assert utils.ocrbench_aggregate_accuracy(_results(1, 0), None) == 0.0

    with patch.object(utils, "_write_ocrbench_report"):
        assert utils.ocrbench_aggregate_raw_score(_results(1, 1), None) == 1
        assert utils.ocrbench_aggregate_raw_score(_results(1, 0), None) == 0


def test_ocrbench_empty_input_fails_clearly():
    with pytest.raises(ValueError, match="at least one evaluated row"):
        utils.ocrbench_aggregate_accuracy([], None)
    with pytest.raises(ValueError, match="at least one evaluated row"):
        utils.ocrbench_aggregate_raw_score([], None)


def test_ocrbench_category_report_uses_only_current_call():
    report = mock_open()
    current_results = _results(1, 1, "Doc-oriented VQA")
    with patch.object(utils, "generate_submission_file", return_value="ocrbench_results.txt"), patch("builtins.open", report):
        utils.ocrbench_aggregate_raw_score(current_results, None)

    report_text = "".join(call.args[0] for call in report().write.call_args_list)
    assert "Regular Text Recognition: 0 (0/50 rows evaluated)" in report_text
    assert "Doc-oriented VQA: 1 (1/200 rows evaluated)" in report_text


def test_ocrbench_process_results_emits_both_metrics():
    doc = {"answer": "Olive", "dataset": "TextVQA", "question_type": "Doc-oriented VQA"}

    metrics = utils.ocrbench_process_results(doc, ["The Olive project"])

    assert metrics["ocrbench_accuracy"]["score"] == 1
    assert metrics["ocrbench_raw_score"] == metrics["ocrbench_accuracy"]
    assert metrics["ocrbench_raw_score"] is not metrics["ocrbench_accuracy"]


def test_ocrbench_raw_score_suppresses_mean_style_stderr():
    task = type(
        "Task",
        (),
        {"aggregation": lambda self: {"ocrbench_raw_score": sum}},
    )()
    output = TaskOutput(
        task=task,
        task_name="ocrbench",
        task_config={"skip_stderr_metrics": ["ocrbench_raw_score"]},
    )
    output.sample_metrics[("ocrbench_raw_score", "none")] = [1, 0, 1]

    output.calculate_aggregate_metric(bootstrap_iters=100)
    output.calculate_clt_aggregate_metric()

    assert output.agg_metrics["ocrbench_raw_score,none"] == 2
    assert output.agg_metrics["ocrbench_raw_score_stderr,none"] == "N/A"
    assert output.agg_metrics["ocrbench_raw_score_stderr_clt,none"] == "N/A"
    assert output.agg_metrics["ocrbench_raw_score_stderr_clustered,none"] == "N/A"


def test_ocrbench_task_config_accepts_raw_score_stderr_suppression():
    task_path = Path(__file__).parents[2] / "lmms_eval" / "tasks" / "ocrbench" / "ocrbench.yaml"

    config = TaskConfig(**load_yaml_config(task_path, mode="simple"))

    assert config.skip_stderr_metrics == ["ocrbench_raw_score"]
