from types import SimpleNamespace

import pytest

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.task import ConfigurableTask
from lmms_eval.evaluator import _clone_padding_request, _record_model_response


def _request(request_type="loglikelihood", repeats=1):
    return Instance(
        request_type=request_type,
        arguments=("context", "continuation"),
        idx=0,
        metadata={"task": "test", "doc_id": 0, "repeats": repeats},
    )


@pytest.mark.parametrize("response", [(1.25, False), [1.25, False]])
def test_loglikelihood_tuple_and_cached_list_are_preserved(response):
    req = _request()

    _record_model_response("loglikelihood", req, response)

    assert req.resps == [response]
    assert req.resps[0] is response
    assert req.token_counts == [None]


def test_repeated_loglikelihood_responses_and_padding_remain_structured():
    req = _request(repeats=2)
    padding_req = _clone_padding_request(req)

    _record_model_response("loglikelihood", req, (0.5, True))
    _record_model_response("loglikelihood", req, [0.75, False])
    _record_model_response("loglikelihood", padding_req, (1.0, False))

    assert req.resps == [(0.5, True), [0.75, False]]
    assert req.token_counts == [None, None]
    assert padding_req.resps == [(1.0, False)]
    assert padding_req.token_counts == [None]
    assert padding_req.metadata["__padding_only__"] is True


def test_generation_result_metadata_is_unchanged():
    req = _request(request_type="generate_until")
    token_counts = TokenCounts(input_tokens=11, output_tokens=3, reasoning_tokens=2)

    _record_model_response("generate_until", req, GenerationResult("answer", token_counts))

    assert req.resps == ["answer"]
    assert req.token_counts == [token_counts]


def test_preserved_loglikelihood_lists_support_multiple_choice_argmin():
    task = SimpleNamespace(
        OUTPUT_TYPE="multiple_choice",
        config=SimpleNamespace(process_results=None),
        _metric_fn_list={"acc": None, "acc_norm": None, "exact_match": None},
        doc_to_choice=lambda doc: doc["choices"],
        doc_to_target=lambda doc: doc["gold"],
        multiple_input=False,
        multiple_target=False,
    )
    results = [[1.2, False], [0.2, True], [0.8, False]]

    processed = ConfigurableTask.process_results(task, {"choices": ["A", "B", "C"], "gold": 1}, results)

    assert processed == {"acc": 1.0, "acc_norm": 1.0, "exact_match": 1}
