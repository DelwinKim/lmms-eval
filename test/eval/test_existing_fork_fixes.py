from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from lmms_eval.models.simple import gemma4
from lmms_eval.tasks.mmmu_pro import utils as mmmu_pro_utils
from lmms_eval.utils import load_yaml_config

TASKS_DIR = Path(__file__).parents[2] / "lmms_eval" / "tasks"


def test_textvqa_default_target_uses_annotator_answers():
    config = load_yaml_config(TASKS_DIR / "textvqa" / "_default_template_textvqa_yaml", mode="simple")

    assert config["doc_to_target"] == "answers"


def test_covost_en_zh_default_uses_dataset_default_config():
    config = load_yaml_config(TASKS_DIR / "covost2" / "_default_template_en_zh_yaml", mode="simple")

    assert "dataset_name" not in config


def test_mmmu_pro_vision_options_use_multiple_choice_parser(monkeypatch):
    monkeypatch.setattr(mmmu_pro_utils, "parse_multi_choice_response", lambda pred, choices, index2ans: "B")
    doc = {"id": "vision-1", "subject": "art", "answer": "B", "options": "['one', 'two']"}

    result = mmmu_pro_utils.mmmu_pro_process_results(doc, ["The answer is (B)."])

    assert result["mmmu_acc"]["parsed_pred"] == "B"


@pytest.mark.parametrize(("dtype", "expected"), [(None, torch.bfloat16), ("float16", torch.float16)])
def test_gemma_dtype_selection_preserves_bfloat16_default(dtype, expected):
    accelerator = MagicMock(num_processes=1)
    model = MagicMock()
    model.eval.return_value = model
    processor = MagicMock()
    kwargs = {"device": "cpu", "device_map": "cpu"}
    if dtype is not None:
        kwargs["dtype"] = dtype

    with (
        patch.object(gemma4, "Accelerator", return_value=accelerator),
        patch.object(gemma4.AutoModelForImageTextToText, "from_pretrained", return_value=model) as from_pretrained,
        patch.object(gemma4.AutoProcessor, "from_pretrained", return_value=processor),
    ):
        gemma4.Gemma4(pretrained="test/gemma4", **kwargs)

    assert from_pretrained.call_args.kwargs["torch_dtype"] is expected
