from pathlib import Path

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.fleurs.utils import GOOGLE_ASR_PROMPT, google_asr_doc_to_text

EXPECTED_GOOGLE_ASR_PROMPT = """Transcribe the following speech segment in English into English text.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
<audio>"""


def test_google_asr_doc_to_text_uses_exact_prompt():
    assert GOOGLE_ASR_PROMPT == EXPECTED_GOOGLE_ASR_PROMPT
    assert google_asr_doc_to_text({}, {"pre_prompt": "ignored"}) == EXPECTED_GOOGLE_ASR_PROMPT


def test_google_asr_task_pins_official_fleurs_revision():
    task_path = Path(__file__).parents[2] / "lmms_eval" / "tasks" / "fleurs" / "fleurs_en_google_asr.yaml"
    task = task_path.read_text(encoding="utf-8")

    assert "dataset_path: google/fleurs" in task
    assert "revision: 70bb2e84b976b7e960aa89f1c648e09c59f894dd" in task
    assert "task: fleurs_en_google_asr" in task
    assert "doc_to_text: !function utils.google_asr_doc_to_text" in task


def test_google_asr_task_is_registered():
    assert "fleurs_en_google_asr" in TaskManager().all_tasks
