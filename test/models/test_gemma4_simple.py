from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from lmms_eval.models.simple.gemma4 import (
    Gemma4,
    _audio_content_from_visual,
    _normalize_audio_array,
)


def test_normalize_audio_array_downmixes_channels():
    waveform = _normalize_audio_array([[1.0, -1.0], [0.0, 1.0]])

    np.testing.assert_allclose(waveform, [0.5, 0.0])
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous


def test_audio_content_from_legacy_audio_dict():
    content = _audio_content_from_visual(
        {
            "array": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            "sampling_rate": 16000,
        }
    )

    assert content["type"] == "audio"
    assert content["sampling_rate"] == 16000
    np.testing.assert_allclose(content["audio"], [0.5, 0.5])


def test_audio_content_from_torchcodec_decoder():
    class FakeAudioDecoder:
        def get_all_samples(self):
            return SimpleNamespace(
                data=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                sample_rate=24000,
            )

    content = _audio_content_from_visual(FakeAudioDecoder())

    assert content["sampling_rate"] == 24000
    np.testing.assert_allclose(content["audio"], [0.5, 0.5])


def test_audio_content_resamples_in_memory_audio(monkeypatch):
    def fake_resample(waveform, original_sr, target_sr):
        assert original_sr == 8000
        assert target_sr == 16000
        return np.repeat(waveform, 2)

    monkeypatch.setattr(
        "lmms_eval.models.model_utils.audio_processing.downsample_audio",
        fake_resample,
    )

    content = _audio_content_from_visual(
        {
            "array": np.array([0.25, -0.25], dtype=np.float32),
            "sampling_rate": 8000,
        },
        target_sample_rate=16000,
    )

    assert content["sampling_rate"] == 16000
    np.testing.assert_allclose(content["audio"], [0.25, 0.25, -0.25, -0.25])


def test_build_user_content_orders_image_text_audio(monkeypatch):
    model = object.__new__(Gemma4)
    model.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000)
    )
    monkeypatch.setattr(
        model, "_encode_image_data_url", lambda image: "data:image/jpeg;base64,test"
    )
    image = Image.new("RGB", (1, 1))
    audio = {"array": np.array([0.1, -0.1]), "sampling_rate": 16000}

    content = model._build_user_content("Transcribe this.<audio>", [image, audio])

    assert [part["type"] for part in content] == ["image", "text", "audio"]
    assert content[1]["text"] == "Transcribe this."


def test_build_user_content_rejects_unknown_media():
    model = object.__new__(Gemma4)
    model.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000)
    )

    with pytest.raises(TypeError, match="Unsupported Gemma 4 media input"):
        model._build_user_content("Prompt", [object()])


def test_build_user_content_rejects_structured_video_dict():
    model = object.__new__(Gemma4)
    model.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000)
    )

    with pytest.raises(TypeError, match="Structured video dictionaries"):
        model._build_user_content(
            "Describe this video.",
            [{"type": "video", "path": "/video.mp4"}],
        )


def test_audio_content_rejects_non_audio_path_dict():
    assert _audio_content_from_visual({"type": "video", "path": "/video.mp4"}) is None
    assert _audio_content_from_visual({"path": "/ambiguous.wav"}) is None


def test_build_user_content_treats_webm_path_as_video(monkeypatch):
    model = object.__new__(Gemma4)
    model.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000)
    )
    monkeypatch.setattr("lmms_eval.models.simple.gemma4.os.path.exists", lambda _: True)

    content = model._build_user_content("Describe this.", ["/video.webm"])

    assert content == [
        {"type": "video", "video": "/video.webm"},
        {"type": "text", "text": "Describe this."},
    ]


def test_audio_content_accepts_explicit_webm_audio_path():
    assert _audio_content_from_visual({"type": "audio", "path": "/audio.webm"}) == {
        "type": "audio",
        "audio": "/audio.webm",
    }


def test_build_messages_omits_empty_system_prompt():
    model = object.__new__(Gemma4)
    model.system_prompt = ""
    model.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000)
    )

    messages = model._build_messages("Transcribe this.", [])

    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Transcribe this."}],
        }
    ]


def test_processor_kwargs_do_not_apply_text_length_to_audio():
    model = object.__new__(Gemma4)
    model._max_length = 8192
    model.max_soft_tokens = 280

    processor_kwargs = model._processor_kwargs()

    assert "max_length" not in processor_kwargs
    assert processor_kwargs["text_kwargs"]["max_length"] == 8192
    assert processor_kwargs["images_kwargs"]["max_soft_tokens"] == 280


def test_decode_generated_ids_returns_content_without_thinking():
    class FakeProcessor:
        tokenizer = SimpleNamespace(response_template=None)

        def decode(self, generated_ids, **kwargs):
            assert generated_ids == [1, 2, 3]
            assert kwargs == {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            }
            return "<|channel>thought\nreasoning<channel|>Transcript.<turn|>"

        def parse_response(self, response):
            assert response.endswith("<turn|>")
            return {
                "role": "assistant",
                "thinking": "reasoning",
                "content": "Transcript.",
            }

    model = object.__new__(Gemma4)
    model.processor = FakeProcessor()
    model._tokenizer = model.processor.tokenizer

    assert model._decode_generated_ids([1, 2, 3], [10, 11]) == "Transcript."


def test_decode_generated_ids_passes_prefix_for_new_response_template():
    class FakeProcessor:
        tokenizer = SimpleNamespace(response_template={"version": "1"})

        def decode(self, generated_ids, **kwargs):
            return "Transcript.<turn|>"

        def parse_response(self, response, *, prefix=None):
            assert response == "Transcript.<turn|>"
            assert prefix == [10, 11]
            return {"role": "assistant", "content": "Transcript."}

    model = object.__new__(Gemma4)
    model.processor = FakeProcessor()
    model._tokenizer = model.processor.tokenizer

    assert model._decode_generated_ids([1, 2, 3], [10, 11]) == "Transcript."
