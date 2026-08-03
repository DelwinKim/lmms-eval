import inspect
import io
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.media_encoder import encode_image_to_data_url

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

# Gemma 4 vision: soft-token budgets (see HF Gemma4 docs)
ALLOWED_MAX_SOFT_TOKENS = frozenset({70, 140, 280, 560, 1120})
DEFAULT_MAX_SOFT_TOKENS = 280
DEFAULT_MAX_FRAMES = 32
_AUDIO_SUFFIXES = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})
_VIDEO_SUFFIXES = frozenset({".avi", ".flv", ".mkv", ".mov", ".mp4", ".webm", ".wmv"})


def _normalize_audio_array(array) -> np.ndarray:
    """Return a contiguous mono float32 waveform."""
    waveform = np.asarray(array, dtype=np.float32)
    if waveform.ndim == 2:
        if waveform.shape[0] <= 8:
            waveform = waveform.mean(axis=0)
        elif waveform.shape[1] <= 8:
            waveform = waveform.mean(axis=1)
        else:
            raise ValueError(f"Cannot infer channel axis for audio array with shape {waveform.shape}.")
    if waveform.ndim != 1:
        raise ValueError(f"Audio array must be one- or two-dimensional, got shape {waveform.shape}.")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _resample_audio(waveform: np.ndarray, sample_rate: int, target_sample_rate: int | None) -> tuple[np.ndarray, int]:
    if target_sample_rate is None or sample_rate == target_sample_rate:
        return waveform, sample_rate

    from lmms_eval.models.model_utils.audio_processing import downsample_audio

    resampled = downsample_audio(waveform, sample_rate, target_sample_rate)
    return _normalize_audio_array(resampled), target_sample_rate


def _audio_content_from_visual(visual, target_sample_rate: int | None = None) -> dict | None:
    """Convert supported lmms-eval audio values to HF structured content."""
    if isinstance(visual, dict):
        media_type = visual.get("type")
        if media_type not in (None, "audio"):
            return None
        if "array" in visual and "sampling_rate" in visual:
            waveform, sample_rate = _resample_audio(
                _normalize_audio_array(visual["array"]),
                int(visual["sampling_rate"]),
                target_sample_rate,
            )
            return {
                "type": "audio",
                "audio": waveform,
                "sampling_rate": sample_rate,
            }
        if media_type == "audio" and visual.get("path"):
            return {"type": "audio", "audio": visual["path"]}

    if isinstance(visual, (str, Path)) and Path(str(visual)).suffix.lower() in _AUDIO_SUFFIXES:
        return {"type": "audio", "audio": str(visual)}

    if hasattr(visual, "get_all_samples"):
        try:
            samples = visual.get_all_samples()
            data = samples.data.detach().cpu().numpy() if hasattr(samples.data, "detach") else samples.data
            waveform, sample_rate = _resample_audio(
                _normalize_audio_array(data),
                int(samples.sample_rate),
                target_sample_rate,
            )
            return {
                "type": "audio",
                "audio": waveform,
                "sampling_rate": sample_rate,
            }
        except RuntimeError as torchcodec_error:
            encoded = getattr(visual, "_hf_encoded", None)
            if not isinstance(encoded, dict):
                raise RuntimeError("TorchCodec failed to decode audio and no encoded source is available.") from torchcodec_error

            source = encoded.get("path")
            if not source and encoded.get("bytes") is not None:
                source = io.BytesIO(encoded["bytes"])
            if source is None:
                raise RuntimeError("TorchCodec audio source contains neither a path nor bytes.") from torchcodec_error

            try:
                import librosa

                waveform, sample_rate = librosa.load(
                    source,
                    sr=target_sample_rate or getattr(visual, "_desired_sample_rate", None),
                    mono=True,
                )
            except (ImportError, OSError, ValueError) as fallback_error:
                raise RuntimeError("Both TorchCodec and librosa failed to decode audio.") from fallback_error

            eval_logger.warning("TorchCodec failed to decode audio; recovered the encoded source with librosa.")
            return {
                "type": "audio",
                "audio": _normalize_audio_array(waveform),
                "sampling_rate": int(sample_rate),
            }

    return None


@register_model("gemma4")
class Gemma4(lmms):
    """
    Gemma 4 multimodal model (Hugging Face `AutoModelForImageTextToText`).

    Default checkpoint is the latest flagship instruction-tuned release; override with
    ``pretrained`` (e.g. ``google/gemma-4-26B-A4B-it``, ``google/gemma-4-E2B-it``).

    Requires a recent ``transformers`` build with Gemma 4 support.
    """

    def __init__(
        self,
        pretrained: str = "google/gemma-4-31B-it",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        max_length: int = 8192,
        max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS,
        max_num_frames: int = DEFAULT_MAX_FRAMES,
        interleave_visuals: Optional[bool] = False,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        reasoning_prompt: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if max_soft_tokens not in ALLOWED_MAX_SOFT_TOKENS:
            raise ValueError(f"max_soft_tokens must be one of {sorted(ALLOWED_MAX_SOFT_TOKENS)}, got {max_soft_tokens}")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Resolve the requested compute dtype. Defaults to bfloat16 (the model's
        # native precision); pass dtype="float16" to load in fp16 (e.g. to match
        # an fp16 ONNX build for a like-for-like comparison). Unknown/"auto"
        # values fall back to bfloat16 to preserve the original behavior.
        _DTYPE_MAP = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved_dtype = _DTYPE_MAP.get(str(dtype).lower(), torch.bfloat16)

        model_kwargs = {
            "torch_dtype": resolved_dtype,
            "device_map": self.device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        self._tokenizer = self.processor.tokenizer

        self._config = self._model.config
        self._max_length = max_length
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self.max_soft_tokens = max_soft_tokens
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.model.eval()

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Not implemented for Gemma4.")

    def flatten(self, input: List[List]) -> List:
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _encode_image_data_url(self, image: Image.Image) -> str:
        return encode_image_to_data_url(
            image,
            image_format="JPEG",
            mime_type="image/jpeg",
            convert_rgb=True,
            quality=85,
        )

    def _build_user_content(self, context: str, visuals: list) -> list[dict]:
        """Build Gemma structured content with model-card media ordering."""
        leading_content = []
        audio_content = []
        target_sample_rate = getattr(
            getattr(self.processor, "feature_extractor", None),
            "sampling_rate",
            None,
        )
        for visual in visuals:
            if isinstance(visual, dict) and visual.get("type") == "video":
                raise TypeError("Structured video dictionaries are not supported by the Gemma 4 wrapper.")
            if isinstance(visual, (str, Path)) and Path(str(visual)).suffix.lower() in _VIDEO_SUFFIXES:
                if not os.path.exists(visual):
                    raise FileNotFoundError(f"Video file not found: {visual}")
                leading_content.append({"type": "video", "video": str(visual)})
                continue
            if isinstance(visual, Image.Image):
                leading_content.append(
                    {
                        "type": "image",
                        "url": self._encode_image_data_url(visual),
                    }
                )
                continue

            audio = _audio_content_from_visual(visual, target_sample_rate)
            if audio is not None:
                audio_content.append(audio)
                continue
            raise TypeError(f"Unsupported Gemma 4 media input: {type(visual).__name__}")

        context = context.replace("<image>", "").replace("<audio>", "")
        return leading_content + [{"type": "text", "text": context}] + audio_content

    def _build_messages(self, context: str, visuals: list) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": self._build_user_content(context, visuals),
            }
        )
        return messages

    def _processor_kwargs(self) -> dict:
        return {
            "text_kwargs": {
                "padding": True,
                "pad_to_multiple_of": 8,
                "max_length": self.max_length,
                "truncation": True,
            },
            "images_kwargs": {"max_soft_tokens": self.max_soft_tokens},
        }

    def _decode_generated_ids(self, generated_ids, prefix_ids) -> str:
        raw_response = self.processor.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        parse_kwargs = {}
        if getattr(self.tokenizer, "response_template", None) is not None:
            if "prefix" not in inspect.signature(self.processor.parse_response).parameters:
                raise RuntimeError("This Gemma 4 response template requires a Transformers version " "whose parse_response method accepts prefix.")
            parse_kwargs["prefix"] = prefix_ids
        parsed_response = self.processor.parse_response(raw_response, **parse_kwargs)
        if not isinstance(parsed_response, dict):
            raise TypeError("Gemma 4 processor returned a non-dictionary parsed response.")
        content = parsed_response.get("content")
        if content is None:
            return ""
        if not isinstance(content, str):
            raise TypeError("Gemma 4 processor returned non-string response content.")
        return content

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            batched_messages = []
            for i, context in enumerate(contexts):
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                batched_messages.append(self._build_messages(context, visual_list[i]))

            # HF Processor.apply_chat_template: only Jinja template variables may appear in **kwargs.
            # Tokenizer / vision options must go in processor_kwargs (see processing_utils.apply_chat_template).
            inputs = self.processor.apply_chat_template(
                batched_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=self._processor_kwargs(),
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = [self._decode_generated_ids(output_ids, input_ids) for output_ids, input_ids in zip(generated_ids_trimmed, inputs.input_ids)]
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
