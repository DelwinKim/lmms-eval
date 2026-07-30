import re

from lmms_eval.filters.extraction import ExtendedRegexFilter


def ai2d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if lmms_eval_specific_kwargs["prompt_format"] == "mcq":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "qa":
        options = "\n".join(choices)
        return f"{pre_prompt}{question}{options}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "mcq_xcomposer":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = " ".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\nContext: N/A\n{choices_str}{post_prompt}"
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs['prompt_format']}")


def ai2d_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def ai2d_doc_to_target(doc, model_specific_target_kwargs):
    if model_specific_target_kwargs == "mcq":
        len_choices = len(doc["options"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        return options[int(doc["answer"])]
    elif model_specific_target_kwargs == "qa":
        return doc["options"][int(doc["answer"])]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            # Process each response
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps


class DirectAnswerFilter:
    """Extract an AI2D option only from a direct or explicitly labeled answer."""

    _DIRECT_RE = re.compile(r"^\s*([A-D])\s*[.):]?\s*$", re.IGNORECASE)
    _LABELED_RE = re.compile(
        r"(?:final\s+answer|correct\s+answer|answer|option|choice)" r"\s*(?:is|:|=)?\s*(?:option\s*)?[\(\[]?([A-D])[\)\]]?\b",
        re.IGNORECASE,
    )

    def apply(self, resps, docs):
        del docs
        filtered_resps = []
        for response_group in resps:
            response = response_group[0] if isinstance(response_group, list) else response_group
            response = str(response).strip()
            normalized = re.sub(r"[*_`]", "", response)
            match = self._DIRECT_RE.fullmatch(normalized)
            if match:
                filtered_resps.append(match.group(1).upper())
                continue
            matches = list(self._LABELED_RE.finditer(normalized))
            filtered_resps.append(matches[-1].group(1).upper() if matches else response)
        return filtered_resps


class StrictDirectAnswerFilter:
    """Accept only the requested single uppercase A-D response."""

    _DIRECT_RE = re.compile(r"^[A-D]$")
    _INVALID = "__invalid__"

    def apply(self, resps, docs):
        del docs
        filtered_resps = []
        for response_group in resps:
            response = response_group[0] if isinstance(response_group, list) else response_group
            response = str(response).strip()
            filtered_resps.append(response if self._DIRECT_RE.fullmatch(response) else self._INVALID)
        return filtered_resps
