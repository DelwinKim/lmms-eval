from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

OCRBENCH_TOTAL_ROWS = 1000
OCRBENCH_CATEGORY_TOTALS = {
    "Regular Text Recognition": 50,
    "Irregular Text Recognition": 50,
    "Artistic Text Recognition": 50,
    "Handwriting Recognition": 50,
    "Digit String Recognition": 50,
    "Non-Semantic Text Recognition": 50,
    "Scene Text-centric VQA": 200,
    "Doc-oriented VQA": 200,
    "Key Information Extraction": 200,
    "Handwritten Mathematical Expression Recognition": 100,
}
OCRBENCH_RECOGNITION_CATEGORIES = (
    "Regular Text Recognition",
    "Irregular Text Recognition",
    "Artistic Text Recognition",
    "Handwriting Recognition",
    "Digit String Recognition",
    "Non-Semantic Text Recognition",
)


def ocrbench_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [doc["image"].convert("RGB")]


def ocrbench_doc_to_text(doc, lmms_eval_specific_kwargs):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def ocrbench_process_results(doc, results):
    pred = results[0].strip()
    gt_ans = doc["answer"]
    dataset_name = doc["dataset"]

    score = 0
    if dataset_name == "HME100k":
        if type(gt_ans) == list:
            for j in range(len(gt_ans)):
                answer = gt_ans[j].strip().replace("\n", " ").replace(" ", "")
                predict = pred.strip().replace("\n", " ").replace(" ", "")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.strip().replace("\n", " ").replace(" ", "")
            predict = pred.strip().replace("\n", " ").replace(" ", "")
            if answer in predict:
                score = 1
    else:
        if type(gt_ans) == list:
            for j in range(len(gt_ans)):
                answer = gt_ans[j].lower().strip().replace("\n", " ")
                predict = pred.lower().strip().replace("\n", " ")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.lower().strip().replace("\n", " ")
            predict = pred.lower().strip().replace("\n", " ")
            if answer in predict:
                score = 1
    result = {"question_type": doc["question_type"], "score": score, "prediction": pred, "ground_truth": gt_ans}
    return {"ocrbench_accuracy": result, "ocrbench_raw_score": result.copy()}


def _summarize_ocrbench_results(results):
    category_scores = dict.fromkeys(OCRBENCH_CATEGORY_TOTALS, 0)
    category_rows = dict.fromkeys(OCRBENCH_CATEGORY_TOTALS, 0)
    for result in results:
        question_type = result["question_type"]
        category_rows[question_type] += 1
        category_scores[question_type] += result["score"]
    return category_scores, category_rows


def _write_ocrbench_report(results, args):
    category_scores, category_rows = _summarize_ocrbench_results(results)
    evaluated_rows = len(results)
    recognition_score = sum(category_scores[category] for category in OCRBENCH_RECOGNITION_CATEGORIES)
    final_score = sum(category_scores.values())
    accuracy = final_score / evaluated_rows if evaluated_rows else 0.0
    is_full_benchmark = evaluated_rows == OCRBENCH_TOTAL_ROWS
    if not is_full_benchmark:
        logger.warning(f"OCRBench raw score {final_score} is from a partial evaluation ({evaluated_rows}/{OCRBENCH_TOTAL_ROWS} rows); use ocrbench_accuracy for limit-aware comparisons.")

    file_name = generate_submission_file("ocrbench_results.txt", args, subpath="results")
    with open(file_name, "w", encoding="utf-8") as f:
        print("######################### OCRBench #############################", file=f)
        print(f"Evaluated rows: {evaluated_rows}/{OCRBENCH_TOTAL_ROWS}", file=f)
        print(f"Accuracy over evaluated rows: {accuracy:.6f}", file=f)
        raw_label = "canonical full-benchmark raw score" if is_full_benchmark else "partial raw score; not comparable to the canonical 1000-row score"
        print(f"Raw score: {final_score} ({raw_label})", file=f)
        print(f"Text Recognition: {recognition_score} ({sum(category_rows[category] for category in OCRBENCH_RECOGNITION_CATEGORIES)}/300 rows evaluated)", file=f)
        print("---------------- Details of Recognition Score ------------------", file=f)
        for category, total_rows in OCRBENCH_CATEGORY_TOTALS.items():
            print(f"{category}: {category_scores[category]} ({category_rows[category]}/{total_rows} rows evaluated)", file=f)
        print("--------------------- Final Score ------------------------------", file=f)
        print(f"Final Raw Score ({evaluated_rows}/{OCRBENCH_TOTAL_ROWS} rows evaluated): {final_score}", file=f)
    logger.info(f"OCR Bench results saved to {file_name}")


def ocrbench_aggregate_accuracy(results, args):
    del args
    if not results:
        raise ValueError("OCRBench accuracy requires at least one evaluated row.")
    return sum(result["score"] for result in results) / len(results)


def ocrbench_aggregate_raw_score(results, args):
    if not results:
        raise ValueError("OCRBench raw score requires at least one evaluated row.")
    _write_ocrbench_report(results, args)
    return sum(result["score"] for result in results)
