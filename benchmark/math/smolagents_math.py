import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

WRITE_LOCK = threading.Lock()


def record_to_jsonl_file(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")


def load_bench_dataset() -> list[dict]:
    # https://huggingface.co/spaces/smolagents/smolagents-leaderboard
    ds = load_dataset("smolagents/benchmark-v1", "math")
    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    rows = []
    for i, d in enumerate(ds["test"]):
        question = d["question"]
        true_answer = d["true_answer"]

        question = f"{question}\n{instruction}"
        rows.append(
            {
                "id": str(i),
                "messages": [{"role": "user", "content": question}],
                "true_answer": true_answer,
            }
        )
    return rows


def call_azure_openai_api(llm_config: dict, messages: list[dict]):
    client = AzureOpenAI(
        azure_endpoint=llm_config["azure_endpoint"],
        api_key=llm_config["api_key"],
        api_version=llm_config["api_version"],
        max_retries=llm_config.get("max_retries", 3),
    )
    _config = {
        "model": llm_config["model"],
        "messages": messages,
        "max_tokens": 1024 * 8,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    response = None
    try:
        response = client.chat.completions.create(**_config)
    except Exception as e:
        print(e)
    return _config, response


def _job(job_id: str, llm_config: dict, sample: dict, output_file: str):

    messages = sample["messages"]
    tic = time.time()
    _config, response = call_azure_openai_api(llm_config, messages)
    latency = time.time() - tic
    _config["response"] = response.choices[0].message.content if response else None
    _config["finish_reason"] = response.choices[0].finish_reason if response else None
    _config["usage"] = {
        "prompt_tokens": response.usage.prompt_tokens if response else None,
        "completion_tokens": response.usage.completion_tokens if response else None,
    }
    _config["latency"] = latency
    _config["job_id"] = job_id
    if "true_answer" in sample:
        _config["true_answer"] = sample["true_answer"]

    record_to_jsonl_file(_config, output_file)


def run(output_file: str, llm_client_configs: list[dict], num_threads: int = 1):
    if not llm_client_configs:
        llm_client_configs = [
            {
                "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                "api_version": os.environ.get("AZURE_API_VERSION"),
                "model": os.environ.get("AZURE_DEPLOYMENT"),
            }
        ]
        print(json.dumps(llm_client_configs, indent=2))
    rows = load_bench_dataset()
    with ThreadPoolExecutor(max_workers=num_threads) as exe:
        futures = [
            exe.submit(
                _job,
                row.get("id", i),
                llm_client_configs[i % len(llm_client_configs)],
                row,
                output_file,
            )
            for i, row in enumerate(rows)
        ]
        for future in as_completed(futures):
            try:
                future.result(timeout=600)
            except Exception as e:
                print(f"Job failed: {e}")
    print("All tasks processed.")


def extract_math_solution(text: str | None) -> str | None:
    if not text:
        return None
    import re

    boxed_value = None
    idx = text.rfind("\\boxed")
    if idx != -1:
        answer_text = text[idx:]
        boxed_values = re.findall(r"\\boxed\{(.*)\}", answer_text)
        if boxed_values:
            boxed_value = boxed_values[0]
    return boxed_value


def is_equal(str1: str, str2: str):
    if str1 is None or str2 is None:
        return False
    try:
        s1 = str1.strip()
        s2 = str2.strip()
        return s1 == s2
    except Exception:
        return str1 == str2


def evel_accuracy(output_file: str):
    num_samples = 0
    acc = 0
    with open(output_file, "r", encoding="utf-8") as fp:
        for _, line in enumerate(fp):
            num_samples += 1
            row = json.loads(line)
            pred_answer = extract_math_solution(row["response"])
            e = is_equal(pred_answer, row["true_answer"])
            if e:
                acc += 1
            print(e, pred_answer, " vs ", row["true_answer"])
    print(f"Accuracy: {acc}/{num_samples} = {acc / num_samples * 100:.2f}%")


if __name__ == "__main__":
    run("./response_smolagents_math.jsonl", None, 1)
    evel_accuracy("./response_smolagents_math.jsonl")
    # gpt-4o-mini Accuracy: 34/50 = 68.00%
