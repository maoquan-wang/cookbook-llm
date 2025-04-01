import json

from datasets import load_dataset

"""
# an example:
row = {
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "3325109185250 - 230752473=?\nExclude words; show only the math."},
        {"role": "assistant", "content": "3324878432777"},
    ]
}
"""


def kodcode_v1_sft_r1(save_file: str):
    # https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1
    ds = load_dataset("KodCode/KodCode-V1-SFT-R1", split="train")

    n = 0
    with open(save_file, "w") as f:
        for row in ds:
            conversations = row["conversations"]
            assert len(conversations) == 2
            assert conversations[0]["from"] == "human"
            assert conversations[1]["from"] == "gpt"
            example = {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": conversations[0]["value"]},
                    {"role": "assistant", "content": conversations[1]["value"]},
                ]
            }
            f.write(json.dumps(example) + "\n")
            n += 1
    print(f"saved {n} example to {save_file}")


def sft_token_distribution(data_dir: str, tokenizer_path: str):

    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )

    def count_messages_tokens(example: dict):
        ids = tokenizer.apply_chat_template(example["messages"], tokenize=True)
        return len(ids)

    ds = load_dataset(data_dir, split="train")
    ds_count = ds.map(
        lambda row: {
            "token_count": count_messages_tokens(row),
        },
        num_proc=8,
        remove_columns=ds.column_names,
    )
    # the distribution of token_count
    df = pd.DataFrame(ds_count["token_count"])
    des = df.describe(percentiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    print(des)
    return des


if __name__ == "__main__":
    # set hf cache dir, the default is ~/.cache/huggingface
    # export HF_HOME=/tmp/hf
    kodcode_v1_sft_r1("/tmp/hf/dataset/tmp/sft_train.jsonl")

    sft_token_distribution(
        "/tmp/hf/dataset/tmp",
        "/tmp/hf/Qwen2.5-Coder-0.5B-Instruct",
    )
