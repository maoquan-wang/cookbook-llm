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


if __name__ == "__main__":
    kodcode_v1_sft_r1("/tmp/sft_train.jsonl")
