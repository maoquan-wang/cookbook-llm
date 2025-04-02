from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/tmp/hf/models/Qwen2.5-Coder-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def call_llm(llm_config: dict, messages: list[dict]):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=llm_config.get("max_tokens", 64000),
        do_sample=True,
        top_p=llm_config.get("top_p", 0.95),
        temperature=llm_config.get("temperature", 0.7),
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


if __name__ == "__main__":

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "write a quick sort algorithm."},
    ]
    print(call_llm({}, messages))

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    print(call_llm({}, messages))
