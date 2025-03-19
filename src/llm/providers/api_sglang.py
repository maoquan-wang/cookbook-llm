import openai


def call_sglang_openai_api(llm_config: dict, messages: list[dict]):

    client = openai.Client(
        base_url=llm_config["base_url"],
        api_key="EMPTY",
        max_retries=llm_config.get("max_retries", 3),
    )
    if "model" not in llm_config:
        llm_config["model"] = "default"

    _config = {
        "model": llm_config["model"],
        "messages": messages,
        "max_tokens": llm_config.get("max_tokens", 1024 * 8),
        "temperature": llm_config.get("temperature", 0.7),
        "top_p": llm_config.get("top_p", 0.95),
    }
    response = None
    try:
        response = client.chat.completions.create(**_config)
    except Exception as e:
        print(e)
    return _config, response


if __name__ == "__main__":
    llm_config = {
        "base_url": "http://0.0.0.0:33983/v1/",
    }
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    config, response = call_sglang_openai_api(llm_config, messages)
    print(response.choices[0].message.content)
