import anthropic


def call_anthropic_api(llm_config: dict, messages: list[dict]):
    client = anthropic.Client(
        api_key=llm_config["api_key"],
        max_retries=llm_config.get("max_retries", 3),
    )
    # models
    # claude-3-5-sonnet-20241022
    # claude-3-7-sonnet-20250219
    # claude-3-7-sonnet-20250219-think
    _config = {
        "model": llm_config["model"],
        "messages": messages,
        "max_tokens": llm_config.get("max_tokens", 1024 * 8),
        "temperature": llm_config.get("temperature", 0.7),
        "top_p": llm_config.get("top_p", 0.95),
    }
    response = None
    try:
        response = client.messages.create(**_config)
    except Exception as e:
        print(e)
    return _config, response
