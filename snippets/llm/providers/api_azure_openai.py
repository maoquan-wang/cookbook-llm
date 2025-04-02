from openai import AzureOpenAI


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
