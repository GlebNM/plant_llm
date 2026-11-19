from pydantic_ai.models.openai import OpenAIResponsesModelSettings

DEFAULT_MODEL = "gpt-6.1-sol"


def settings(effort: str = "medium"):
    return OpenAIResponsesModelSettings(openai_reasoning_effort=effort, timeout=120, request_timeout=1200, max_retries=0)
