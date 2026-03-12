import os

from google.genai import Client as GenAIClient
from jinja2 import Template
from llama_cloud_services import LlamaParse
from llama_cloud_services.parse.utils import ResultType


def get_llama_parse() -> LlamaParse:
    return LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),  # type: ignore
        parse_mode="parse_page_with_agent",
        model="gemini-2.5-pro",
        result_type=ResultType.MD,
    )


def get_llm() -> GenAIClient:
    # google.genai.Client will look in GEMINI_API_KEY or GOOGLE_API_KEY
    # if no key is explictly speficied here.
    return GenAIClient()


def get_prompt_template() -> Template:
    return Template(
        "# Main Text\n\n {{extracted_text}}\n\n# Tables\n\n {{extracted_tables}}\n\n## Instructions\n\nYour task is to assist the user understand their brockerage statement by providing explanations of its content. Based on the [Main Text](#main-text) and [Tables](#tables) sections reported above, generate a comprehensive explanation in everyday language that can be understood by everyone."
    )
