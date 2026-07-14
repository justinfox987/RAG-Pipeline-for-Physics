import os
from .cborg_provider import CBorgProvider


def get_provider():
    provider_name = os.environ.get("LLM_PROVIDER", "cborg").lower()

    if provider_name == "cborg":
        api_key = os.environ["CBORG_API_KEY"]
        return CBorgProvider(
            api_key=api_key,
            base_url="https://api.cborg.lbl.gov",
        )

    raise SystemExit(f"Unknown LLM_PROVIDER='{provider_name}'")
