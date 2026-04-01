"""
config.py — Environment loading and backend selection.

Reads from environment variables (or a .env file) and provides:
  - get_esm_client()      → ESM3 inference client (Forge API or local model)
  - get_anthropic_client() → Anthropic client for NL parsing
  - BACKEND_MODE          → "forge" | "local"
  - GPU info              → device string + GPU name if available
"""

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# ── Raw env values ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
FORGE_API_TOKEN: str | None = os.getenv("FORGE_API_TOKEN")
USE_LOCAL_ESM3: bool = os.getenv("USE_LOCAL_ESM3", "false").lower() == "true"
FORGE_MODEL: str = os.getenv("FORGE_MODEL", "esm3-medium-2024-08")
FORGE_URL: str = "https://forge.evolutionaryscale.ai"

# ── Backend mode ───────────────────────────────────────────────────────────────
BACKEND_MODE: Literal["forge", "local"] = (
    "local" if (USE_LOCAL_ESM3 or not FORGE_API_TOKEN) else "forge"
)


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_gpu_name() -> str | None:
    """Return the GPU device name, or None if no GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def get_esm_client():
    """
    Return the appropriate ESM3 inference client based on env config.

    - If FORGE_API_TOKEN is set and USE_LOCAL_ESM3 is false → Forge API client
    - Otherwise → local ESM3-open model (loaded onto GPU if available)

    Raises RuntimeError if neither backend can be initialised.
    """
    if BACKEND_MODE == "forge":
        try:
            from esm.sdk.forge import ESM3ForgeInferenceClient
            return ESM3ForgeInferenceClient(
                model=FORGE_MODEL,
                url=FORGE_URL,
                token=FORGE_API_TOKEN,
            )
        except ImportError as e:
            raise RuntimeError(
                "Could not import ESM SDK. Run: pip install esm"
            ) from e

    # Local ESM3-open
    try:
        from esm.models.esm3 import ESM3
        device = get_device()
        gpu_name = get_gpu_name()

        if device == "cpu":
            print(
                "WARNING: No GPU detected. ESM3-open on CPU is very slow (~20+ min/candidate). "
                "Consider using Colab Pro with an A100 GPU or setting FORGE_API_TOKEN."
            )
        else:
            if gpu_name and "A100" not in gpu_name:
                print(
                    f"INFO: Running on {gpu_name}. For best performance an A100 is recommended "
                    f"(~10-30s/candidate). Current GPU may be slower."
                )
            elif gpu_name:
                print(f"INFO: A100 detected ({gpu_name}). Optimal performance.")

        model = ESM3.from_pretrained("esm3-open")
        return model.to(device)

    except ImportError as e:
        raise RuntimeError(
            "Could not import ESM SDK. Run: pip install esm"
        ) from e


def get_anthropic_client():
    """Return an Anthropic client. Raises if ANTHROPIC_API_KEY is not set."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file or environment."
        )
    try:
        import anthropic
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError as e:
        raise RuntimeError(
            "Could not import anthropic SDK. Run: pip install anthropic"
        ) from e


def validate_config() -> list[str]:
    """
    Validate the current configuration and return a list of warning strings.
    An empty list means configuration is fully valid.
    """
    warnings = []

    if not ANTHROPIC_API_KEY:
        warnings.append("ANTHROPIC_API_KEY is not set — NL parsing will not work.")

    if BACKEND_MODE == "local":
        device = get_device()
        if not FORGE_API_TOKEN:
            warnings.append(
                "FORGE_API_TOKEN is not set — falling back to local ESM3-open (1.4B). "
                "Quality will be lower than Forge API models."
            )
        if device == "cpu":
            warnings.append(
                "No GPU detected — local ESM3-open will be extremely slow on CPU. "
                "Run on Colab Pro (A100) for acceptable performance."
            )

    return warnings
