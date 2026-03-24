from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack

from daft.ai.openai.provider import OpenAIProvider
from daft.ai.protocols import TextEmbedderDescriptor

if TYPE_CHECKING:
    from daft.ai.openai.typing import OpenAIProviderOptions
    from daft.ai.typing import EmbedTextOptions


class LMStudioProvider(OpenAIProvider):
    """LM Studio provider that extends OpenAI provider with local server configuration.

    LM Studio runs a local server that's API-compatible with OpenAI, so we can reuse
    all the OpenAI logic and just configure the base URL to point to the local instance.
    """

    def __init__(
        self,
        name: str | None = None,
        **options: Unpack[OpenAIProviderOptions],
    ):
        if "api_key" not in options:
            options["api_key"] = "not-needed-for-lm-studio"
        if "base_url" not in options:
            options["base_url"] = "http://localhost:1234/v1"
        else:
            # Ensure base_url ends with /v1 for LM Studio compatibility.
            base_url = options["base_url"]
            if base_url is not None and not base_url.endswith("/v1"):
                options["base_url"] = base_url.rstrip("/") + "/v1"
        super().__init__(name or "lm_studio", **options)

    # ------------------------------------------------------------------
    # LM Studio Text Embedder
    # ------------------------------------------------------------------

    def get_text_embedder_descriptor(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        **options: Unpack[EmbedTextOptions],
    ) -> TextEmbedderDescriptor:
        """Validates options and returns a serializable TextEmbedderDescriptor."""
        from daft.ai.openai.protocols.text_embedder import _resolve_dimensions

        model_name = model or self.DEFAULT_TEXT_EMBEDDER
        embed_options: EmbedTextOptions = options

        if dimensions is not None:
            if dimensions <= 0:
                raise ValueError("Embedding dimensions must be a positive integer.")
            if "supports_overriding_dimensions" not in embed_options:
                embed_options = {**embed_options, "supports_overriding_dimensions": True}

        resolved_dims = _resolve_dimensions(model_name, dimensions, self._options, embed_options)

        return TextEmbedderDescriptor(
            provider="lm_studio",
            model=model_name,
            dimensions=resolved_dims,
            embed_options=embed_options,
        )
