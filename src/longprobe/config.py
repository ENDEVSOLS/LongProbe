"""
Configuration dataclasses for LongProbe.

All LongProbe configuration is loaded from a single YAML file (or a plain
``dict``) and materialised into typed dataclass instances.  String values
support ``${ENV_VAR}`` interpolation so secrets such as API keys can be
injected via environment variables.

Typical usage::

    config = ProbeConfig.from_yaml("longprobe.yaml")
    print(config.retriever.type)          # e.g. "langchain"
    print(config.embedder.model)          # e.g. "text-embedding-3-small"
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


# ---------------------------------------------------------------------------
# Environment-variable expansion
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: str) -> str:
    """Replace ``${ENV_VAR}`` placeholders in *value* with the corresponding
    environment variable.

    If an environment variable is referenced but not set, the placeholder is
    left intact so that downstream code can raise a more contextual error.
    """
    def _replacer(match: re.Match) -> str:
        env_name = match.group(1)
        env_val = os.environ.get(env_name)
        if env_val is None:
            return match.group(0)  # leave placeholder as-is
        return env_val

    return _ENV_PATTERN.sub(_replacer, value)


# ---------------------------------------------------------------------------
# RetrieverConfig
# ---------------------------------------------------------------------------

@dataclass
class HttpResponseMapping:
    """Configuration for mapping JSON response fields to LongProbe format.

    Attributes:
        results_path: Dot-notation path to the results array in the JSON
            response (e.g. ``"data.chunks"``).
        id_field: Field name for chunk identifier within each result item.
        text_field: Field name for chunk text content within each result item.
        score_field: Field name for relevance score within each result item.
    """

    results_path: str = "results"
    id_field: str = "id"
    text_field: str = "text"
    score_field: str = "score"


@dataclass
class HttpRetrieverConfig:
    """Configuration for the generic HTTP retriever adapter.

    Attributes:
        url: URL of the RAG API endpoint.
        method: HTTP method (``"POST"`` or ``"GET"``).
        body_template: JSON string template with ``{question}`` and
            ``{top_k}`` placeholders.
        headers: HTTP headers to send with each request.  Values support
            ``${ENV_VAR}`` expansion.
        response_mapping: Mapping configuration for parsing the response.
        timeout: Request timeout in seconds.
    """

    url: str = ""
    method: str = "POST"
    body_template: str = '{"query": "{question}", "top_k": {top_k}}'
    headers: Dict[str, str] = field(default_factory=dict)
    response_mapping: HttpResponseMapping = field(default_factory=HttpResponseMapping)
    timeout: int = 30


@dataclass
class RetrieverConfig:
    """Configuration for the vector-store / retriever backend.

    Attributes:
        type: Retriever implementation to use.  One of ``"langchain"``,
            ``"llamaindex"``, ``"pinecone"``, ``"qdrant"``, ``"chroma"``,
            ``"http"``.
        collection: Collection name (LangChain / Chroma).
        persist_directory: Local persistence directory (Chroma / LangChain).
        index_name: Index name (Pinecone).
        namespace: Namespace scope (Pinecone / Qdrant).
        host: Hostname for remote retrievers (Qdrant).
        port: Port for remote retrievers (Qdrant).
        api_key: API key for cloud-hosted retrievers.  Supports
            ``${ENV_VAR}`` expansion.
        http: HTTP-specific configuration (used when ``type`` is ``"http"``).
        extra: Provider-specific key-value pairs not covered by the typed
            fields above.
    """

    type: str = "langchain"
    collection: str = ""
    persist_directory: str = ""
    index_name: str = ""
    namespace: str = ""
    host: str = ""
    port: int = 6333
    api_key: str = ""
    http: HttpRetrieverConfig = field(default_factory=HttpRetrieverConfig)
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EmbedderConfig
# ---------------------------------------------------------------------------

@dataclass
class EmbedderConfig:
    """Configuration for the embedding model.

    Attributes:
        provider: Embedding provider.  One of ``"openai"``, ``"huggingface"``,
            ``"local"``.
        model: Model identifier (e.g. ``"text-embedding-3-small"``).
        dimensions: Target embedding dimensionality.  ``0`` means use the
            model's default.
        batch_size: Number of texts to embed in a single API call.
    """

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 0
    batch_size: int = 32


# ---------------------------------------------------------------------------
# ScoringConfig
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """Configuration for the scoring / evaluation step.

    Attributes:
        recall_threshold: Minimum recall score for a question to be
            considered a pass.  Must be in ``[0.0, 1.0]``.
        fail_on_regression: When ``True``, the probe exits with a non-zero
            code if *any* question falls below the recall threshold.
    """

    recall_threshold: float = 0.8
    fail_on_regression: bool = True


# ---------------------------------------------------------------------------
# BaselineConfig
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    """Configuration for baseline tracking.

    Attributes:
        db_path: Path to the SQLite database used to store historical
            baseline results.
        auto_compare: When ``True``, automatically compare current results
            against the most recent stored baseline.
    """

    db_path: str = ".longprobe/baselines.db"
    auto_compare: bool = True


# ---------------------------------------------------------------------------
# ProbeConfig
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """Configuration for the question generation module.

    Attributes:
        provider: LLM provider to use.  One of ``"openai"``, ``"anthropic"``,
            ``"gemini"``, ``"ollama"``, or any provider supported by litellm.
        model: Model identifier (e.g. ``"gpt-4o-mini"``, ``"claude-3-haiku-20240307"``).
        api_key: API key for the LLM provider.  Supports ``${ENV_VAR}`` expansion.
        base_url: Custom base URL for the provider (useful for Ollama or
            self-hosted endpoints).
        num_questions: Default number of questions to generate.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens in the LLM response.
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""
    num_questions: int = 50
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class ProbeConfig:
    """Top-level configuration container.

    Aggregates all sub-configurations required to run a LongProbe evaluation.

    Attributes:
        retriever: Vector-store / retriever settings.
        embedder: Embedding model settings.
        scoring: Evaluation / scoring settings.
        baseline: Baseline comparison settings.
        generator: Question generation settings.
    """

    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def defaults(cls) -> "ProbeConfig":
        """Return a :class:`ProbeConfig` instance with every field set to its
        declared default value.

        This is useful for quick experimentation or as a base that is later
        selectively overridden.
        """
        return cls()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbeConfig":
        """Build a :class:`ProbeConfig` from a plain dictionary.

        Unknown keys in *data* are silently ignored so that configuration
        files can contain extra provider-specific fields without breaking
        parsing.

        Args:
            data: Dictionary of configuration values.  Nested sections
                (``retriever``, ``embedder``, ``scoring``, ``baseline``) are
                mapped to their respective sub-config dataclasses.

        Returns:
            A fully-populated :class:`ProbeConfig` instance.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"ProbeConfig.from_dict expects a dict, got {type(data).__name__}."
            )

        # Expand ``${ENV_VAR}`` in all string values, recursively.
        data = _expand_env_recursive(data)

        retriever_raw = data.get("retriever", {})
        embedder_raw = data.get("embedder", {})
        scoring_raw = data.get("scoring", {})
        baseline_raw = data.get("baseline", {})
        generator_raw = data.get("generator", {})

        retriever = _build_retriever_config(retriever_raw)
        embedder = _build_embedder_config(embedder_raw)
        scoring = _build_scoring_config(scoring_raw)
        baseline = _build_baseline_config(baseline_raw)
        generator = _build_generator_config(generator_raw)

        return cls(
            retriever=retriever,
            embedder=embedder,
            scoring=scoring,
            baseline=baseline,
            generator=generator,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ProbeConfig":
        """Load configuration from a YAML file.

        Args:
            path: Filesystem path to the YAML configuration file.

        Returns:
            A fully-populated :class:`ProbeConfig` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            yaml.YAMLError: If the file cannot be parsed as YAML.
            ValueError: If the top-level structure is not a mapping.
        """
        import os as _os

        if not _os.path.isfile(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a YAML mapping at the top level of '{path}', "
                f"got {type(data).__name__}."
            )

        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_env_recursive(obj: Any) -> Any:
    """Walk a nested structure and expand ``${ENV_VAR}`` in every string."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_recursive(item) for item in obj]
    return obj


def _build_http_response_mapping(raw: Dict[str, Any]) -> HttpResponseMapping:
    """Construct an :class:`HttpResponseMapping` from a raw dict."""
    if not isinstance(raw, dict):
        raw = {}
    return HttpResponseMapping(
        results_path=_str(raw, "results_path", "results"),
        id_field=_str(raw, "id_field", "id"),
        text_field=_str(raw, "text_field", "text"),
        score_field=_str(raw, "score_field", "score"),
    )


def _build_http_config(raw: Dict[str, Any]) -> HttpRetrieverConfig:
    """Construct an :class:`HttpRetrieverConfig` from a raw dict."""
    if not isinstance(raw, dict):
        raw = {}
    return HttpRetrieverConfig(
        url=_str(raw, "url", ""),
        method=_str(raw, "method", "POST"),
        body_template=_str(
            raw, "body_template",
            '{"query": "{question}", "top_k": {top_k}}',
        ),
        headers={str(k): str(v) for k, v in (raw.get("headers") or {}).items()},
        response_mapping=_build_http_response_mapping(raw.get("response_mapping") or {}),
        timeout=_int(raw, "timeout", 30),
    )


def _build_retriever_config(raw: Dict[str, Any]) -> RetrieverConfig:
    """Construct a :class:`RetrieverConfig`, isolating unknown fields into
    ``extra``."""
    if not isinstance(raw, dict):
        raw = {}

    known_keys = {
        "type", "collection", "persist_directory", "index_name",
        "namespace", "host", "port", "api_key", "http", "extra",
    }
    extra = {k: v for k, v in raw.items() if k not in known_keys}

    return RetrieverConfig(
        type=_str(raw, "type", "langchain"),
        collection=_str(raw, "collection", ""),
        persist_directory=_str(raw, "persist_directory", ""),
        index_name=_str(raw, "index_name", ""),
        namespace=_str(raw, "namespace", ""),
        host=_str(raw, "host", ""),
        port=_int(raw, "port", 6333),
        api_key=_str(raw, "api_key", ""),
        http=_build_http_config(raw.get("http") or {}),
        extra={**dict(raw.get("extra") or {}), **extra},
    )


def _build_embedder_config(raw: Dict[str, Any]) -> EmbedderConfig:
    if not isinstance(raw, dict):
        raw = {}

    return EmbedderConfig(
        provider=_str(raw, "provider", "openai"),
        model=_str(raw, "model", "text-embedding-3-small"),
        dimensions=_int(raw, "dimensions", 0),
        batch_size=_int(raw, "batch_size", 32),
    )


def _build_scoring_config(raw: Dict[str, Any]) -> ScoringConfig:
    if not isinstance(raw, dict):
        raw = {}

    return ScoringConfig(
        recall_threshold=_float(raw, "recall_threshold", 0.8),
        fail_on_regression=_bool(raw, "fail_on_regression", True),
    )


def _build_baseline_config(raw: Dict[str, Any]) -> BaselineConfig:
    if not isinstance(raw, dict):
        raw = {}

    return BaselineConfig(
        db_path=_str(raw, "db_path", ".longprobe/baselines.db"),
        auto_compare=_bool(raw, "auto_compare", True),
    )


def _build_generator_config(raw: Dict[str, Any]) -> GeneratorConfig:
    if not isinstance(raw, dict):
        raw = {}

    return GeneratorConfig(
        provider=_str(raw, "provider", "openai"),
        model=_str(raw, "model", "gpt-4o-mini"),
        api_key=_str(raw, "api_key", ""),
        base_url=_str(raw, "base_url", ""),
        num_questions=_int(raw, "num_questions", 50),
        temperature=_float(raw, "temperature", 0.7),
        max_tokens=_int(raw, "max_tokens", 4096),
    )


# ---------------------------------------------------------------------------
# Type-safe field extractors
# ---------------------------------------------------------------------------

def _str(d: Dict[str, Any], key: str, default: str) -> str:
    val = d.get(key, default)
    return str(val) if val is not None else default


def _int(d: Dict[str, Any], key: str, default: int) -> int:
    val = d.get(key, default)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _float(d: Dict[str, Any], key: str, default: float) -> float:
    val = d.get(key, default)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _bool(d: Dict[str, Any], key: str, default: bool) -> bool:
    val = d.get(key, default)
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    # Accept common string truthy/falsy values.
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)
