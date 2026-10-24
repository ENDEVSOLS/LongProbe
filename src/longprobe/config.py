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
from typing import Any

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
    headers: dict[str, str] = field(default_factory=dict)
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
    extra: dict[str, Any] = field(default_factory=dict)


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
    question_thresholds: dict[str, float] = field(default_factory=dict)


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
            against the literal label ``latest`` when it exists.
    """

    db_path: str = ".longprobe/baselines.db"
    auto_compare: bool = True
    quality_gate: str = "all_questions"


# ---------------------------------------------------------------------------
# ProbeConfig
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    """Configuration for the question generation module.

    Attributes:
        provider: LLM provider to use.  One of ``"openai"``, ``"anthropic"``,
            ``"gemini"``, ``"ollama"``, or any provider supported by litellm.
        model: Model identifier (e.g. ``"gpt-5.5-instant"``, ``"claude-haiku-4-5"``).
        api_key: API key for the LLM provider.  Supports ``${ENV_VAR}`` expansion.
        base_url: Custom base URL for the provider (useful for Ollama or
            self-hosted endpoints).
        num_questions: Default number of questions to generate.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens in the LLM response.
    """

    provider: str = "openai"
    model: str = "gpt-5.5-instant"
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

    @classmethod
    def defaults(cls) -> ProbeConfig:
        """Return documented defaults for deliberate programmatic use."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbeConfig:
        """Parse supplied values strictly; omitted fields retain defaults."""
        return _build_section(cls, _expand_env_recursive(data), "")

    @classmethod
    def from_yaml(cls, path: str) -> ProbeConfig:
        """Load a configuration without silently replacing invalid input."""
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            raise ConfigurationError(
                f"Invalid YAML in {path}{location}; correct the syntax."
            ) from None
        except UnicodeError:
            raise ConfigurationError(f"Configuration {path} must contain UTF-8 text.") from None
        return cls.from_dict(data)

    def validate_for(self, operation: str, *, owns_retriever: bool = True) -> None:
        """Validate only providers used by this operation, before client calls."""
        from dataclasses import asdict
        from urllib.parse import urlsplit

        normalized = self.from_dict(asdict(self))
        for name in ("retriever", "embedder", "scoring", "baseline", "generator"):
            setattr(self, name, getattr(normalized, name))

        if operation in {"evaluate", "baseline"}:
            _active(self.baseline.db_path, "baseline.db_path", required=True)
        if owns_retriever and operation in {"evaluate", "capture", "auto_capture"}:
            r = self.retriever
            if r.type not in {"chroma", "pinecone", "qdrant", "http"}:
                raise ConfigurationError(
                    "retriever.type must select chroma, pinecone, qdrant, or http; use the Python API for framework retrievers."
                )
            if r.type == "chroma":
                _active(r.collection, "retriever.collection", required=True)
                _active(r.persist_directory, "retriever.persist_directory")
            elif r.type == "http":
                _active(asdict(r.http), "retriever.http")
                try:
                    url = urlsplit(r.http.url)
                    valid = url.scheme in {"http", "https"} and bool(url.hostname)
                    valid = valid and (url.port is None or 1 <= url.port <= 65535)
                except ValueError:
                    valid = False
                if not valid:
                    raise ConfigurationError("retriever.http.url must be an HTTP(S) endpoint URL.")
                if r.http.method.upper() not in {"POST", "GET"}:
                    raise ConfigurationError("retriever.http.method must be POST or GET.")
            elif r.type == "pinecone":
                _active(r.index_name, "retriever.index_name", required=True)
                _active(r.namespace, "retriever.namespace")
                _credential(r.api_key, "retriever.api_key", "PINECONE_API_KEY")
            else:
                _active(r.collection, "retriever.collection", required=True)
                _active(r.host, "retriever.host", required=True)
                _active(r.api_key, "retriever.api_key")
        if operation in {"generate", "auto_capture"}:
            g = self.generator
            _active(asdict(g), "generator")
            _active(g.provider, "generator.provider", required=True)
            _active(g.model, "generator.model", required=True)
            if g.provider.lower() not in {"ollama", "local"}:
                env_name = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "gemini": "GEMINI_API_KEY",
                    "google": "GOOGLE_API_KEY",
                }.get(g.provider.lower(), f"{g.provider.upper()}_API_KEY")
                _credential(g.api_key, "generator.api_key", env_name)


class ConfigurationError(ValueError):
    """Safe, actionable configuration failure without input values."""


def _expand_env_recursive(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_recursive(v) for v in obj]
    return obj


def _build_section(cls, raw: Any, path: str):
    import math
    from dataclasses import fields, is_dataclass

    if not isinstance(raw, dict):
        raise ConfigurationError(f"{path or 'configuration'} must be a mapping.")
    defaults = cls()
    values = {}
    known = {f.name for f in fields(cls)}
    for f in fields(cls):
        default = getattr(defaults, f.name)
        value = raw.get(f.name, default)
        key = f"{path}.{f.name}" if path else f.name
        if is_dataclass(default):
            values[f.name] = _build_section(type(default), raw.get(f.name, {}), key)
            continue
        if isinstance(default, dict):
            if not isinstance(value, dict):
                raise ConfigurationError(f"{key} must be a mapping.")
            if f.name == "headers" and any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()
            ):
                raise ConfigurationError(f"{key} must map header names to strings.")
            if key == "scoring.question_thresholds" and any(
                not isinstance(k, str)
                or not k.strip()
                or isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0 <= v <= 1
                for k, v in value.items()
            ):
                raise ConfigurationError(
                    "scoring.question_thresholds must map question IDs to finite numbers between 0 and 1."
                )
        elif isinstance(default, bool):
            forms = {
                "true": True,
                "1": True,
                "yes": True,
                "on": True,
                "false": False,
                "0": False,
                "no": False,
                "off": False,
            }
            if isinstance(value, str) and value.lower() in forms:
                value = forms[value.lower()]
            if not isinstance(value, bool):
                raise ConfigurationError(f"{key} must be a boolean (true or false).")
        elif isinstance(default, int):
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                value = int(value)
            if type(value) is not int:
                raise ConfigurationError(f"{key} must be an integer.")
        elif isinstance(default, float):
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    raise ConfigurationError(f"{key} must be a finite number.") from None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ConfigurationError(f"{key} must be a finite number.")
            value = float(value)
        elif not isinstance(value, str):
            raise ConfigurationError(f"{key} must be a string.")
        if f.name in {"batch_size", "timeout", "num_questions", "max_tokens"} and value <= 0:
            raise ConfigurationError(f"{key} must be positive.")
        if key == "scoring.recall_threshold" and not 0 <= value <= 1:
            raise ConfigurationError(f"{key} must be between 0 and 1.")
        if key == "baseline.quality_gate" and value not in {"all_questions", "aggregate"}:
            raise ConfigurationError("baseline.quality_gate must be all_questions or aggregate.")
        if key == "retriever.port" and not 1 <= value <= 65535:
            raise ConfigurationError(f"{key} must be between 1 and 65535.")
        if key == "embedder.dimensions" and value < 0:
            raise ConfigurationError(f"{key} must be nonnegative.")
        values[f.name] = value
    if cls is RetrieverConfig:
        values["extra"] = {**values["extra"], **{k: v for k, v in raw.items() if k not in known}}
    return cls(**values)


def _active(value: Any, path: str, *, required: bool = False) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            # Header names and extra keys may contain secrets: do not expose them.
            child_path = path if path.endswith("headers") else f"{path}.{key}"
            _active(child, child_path)
    elif isinstance(value, str):
        missing = _ENV_PATTERN.search(value)
        if missing:
            raise ConfigurationError(
                f"{path} references unset environment variable {missing.group(1)}; set it before running."
            )
        if required and not value.strip():
            raise ConfigurationError(f"{path} must be non-empty.")


def _credential(value: str, path: str, env_name: str) -> None:
    _active(value, path)
    if not value.strip() and not os.environ.get(env_name, "").strip():
        raise ConfigurationError(f"{path} or environment variable {env_name} is required.")
