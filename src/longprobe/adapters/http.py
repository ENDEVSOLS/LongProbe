"""Generic HTTP retriever adapter for LongProbe.

Sends queries to any HTTP-based RAG API endpoint and maps the JSON
response back to LongProbe's standard format.  This adapter lets LongProbe
test the *entire* retrieval pipeline (including any reranking, prompt
rewriting, or filtering) rather than just the raw vector database.

Typical configuration in ``longprobe.yaml``::

    retriever:
      type: http
      http:
        url: "http://localhost:8000/api/retrieve"
        method: POST
        body_template: '{"query": "{question}", "top_k": {top_k}}'
        headers:
          Authorization: "Bearer ${API_KEY}"
        response_mapping:
          results_path: "data.chunks"
          id_field: "chunk_id"
          text_field: "content"
          score_field: "similarity"
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from .base import AbstractRetrieverAdapter
from ..config import HttpRetrieverConfig

logger = logging.getLogger(__name__)


class HttpAdapter(AbstractRetrieverAdapter):
    """Adapter that queries an HTTP-based RAG API endpoint.

    The request body is built from ``body_template`` by replacing
    ``{question}`` and ``{top_k}`` placeholders.  The JSON response is
    parsed using ``response_mapping`` to locate the results array and
    extract per-result fields.

    Args:
        config: HTTP-specific configuration (URL, method, headers, etc.).
    """

    def __init__(self, config: HttpRetrieverConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Send *query* to the configured HTTP endpoint and return normalised
        results.

        Args:
            query: Natural-language question to send.
            top_k: Number of results to request.

        Returns:
            List of dicts with ``id``, ``text``, ``score``, and ``metadata``
            keys.

        Raises:
            RuntimeError: If the endpoint returns a non-2xx status.
            ValueError: If the body template produces invalid JSON after
                placeholder substitution.
            requests.Timeout: If the request exceeds the configured timeout.
        """
        cfg = self._config

        # 1. Build request body from template.
        body = self._build_body(cfg.body_template, query, top_k)

        # 2. Make HTTP request.
        try:
            response = requests.request(
                method=cfg.method,
                url=cfg.url,
                json=body,
                headers=cfg.headers,
                timeout=cfg.timeout,
            )
        except requests.Timeout as exc:
            raise requests.Timeout(
                f"HTTP adapter request timed out after {cfg.timeout}s "
                f"for URL '{cfg.url}'."
            ) from exc

        # 3. Check status.
        if response.status_code >= 300:
            snippet = response.text[:500] if response.text else "(empty body)"
            raise RuntimeError(
                f"HTTP adapter received status {response.status_code} from "
                f"'{cfg.url}': {snippet}"
            )

        # 4. Parse response.
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"HTTP adapter received non-JSON response from '{cfg.url}': "
                f"{response.text[:200]}"
            ) from exc

        # 5. Resolve results path.
        mapping = cfg.response_mapping
        items = self._resolve_path(data, mapping.results_path)

        if not isinstance(items, list):
            logger.warning(
                "HTTP adapter: path '%s' did not resolve to a list "
                "(got %s). Returning empty results.",
                mapping.results_path,
                type(items).__name__,
            )
            return []

        # 6. Map each item to LongProbe format.
        results: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                logger.warning(
                    "HTTP adapter: skipping non-dict item in results array."
                )
                continue
            results.append(
                {
                    "id": str(item.get(mapping.id_field, "")),
                    "text": str(item.get(mapping.text_field, "")),
                    "score": float(item.get(mapping.score_field, 0.0)),
                    "metadata": {
                        k: v
                        for k, v in item.items()
                        if k not in (mapping.id_field, mapping.text_field, mapping.score_field)
                    },
                }
            )

        return results

    def health_check(self) -> bool:
        """Check if the configured endpoint is reachable.

        Sends a HEAD (or GET for endpoints that don't support HEAD) to the
        URL and returns ``True`` for any response with status < 400.
        Timeouts and connection errors return ``False``.
        """
        try:
            # Try HEAD first (lighter).
            resp = requests.head(
                self._config.url,
                headers=self._config.headers,
                timeout=min(self._config.timeout, 10),
            )
            if resp.status_code == 405:
                # Fallback to GET if HEAD is not allowed.
                resp = requests.get(
                    self._config.url,
                    headers=self._config.headers,
                    timeout=min(self._config.timeout, 10),
                )
            return resp.status_code < 400
        except (requests.RequestException, OSError):
            logger.debug("HTTP adapter health check failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_body(
        template: str,
        query: str,
        top_k: int,
    ) -> Any:
        """Substitute ``{question}`` and ``{top_k}`` in *template* and
        parse the result as JSON.

        ``{question}`` is replaced with a JSON-safe escaped string.
        ``{top_k}`` is replaced with the integer value.

        Returns:
            The parsed JSON object (usually a dict).

        Raises:
            ValueError: If the substituted template is not valid JSON.
        """
        # JSON-escape the question string, then strip surrounding quotes
        # so it fits naturally inside the template's own quoting.
        escaped_question = json.dumps(query)[1:-1]
        body_str = template.replace("{question}", escaped_question)
        body_str = body_str.replace("{top_k}", str(top_k))

        try:
            return json.loads(body_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"HTTP adapter body template produced invalid JSON after "
                f"substitution: {exc}\n"
                f"Template: {template}\n"
                f"Result:   {body_str[:200]}"
            ) from exc

    @staticmethod
    def _resolve_path(data: Any, path: str) -> Any:
        """Traverse *data* using a dot-notation *path*.

        ``\"data.chunks\"`` resolves to ``data[\"data\"][\"chunks\"]``.
        If any segment is missing, an empty list is returned.

        Args:
            data: The parsed JSON response.
            path: Dot-separated key path.

        Returns:
            The value at the path, or an empty list if not found.
        """
        current = data
        parts = path.split(".") if path else []
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return []
        return current
