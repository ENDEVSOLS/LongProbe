"""
Golden set data models for RAG regression testing.

Provides the core data structures used to define a "golden set" of questions,
each annotated with the expected retrieval results. These golden sets serve
as the ground truth when evaluating whether changes to a retrieval pipeline
introduce regressions.

Typical usage::

    gs = GoldenSet.from_yaml("golden_set.yaml")
    for q in gs.questions:
        retrieved = retriever.retrieve(q.question, top_k=q.top_k)
        # compare retrieved against q.required_chunks
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_MATCH_MODES: Set[str] = {"id", "text", "semantic"}


# ---------------------------------------------------------------------------
# GoldenQuestion
# ---------------------------------------------------------------------------

@dataclass
class GoldenQuestion:
    """A single golden question together with the chunks the retriever *must*
    return for the answer to be considered correct.

    Attributes:
        id: Unique identifier for this question within a golden set.
        question: The natural-language question to send to the retriever.
        required_chunks: List of chunk IDs (or text snippets / embeddings
            depending on *match_mode*) that the retriever is expected to return.
        match_mode: Strategy used to compare retrieved chunks against the
            expected set.  One of ``"id"``, ``"text"``, ``"semantic"``.
        semantic_threshold: Cosine-similarity threshold used when
            *match_mode* is ``"semantic"``.  Chunks scoring below this value
            are treated as non-matching.
        top_k: How many results the retriever should return.
        tags: Free-form tags for filtering / grouping questions.
        metadata: Arbitrary additional metadata attached to this question.
    """

    id: str
    question: str
    required_chunks: List[str]
    match_mode: str = "id"
    semantic_threshold: float = 0.85
    top_k: int = 5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GoldenSet
# ---------------------------------------------------------------------------

@dataclass
class GoldenSet:
    """A named collection of :class:`GoldenQuestion` instances that together
    form the regression test suite for a retrieval pipeline.

    Attributes:
        name: Human-readable name for this golden set.
        version: Semantic version string for this golden set iteration.
        questions: The list of golden questions.
    """

    name: str
    version: str
    questions: List[GoldenQuestion] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_questions(questions: List[Dict[str, Any]]) -> None:
        """Run all validation rules over the raw question dicts.

        Raises:
            ValueError: On the *first* validation failure encountered, with a
                human-readable message.
        """
        seen_ids: Set[str] = set()

        for idx, raw in enumerate(questions):
            qid = raw.get("id")

            # -- unique id ------------------------------------------------
            if qid is None or str(qid).strip() == "":
                raise ValueError(
                    f"Question at index {idx} is missing a non-empty 'id'."
                )
            qid = str(qid).strip()
            if qid in seen_ids:
                raise ValueError(
                    f"Duplicate question id '{qid}' "
                    f"(first occurrence kept, second at index {idx})."
                )
            seen_ids.add(qid)

            # -- question text --------------------------------------------
            if not raw.get("question", "").strip():
                raise ValueError(
                    f"Question '{qid}' has an empty 'question' field."
                )

            # -- required_chunks ------------------------------------------
            required = raw.get("required_chunks")
            if not required or not isinstance(required, list) or len(required) == 0:
                raise ValueError(
                    f"Question '{qid}' must have a non-empty 'required_chunks' list."
                )

            # -- match_mode -----------------------------------------------
            match_mode = str(raw.get("match_mode", "id")).strip().lower()
            if match_mode not in _VALID_MATCH_MODES:
                raise ValueError(
                    f"Question '{qid}' has invalid match_mode '{match_mode}'. "
                    f"Must be one of {sorted(_VALID_MATCH_MODES)}."
                )

            # -- top_k ----------------------------------------------------
            top_k = raw.get("top_k", 5)
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(
                    f"Question '{qid}' has invalid top_k={top_k!r}. "
                    f"Must be a positive integer."
                )

            # -- semantic_threshold when match_mode == "semantic" ----------
            if match_mode == "semantic":
                threshold = raw.get("semantic_threshold")
                if threshold is None:
                    raise ValueError(
                        f"Question '{qid}' uses match_mode='semantic' but does "
                        f"not specify 'semantic_threshold'."
                    )
                try:
                    threshold_val = float(threshold)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Question '{qid}' has non-numeric semantic_threshold="
                        f"{threshold!r}."
                    )
                if not (0.0 <= threshold_val <= 1.0):
                    raise ValueError(
                        f"Question '{qid}' has semantic_threshold={threshold_val}, "
                        f"which is outside the valid range [0.0, 1.0]."
                    )

            # -- semantic_threshold range for non-semantic modes -----------
            if match_mode != "semantic":
                threshold = raw.get("semantic_threshold")
                if threshold is not None:
                    try:
                        threshold_val = float(threshold)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"Question '{qid}' has non-numeric semantic_threshold="
                            f"{threshold!r}."
                        )
                    if not (0.0 <= threshold_val <= 1.0):
                        raise ValueError(
                            f"Question '{qid}' has semantic_threshold={threshold_val}, "
                            f"which is outside the valid range [0.0, 1.0]."
                        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "GoldenSet":
        """Load a golden set from a YAML file.

        The file must contain ``name``, ``version``, and ``questions`` keys.
        Each entry in ``questions`` is deserialized into a
        :class:`GoldenQuestion` and validated.

        Args:
            path: Filesystem path to the YAML file.

        Returns:
            A fully-validated :class:`GoldenSet` instance.

        Raises:
            ValueError: If validation fails.
            FileNotFoundError: If *path* does not exist.
            yaml.YAMLError: If the file cannot be parsed as YAML.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Golden-set file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a YAML mapping at the top level of '{path}', "
                f"got {type(data).__name__}."
            )

        name = data.get("name", "")
        version = data.get("version", "")

        if not name:
            raise ValueError("Golden set is missing a non-empty 'name' field.")
        if not version:
            raise ValueError("Golden set is missing a non-empty 'version' field.")

        raw_questions = data.get("questions")
        if raw_questions is None:
            raise ValueError("Golden set is missing the 'questions' key.")
        if not isinstance(raw_questions, list):
            raise ValueError(
                f"'questions' must be a list, got {type(raw_questions).__name__}."
            )

        # Validate *before* we start building objects so we fail fast with a
        # clear message.
        cls._validate_questions(raw_questions)

        questions: List[GoldenQuestion] = []
        for raw in raw_questions:
            questions.append(
                GoldenQuestion(
                    id=str(raw["id"]).strip(),
                    question=str(raw["question"]).strip(),
                    required_chunks=[str(c) for c in raw["required_chunks"]],
                    match_mode=str(raw.get("match_mode", "id")).strip().lower(),
                    semantic_threshold=float(raw.get("semantic_threshold", 0.85)),
                    top_k=int(raw.get("top_k", 5)),
                    tags=[str(t) for t in raw.get("tags", [])],
                    metadata=dict(raw.get("metadata", {})),
                )
            )

        return cls(name=name, version=version, questions=questions)

    def to_yaml(self, path: str) -> None:
        """Serialize this golden set to a YAML file.

        Args:
            path: Destination filesystem path.  Parent directories are created
                automatically if they do not exist.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        data = {
            "name": self.name,
            "version": self.version,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "required_chunks": q.required_chunks,
                    "match_mode": q.match_mode,
                    "semantic_threshold": q.semantic_threshold,
                    "top_k": q.top_k,
                    "tags": q.tags,
                    "metadata": q.metadata,
                }
                for q in self.questions
            ],
        }

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(
                data,
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    # ------------------------------------------------------------------
    # Filtering & merging
    # ------------------------------------------------------------------

    def filter_by_tags(self, tags: List[str]) -> "GoldenSet":
        """Return a new :class:`GoldenSet` containing only questions whose
        ``tags`` list contains **all** of the specified *tags*.

        Args:
            tags: Tag strings that every returned question must have.

        Returns:
            A new :class:`GoldenSet` with the same name and version but a
            filtered question list.  If *tags* is empty, all questions are
            returned (no filtering applied).
        """
        if not tags:
            return GoldenSet(
                name=self.name,
                version=self.version,
                questions=list(self.questions),
            )

        filtered = [
            q for q in self.questions
            if all(t in q.tags for t in tags)
        ]
        return GoldenSet(
            name=self.name,
            version=self.version,
            questions=filtered,
        )

    def merge(self, new_questions: List[GoldenQuestion]) -> int:
        """Append questions that do not already exist (matched by ``id``).

        Duplicate IDs are silently skipped.

        Args:
            new_questions: Questions to add.

        Returns:
            The number of questions actually added.
        """
        existing_ids: Set[str] = {q.id for q in self.questions}
        added = 0
        for q in new_questions:
            if q.id not in existing_ids:
                self.questions.append(q)
                existing_ids.add(q.id)
                added += 1
        return added


# ---------------------------------------------------------------------------
# Question-ID generation
# ---------------------------------------------------------------------------

import re as _re

_SLUG_RE = _re.compile(r"[^a-z0-9]+")


def generate_question_id(
    question_text: str,
    prefix: str = "q",
    existing_ids: Set[str] | None = None,
    max_words: int = 5,
) -> str:
    """Generate a URL-safe, unique question ID from the question text.

    Converts ``"What is the termination clause?"`` into
    ``"q_what_is_the_termination_clause"``.  If the resulting ID already
    exists in *existing_ids*, a numeric suffix is appended
    (``q_what_is_the_termination_clause_2``).

    Args:
        question_text: The natural-language question.
        prefix: Short prefix for all generated IDs.
        existing_ids: Set of IDs already in use.  When provided, the
            generated ID is guaranteed to be unique within this set.
        max_words: Maximum number of words to include in the slug.

    Returns:
        A unique, lowercase, underscore-separated identifier string.
    """
    if existing_ids is None:
        existing_ids = set()

    # Lowercase, strip punctuation, collapse whitespace → slug
    slug = question_text.lower().strip()
    slug = _SLUG_RE.sub("_", slug).strip("_")

    # Truncate to max_words
    parts = slug.split("_")
    if len(parts) > max_words:
        parts = parts[:max_words]
    slug = "_".join(parts)

    candidate = f"{prefix}_{slug}" if prefix else slug

    # Ensure uniqueness
    if candidate not in existing_ids:
        return candidate

    counter = 2
    while f"{candidate}_{counter}" in existing_ids:
        counter += 1
    return f"{candidate}_{counter}"
