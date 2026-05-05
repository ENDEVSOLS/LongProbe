"""
Tests for ``longprobe.core.golden`` — GoldenSet and GoldenQuestion data models.

Covers loading from YAML, validation of every constraint, default values,
and round-trip serialisation.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from longprobe.core.golden import GoldenQuestion, GoldenSet, generate_question_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path, data: dict[str, Any]) -> str:
    """Write *data* as YAML to a temp file and return the path."""
    p = tmp_path / "golden.yaml"
    p.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return str(p)


def _minimal_yaml_data(**overrides: Any) -> dict[str, Any]:
    """Return a valid minimal golden-set YAML dict, with optional overrides."""
    data: dict[str, Any] = {
        "name": "test-set",
        "version": "1.0.0",
        "questions": [
            {
                "id": "q1",
                "question": "What is Python?",
                "required_chunks": ["chunk-1", "chunk-2"],
                "match_mode": "id",
                "top_k": 5,
            }
        ],
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Loading valid YAML
# ---------------------------------------------------------------------------

class TestLoadValidYaml:

    def test_load_basic_valid_file(self, tmp_path):
        """A well-formed YAML file loads without error."""
        path = _write_yaml(tmp_path, _minimal_yaml_data())
        gs = GoldenSet.from_yaml(path)

        assert gs.name == "test-set"
        assert gs.version == "1.0.0"
        assert len(gs.questions) == 1
        assert gs.questions[0].id == "q1"
        assert gs.questions[0].question == "What is Python?"

    def test_load_multiple_questions(self, tmp_path):
        """All questions in the YAML are loaded."""
        data = _minimal_yaml_data()
        data["questions"].append(
            {
                "id": "q2",
                "question": "What is Rust?",
                "required_chunks": ["chunk-3"],
            }
        )
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        assert len(gs.questions) == 2
        assert {q.id for q in gs.questions} == {"q1", "q2"}


# ---------------------------------------------------------------------------
# All three match modes
# ---------------------------------------------------------------------------

class TestMatchModes:

    def test_id_match_mode(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["match_mode"] = "id"
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)
        assert gs.questions[0].match_mode == "id"

    def test_text_match_mode(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["match_mode"] = "text"
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)
        assert gs.questions[0].match_mode == "text"

    def test_semantic_match_mode(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["match_mode"] = "semantic"
        data["questions"][0]["semantic_threshold"] = 0.9
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)
        assert gs.questions[0].match_mode == "semantic"
        assert gs.questions[0].semantic_threshold == 0.9


# ---------------------------------------------------------------------------
# Validation: duplicate question IDs
# ---------------------------------------------------------------------------

class TestDuplicateIds:

    def test_duplicate_question_id_raises_value_error(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"].append(
            {
                "id": "q1",  # duplicate!
                "question": "Another question?",
                "required_chunks": ["chunk-x"],
            }
        )
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ValueError, match="Duplicate question id"):
            GoldenSet.from_yaml(path)


# ---------------------------------------------------------------------------
# Validation: empty required_chunks
# ---------------------------------------------------------------------------

class TestEmptyRequiredChunks:

    def test_empty_required_chunks_list_raises(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["required_chunks"] = []
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ValueError, match="non-empty 'required_chunks'"):
            GoldenSet.from_yaml(path)

    def test_missing_required_chunks_key_raises(self, tmp_path):
        data = _minimal_yaml_data()
        del data["questions"][0]["required_chunks"]
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ValueError, match="non-empty 'required_chunks'"):
            GoldenSet.from_yaml(path)


# ---------------------------------------------------------------------------
# Validation: invalid match_mode
# ---------------------------------------------------------------------------

class TestInvalidMatchMode:

    def test_invalid_match_mode_raises(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["match_mode"] = "fuzzy"
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ValueError, match="invalid match_mode"):
            GoldenSet.from_yaml(path)


# ---------------------------------------------------------------------------
# Validation: semantic_threshold out of range
# ---------------------------------------------------------------------------

class TestSemanticThresholdRange:

    def test_threshold_negative_raises(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["match_mode"] = "semantic"
        data["questions"][0]["semantic_threshold"] = -0.1
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ValueError, match="outside the valid range"):
            GoldenSet.from_yaml(path)

    def test_threshold_above_one_raises(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["match_mode"] = "semantic"
        data["questions"][0]["semantic_threshold"] = 1.5
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ValueError, match="outside the valid range"):
            GoldenSet.from_yaml(path)

    def test_threshold_out_of_range_non_semantic_also_raises(self, tmp_path):
        """Even for non-semantic modes, an out-of-range threshold is rejected."""
        data = _minimal_yaml_data()
        data["questions"][0]["match_mode"] = "id"
        data["questions"][0]["semantic_threshold"] = 2.0
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ValueError, match="outside the valid range"):
            GoldenSet.from_yaml(path)


# ---------------------------------------------------------------------------
# Validation: top_k <= 0
# ---------------------------------------------------------------------------

class TestTopKValidation:

    def test_top_k_zero_raises(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["top_k"] = 0
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ValueError, match="invalid top_k"):
            GoldenSet.from_yaml(path)

    def test_top_k_negative_raises(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["top_k"] = -3
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ValueError, match="invalid top_k"):
            GoldenSet.from_yaml(path)


# ---------------------------------------------------------------------------
# Tags and metadata are preserved
# ---------------------------------------------------------------------------

class TestTagsAndMetadata:

    def test_tags_preserved(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["tags"] = ["rag", "python", "basics"]
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        assert gs.questions[0].tags == ["rag", "python", "basics"]

    def test_metadata_preserved(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"][0]["metadata"] = {
            "category": "programming",
            "difficulty": 1,
        }
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        assert gs.questions[0].metadata == {
            "category": "programming",
            "difficulty": 1,
        }

    def test_empty_tags_and_metadata(self, tmp_path):
        """Tags and metadata default to empty when omitted."""
        path = _write_yaml(tmp_path, _minimal_yaml_data())
        gs = GoldenSet.from_yaml(path)

        assert gs.questions[0].tags == []
        assert gs.questions[0].metadata == {}


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_default_match_mode_is_id(self, tmp_path):
        data = _minimal_yaml_data()
        # Deliberately omit match_mode
        del data["questions"][0]["match_mode"]
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        assert gs.questions[0].match_mode == "id"

    def test_default_semantic_threshold(self, tmp_path):
        data = _minimal_yaml_data()
        # Omit semantic_threshold entirely
        if "semantic_threshold" in data["questions"][0]:
            del data["questions"][0]["semantic_threshold"]
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        assert gs.questions[0].semantic_threshold == 0.85

    def test_default_top_k(self, tmp_path):
        data = _minimal_yaml_data()
        if "top_k" in data["questions"][0]:
            del data["questions"][0]["top_k"]
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        assert gs.questions[0].top_k == 5


# ---------------------------------------------------------------------------
# to_yaml round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_to_yaml_round_trip(self, tmp_path):
        """Save a GoldenSet and reload it — all fields must match."""
        original_data = _minimal_yaml_data()
        original_data["questions"].append(
            {
                "id": "q2",
                "question": "Explain async in Python",
                "required_chunks": ["chunk-3", "chunk-4"],
                "match_mode": "text",
                "top_k": 10,
                "tags": ["async", "python"],
                "metadata": {"source": "docs.python.org"},
            }
        )

        load_path = _write_yaml(tmp_path, original_data)
        gs = GoldenSet.from_yaml(load_path)

        # Save to a new file
        save_path = str(tmp_path / "round_trip.yaml")
        gs.to_yaml(save_path)

        # Reload and verify
        gs2 = GoldenSet.from_yaml(save_path)

        assert gs2.name == gs.name
        assert gs2.version == gs.version
        assert len(gs2.questions) == len(gs.questions)

        for orig, reloaded in zip(gs.questions, gs2.questions):
            assert reloaded.id == orig.id
            assert reloaded.question == orig.question
            assert reloaded.required_chunks == orig.required_chunks
            assert reloaded.match_mode == orig.match_mode
            assert reloaded.semantic_threshold == orig.semantic_threshold
            assert reloaded.top_k == orig.top_k
            assert reloaded.tags == orig.tags
            assert reloaded.metadata == orig.metadata

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFilterByTags:

    def test_filter_by_single_tag(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"].extend([
            {"id": "q2", "question": "Q2", "required_chunks": ["c2"], "tags": ["doc:legal", "critical"]},
            {"id": "q3", "question": "Q3", "required_chunks": ["c3"], "tags": ["doc:billing"]},
        ])
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        filtered = gs.filter_by_tags(["doc:legal"])
        assert len(filtered.questions) == 1
        assert filtered.questions[0].id == "q2"

    def test_filter_by_multiple_tags(self, tmp_path):
        data = _minimal_yaml_data()
        data["questions"].extend([
            {"id": "q2", "question": "Q2", "required_chunks": ["c2"], "tags": ["doc:legal", "critical"]},
            {"id": "q3", "question": "Q3", "required_chunks": ["c3"], "tags": ["doc:legal", "minor"]},
        ])
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        filtered = gs.filter_by_tags(["doc:legal", "critical"])
        assert len(filtered.questions) == 1
        assert filtered.questions[0].id == "q2"

    def test_filter_empty_tags_returns_all(self, tmp_path):
        data = _minimal_yaml_data()
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        filtered = gs.filter_by_tags([])
        assert len(filtered.questions) == 1

# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

class TestMerge:

    def test_merge_new_questions(self, tmp_path):
        data = _minimal_yaml_data()
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        new_q = GoldenQuestion(id="q2", question="Q2", required_chunks=["c2"])
        added = gs.merge([new_q])

        assert added == 1
        assert len(gs.questions) == 2
        assert gs.questions[1].id == "q2"

    def test_merge_ignores_duplicates(self, tmp_path):
        data = _minimal_yaml_data()
        path = _write_yaml(tmp_path, data)
        gs = GoldenSet.from_yaml(path)

        dup_q = GoldenQuestion(id="q1", question="Duplicate ID", required_chunks=["cx"])
        new_q = GoldenQuestion(id="q2", question="Q2", required_chunks=["c2"])
        added = gs.merge([dup_q, new_q])

        assert added == 1
        assert len(gs.questions) == 2
        assert gs.questions[1].id == "q2"
        # The original q1 should remain unchanged
        assert gs.questions[0].question == "What is Python?"

# ---------------------------------------------------------------------------
# Generate Question ID
# ---------------------------------------------------------------------------

class TestGenerateQuestionId:

    def test_basic_generation(self):
        q_id = generate_question_id("What is the refund policy?")
        assert q_id == "q_what_is_the_refund_policy"

    def test_punctuation_stripped(self):
        q_id = generate_question_id("Hello, world! What's up?")
        assert q_id == "q_hello_world_what_s_up"

    def test_max_words(self):
        q_id = generate_question_id("One two three four five six seven", max_words=3)
        assert q_id == "q_one_two_three"

    def test_custom_prefix(self):
        q_id = generate_question_id("Test question", prefix="test")
        assert q_id == "test_test_question"
        
    def test_empty_prefix(self):
        q_id = generate_question_id("Test question", prefix="")
        assert q_id == "test_question"

    def test_uniqueness(self):
        existing = {"q_test_question", "q_test_question_2"}
        q_id = generate_question_id("Test question", existing_ids=existing)
        assert q_id == "q_test_question_3"
