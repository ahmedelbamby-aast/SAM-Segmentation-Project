"""Unit tests for src/class_registry.py.

Tests every public method of ClassRegistry, covering:
- Identity remapping (no class_remapping)
- Many-to-one remapping (5 prompts → 2 output classes)
- Validation errors (empty prompts, unknown keys, empty strings)
- Serialisation round-trip (to_dict / from_dict)
- Factory construction (from_config)
- Edge cases: single class, all prompts remapped to same class

Author: Ahmed Hany ElBamby
Date: 22-02-2026
"""
from __future__ import annotations

import pytest
from typing import Dict, List, Optional
from unittest.mock import MagicMock

from src.class_registry import ClassRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def identity_registry() -> ClassRegistry:
    """2-class registry with no remapping."""
    return ClassRegistry(prompts=["teacher", "student"])


@pytest.fixture()
def many_to_one_registry() -> ClassRegistry:
    """5-prompt → 2-class registry (the canonical example from the plan)."""
    return ClassRegistry(
        prompts=["teacher", "student", "kid", "child", "Adult"],
        class_remapping={"kid": "student", "child": "student", "Adult": "teacher"},
    )


# ---------------------------------------------------------------------------
# TestInstantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_identity_two_classes(self, identity_registry):
        assert identity_registry.num_prompts == 2
        assert identity_registry.num_classes == 2
        assert identity_registry.class_names == ["teacher", "student"]
        assert identity_registry.prompts == ["teacher", "student"]

    def test_many_to_one_five_to_two(self, many_to_one_registry):
        r = many_to_one_registry
        assert r.num_prompts == 5
        assert r.num_classes == 2
        assert r.class_names == ["teacher", "student"]

    def test_single_class_no_remapping(self):
        r = ClassRegistry(prompts=["person"])
        assert r.num_classes == 1
        assert r.class_names == ["person"]

    def test_all_remapped_to_one_class(self):
        r = ClassRegistry(
            prompts=["a", "b", "c"],
            class_remapping={"b": "a", "c": "a"},
        )
        assert r.num_classes == 1
        assert r.class_names == ["a"]

    def test_ten_prompts_identity(self):
        prompts = [f"class_{i}" for i in range(10)]
        r = ClassRegistry(prompts=prompts)
        assert r.num_classes == 10
        assert r.class_names == prompts


# ---------------------------------------------------------------------------
# TestRemapPromptIndex
# ---------------------------------------------------------------------------

class TestRemapPromptIndex:
    def test_identity_index_zero(self, identity_registry):
        assert identity_registry.remap_prompt_index(0) == 0  # teacher → 0

    def test_identity_index_one(self, identity_registry):
        assert identity_registry.remap_prompt_index(1) == 1  # student → 1

    def test_many_to_one_teacher_prompt(self, many_to_one_registry):
        # prompt 0 = "teacher" → output id 0
        assert many_to_one_registry.remap_prompt_index(0) == 0

    def test_many_to_one_student_prompt(self, many_to_one_registry):
        # prompt 1 = "student" → output id 1
        assert many_to_one_registry.remap_prompt_index(1) == 1

    def test_many_to_one_kid_prompt(self, many_to_one_registry):
        # prompt 2 = "kid" → "student" → output id 1
        assert many_to_one_registry.remap_prompt_index(2) == 1

    def test_many_to_one_child_prompt(self, many_to_one_registry):
        # prompt 3 = "child" → "student" → output id 1
        assert many_to_one_registry.remap_prompt_index(3) == 1

    def test_many_to_one_adult_prompt(self, many_to_one_registry):
        # prompt 4 = "Adult" → "teacher" → output id 0
        assert many_to_one_registry.remap_prompt_index(4) == 0

    def test_out_of_range_raises(self, identity_registry):
        with pytest.raises(ValueError, match="out of range"):
            identity_registry.remap_prompt_index(99)

    def test_negative_index_raises(self, identity_registry):
        with pytest.raises(ValueError):
            identity_registry.remap_prompt_index(-1)


# ---------------------------------------------------------------------------
# TestGetClassName
# ---------------------------------------------------------------------------

class TestGetClassName:
    def test_get_teacher(self, identity_registry):
        assert identity_registry.get_class_name(0) == "teacher"

    def test_get_student(self, identity_registry):
        assert identity_registry.get_class_name(1) == "student"

    def test_many_to_one_teacher_output(self, many_to_one_registry):
        assert many_to_one_registry.get_class_name(0) == "teacher"

    def test_many_to_one_student_output(self, many_to_one_registry):
        assert many_to_one_registry.get_class_name(1) == "student"

    def test_out_of_range_raises(self, identity_registry):
        with pytest.raises(ValueError, match="out of range"):
            identity_registry.get_class_name(99)

    def test_negative_raises(self, identity_registry):
        with pytest.raises(ValueError):
            identity_registry.get_class_name(-1)


# ---------------------------------------------------------------------------
# TestGetYoloNames
# ---------------------------------------------------------------------------

class TestGetYoloNames:
    def test_identity_yolo_names(self, identity_registry):
        expected = {0: "teacher", 1: "student"}
        assert identity_registry.get_yolo_names() == expected

    def test_many_to_one_yolo_names(self, many_to_one_registry):
        expected = {0: "teacher", 1: "student"}
        assert many_to_one_registry.get_yolo_names() == expected

    def test_single_class_yolo_names(self):
        r = ClassRegistry(prompts=["person"])
        assert r.get_yolo_names() == {0: "person"}

    def test_returns_new_dict_each_call(self, identity_registry):
        d1 = identity_registry.get_yolo_names()
        d2 = identity_registry.get_yolo_names()
        assert d1 == d2
        assert d1 is not d2  # independent copies


# ---------------------------------------------------------------------------
# TestGetOutputIdForPromptName
# ---------------------------------------------------------------------------

class TestGetOutputIdForPromptName:
    def test_teacher_by_name(self, identity_registry):
        assert identity_registry.get_output_id_for_prompt_name("teacher") == 0

    def test_student_by_name(self, identity_registry):
        assert identity_registry.get_output_id_for_prompt_name("student") == 1

    def test_remapped_kid(self, many_to_one_registry):
        assert many_to_one_registry.get_output_id_for_prompt_name("kid") == 1

    def test_remapped_adult(self, many_to_one_registry):
        assert many_to_one_registry.get_output_id_for_prompt_name("Adult") == 0

    def test_unknown_name_raises(self, identity_registry):
        with pytest.raises(ValueError, match="not in prompts"):
            identity_registry.get_output_id_for_prompt_name("ghost")


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_prompts_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ClassRegistry(prompts=[])

    def test_empty_string_prompt_raises(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            ClassRegistry(prompts=["teacher", ""])

    def test_non_string_prompt_raises(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            ClassRegistry(prompts=["teacher", None])  # type: ignore[list-item]

    def test_unknown_remapping_key_raises(self):
        with pytest.raises(ValueError, match="not in prompts"):
            ClassRegistry(
                prompts=["teacher", "student"],
                class_remapping={"ghost": "teacher"},
            )

    def test_empty_remapping_value_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ClassRegistry(
                prompts=["teacher", "student"],
                class_remapping={"student": ""},
            )

    def test_none_remapping_value_raises(self):
        with pytest.raises((ValueError, TypeError)):
            ClassRegistry(
                prompts=["teacher", "student"],
                class_remapping={"student": None},  # type: ignore[dict-item]
            )

    def test_valid_many_to_one_passes(self):
        # Should not raise
        r = ClassRegistry(
            prompts=["teacher", "student", "kid"],
            class_remapping={"kid": "student"},
        )
        assert r.num_classes == 2


# ---------------------------------------------------------------------------
# TestSerialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_dict_identity(self, identity_registry):
        d = identity_registry.to_dict()
        assert d["prompts"] == ["teacher", "student"]
        assert d["class_remapping"] == {}

    def test_to_dict_many_to_one(self, many_to_one_registry):
        d = many_to_one_registry.to_dict()
        assert d["prompts"] == ["teacher", "student", "kid", "child", "Adult"]
        assert d["class_remapping"] == {"kid": "student", "child": "student", "Adult": "teacher"}

    def test_round_trip_identity(self, identity_registry):
        d = identity_registry.to_dict()
        restored = ClassRegistry.from_dict(d)
        assert restored == identity_registry

    def test_round_trip_many_to_one(self, many_to_one_registry):
        d = many_to_one_registry.to_dict()
        restored = ClassRegistry.from_dict(d)
        assert restored == many_to_one_registry
        # Verify remapping is preserved
        assert restored.remap_prompt_index(4) == 0  # Adult → teacher

    def test_from_dict_no_remapping_key(self):
        """from_dict must accept dicts without class_remapping key."""
        r = ClassRegistry.from_dict({"prompts": ["cat", "dog"]})
        assert r.num_classes == 2
        assert r.class_remapping == {}

    def test_to_dict_returns_copies(self, identity_registry):
        d = identity_registry.to_dict()
        d["prompts"].append("ghost")
        # original must not be mutated
        assert identity_registry.prompts == ["teacher", "student"]


# ---------------------------------------------------------------------------
# TestFromConfig
# ---------------------------------------------------------------------------

class TestFromConfig:
    def _make_model_config(
        self,
        prompts: List[str],
        class_remapping: Optional[Dict[str, str]] = None,
    ) -> MagicMock:
        cfg = MagicMock()
        cfg.prompts = prompts
        cfg.class_remapping = class_remapping
        return cfg

    def test_basic_from_config(self):
        cfg = self._make_model_config(["teacher", "student"])
        r = ClassRegistry.from_config(cfg)
        assert r.class_names == ["teacher", "student"]

    def test_from_config_with_remapping(self):
        cfg = self._make_model_config(
            ["teacher", "student", "kid"],
            {"kid": "student"},
        )
        r = ClassRegistry.from_config(cfg)
        assert r.num_classes == 2
        assert r.remap_prompt_index(2) == 1

    def test_from_config_no_remapping_attr(self):
        """from_config must handle configs without class_remapping attribute."""
        cfg = MagicMock(spec=["prompts"])
        cfg.prompts = ["a", "b"]
        r = ClassRegistry.from_config(cfg)
        assert r.num_classes == 2

    def test_from_config_none_remapping(self):
        cfg = self._make_model_config(["teacher", "student"], None)
        r = ClassRegistry.from_config(cfg)
        assert r.class_remapping == {}


# ---------------------------------------------------------------------------
# TestEquality
# ---------------------------------------------------------------------------

class TestEquality:
    def test_equal_registries(self):
        r1 = ClassRegistry(["teacher", "student"])
        r2 = ClassRegistry(["teacher", "student"])
        assert r1 == r2

    def test_different_prompts_not_equal(self):
        r1 = ClassRegistry(["teacher", "student"])
        r2 = ClassRegistry(["cat", "dog"])
        assert r1 != r2

    def test_different_remapping_not_equal(self):
        r1 = ClassRegistry(["a", "b", "c"], {"c": "a"})
        r2 = ClassRegistry(["a", "b", "c"], {"c": "b"})
        assert r1 != r2

    def test_not_equal_to_non_registry(self, identity_registry):
        assert identity_registry != "not a registry"
        assert identity_registry != 42


# ---------------------------------------------------------------------------
# TestOutputIdOrdering
# ---------------------------------------------------------------------------

class TestOutputIdOrdering:
    def test_first_seen_order_preserved(self):
        """Output IDs assigned in first-seen order from left to right."""
        r = ClassRegistry(
            prompts=["adult", "child", "teen"],
            class_remapping={"teen": "adult"},
        )
        # adult seen first → id 0, child → id 1
        assert r.remap_prompt_index(0) == 0  # adult
        assert r.remap_prompt_index(1) == 1  # child
        assert r.remap_prompt_index(2) == 0  # teen → adult → 0

    def test_num_classes_correct_after_merge(self):
        r = ClassRegistry(
            prompts=["a", "b", "c", "d"],
            class_remapping={"c": "a", "d": "b"},
        )
        assert r.num_classes == 2
