"""Tests for the audit_serialization script.

This test file verifies that the audit script correctly identifies
GridObj subclasses and categorizes them by serialization status.
"""

import pytest
from pathlib import Path

from cogrid.scripts.audit_serialization import (
    find_gridobj_subclasses,
    categorize_classes,
    format_text_output,
    format_json_output,
    ClassInfo,
    get_cogrid_path,
)


@pytest.fixture
def cogrid_path() -> Path:
    """Get the cogrid directory path."""
    return get_cogrid_path()


@pytest.fixture
def all_classes(cogrid_path: Path) -> list[ClassInfo]:
    """Find all GridObj subclasses."""
    return find_gridobj_subclasses(cogrid_path)


class TestFindGridObjSubclasses:
    """Tests for the find_gridobj_subclasses function."""

    def test_finds_wall(self, all_classes: list[ClassInfo]):
        """Wall should be found in grid_object.py."""
        class_names = [c.name for c in all_classes]
        assert "Wall" in class_names

    def test_finds_floor(self, all_classes: list[ClassInfo]):
        """Floor should be found in grid_object.py."""
        class_names = [c.name for c in all_classes]
        assert "Floor" in class_names

    def test_finds_counter(self, all_classes: list[ClassInfo]):
        """Counter should be found in grid_object.py."""
        class_names = [c.name for c in all_classes]
        assert "Counter" in class_names

    def test_finds_door(self, all_classes: list[ClassInfo]):
        """Door should be found in grid_object.py."""
        class_names = [c.name for c in all_classes]
        assert "Door" in class_names

    def test_finds_key(self, all_classes: list[ClassInfo]):
        """Key should be found in grid_object.py."""
        class_names = [c.name for c in all_classes]
        assert "Key" in class_names

    def test_finds_onion(self, all_classes: list[ClassInfo]):
        """Onion should be found in overcooked_grid_objects.py."""
        class_names = [c.name for c in all_classes]
        assert "Onion" in class_names

    def test_finds_pot(self, all_classes: list[ClassInfo]):
        """Pot should be found in overcooked_grid_objects.py."""
        class_names = [c.name for c in all_classes]
        assert "Pot" in class_names

    def test_finds_plate(self, all_classes: list[ClassInfo]):
        """Plate should be found in overcooked_grid_objects.py."""
        class_names = [c.name for c in all_classes]
        assert "Plate" in class_names

    def test_finds_green_victim(self, all_classes: list[ClassInfo]):
        """GreenVictim should be found in search_rescue_grid_objects.py."""
        class_names = [c.name for c in all_classes]
        assert "GreenVictim" in class_names

    def test_finds_red_victim(self, all_classes: list[ClassInfo]):
        """RedVictim should be found in search_rescue_grid_objects.py."""
        class_names = [c.name for c in all_classes]
        assert "RedVictim" in class_names

    def test_total_class_count(self, all_classes: list[ClassInfo]):
        """Should find at least 20 GridObj subclasses."""
        assert len(all_classes) >= 20


class TestStatusCategorization:
    """Tests for categorizing classes by serialization status."""

    def test_counter_is_implemented(self, all_classes: list[ClassInfo]):
        """Counter should be categorized as IMPLEMENTED."""
        categories = categorize_classes(all_classes)
        implemented_names = [c.name for c in categories["IMPLEMENTED"]]
        assert "Counter" in implemented_names

    def test_pot_is_implemented(self, all_classes: list[ClassInfo]):
        """Pot should be categorized as IMPLEMENTED."""
        categories = categorize_classes(all_classes)
        implemented_names = [c.name for c in categories["IMPLEMENTED"]]
        assert "Pot" in implemented_names

    def test_wall_is_stateless(self, all_classes: list[ClassInfo]):
        """Wall should be categorized as STATELESS."""
        categories = categorize_classes(all_classes)
        stateless_names = [c.name for c in categories["STATELESS"]]
        assert "Wall" in stateless_names

    def test_floor_is_stateless(self, all_classes: list[ClassInfo]):
        """Floor should be categorized as STATELESS."""
        categories = categorize_classes(all_classes)
        stateless_names = [c.name for c in categories["STATELESS"]]
        assert "Floor" in stateless_names

    def test_onion_is_stateless(self, all_classes: list[ClassInfo]):
        """Onion should be categorized as STATELESS."""
        categories = categorize_classes(all_classes)
        stateless_names = [c.name for c in categories["STATELESS"]]
        assert "Onion" in stateless_names

    def test_tomato_is_stateless(self, all_classes: list[ClassInfo]):
        """Tomato should be categorized as STATELESS."""
        categories = categorize_classes(all_classes)
        stateless_names = [c.name for c in categories["STATELESS"]]
        assert "Tomato" in stateless_names

    def test_door_has_extra_attributes(self, all_classes: list[ClassInfo]):
        """Door should have is_open and is_locked as extra attributes."""
        door_class = next((c for c in all_classes if c.name == "Door"), None)
        assert door_class is not None
        assert "is_open" in door_class.extra_attributes
        assert "is_locked" in door_class.extra_attributes


class TestOutputFormat:
    """Tests for output formatting functions."""

    def test_text_output_contains_sections(self, all_classes: list[ClassInfo]):
        """Text output should contain all required sections."""
        categories = categorize_classes(all_classes)
        text = format_text_output(categories)

        assert "=== GridObj Serialization Audit ===" in text
        assert "STATELESS" in text
        assert "IMPLEMENTED" in text
        assert "PARTIAL" in text
        assert "MISSING" in text
        assert "Summary:" in text
        assert "Total classes:" in text

    def test_text_output_no_crashes(self, all_classes: list[ClassInfo]):
        """Text output should not raise any exceptions."""
        categories = categorize_classes(all_classes)
        text = format_text_output(categories)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_json_output_is_valid(self, all_classes: list[ClassInfo]):
        """JSON output should be valid JSON."""
        import json

        categories = categorize_classes(all_classes)
        json_str = format_json_output(categories)

        # Should not raise
        data = json.loads(json_str)

        assert "stateless" in data
        assert "implemented" in data
        assert "partial" in data
        assert "missing" in data
        assert "summary" in data

    def test_json_summary_counts_match(self, all_classes: list[ClassInfo]):
        """JSON summary counts should match category lengths."""
        import json

        categories = categorize_classes(all_classes)
        json_str = format_json_output(categories)
        data = json.loads(json_str)

        assert len(data["stateless"]) == data["summary"]["stateless"]
        assert len(data["implemented"]) == data["summary"]["implemented"]
        assert len(data["partial"]) == data["summary"]["partial"]
        assert len(data["missing"]) == data["summary"]["missing"]
        assert data["summary"]["total"] == sum(
            data["summary"][k] for k in ["stateless", "implemented", "partial", "missing"]
        )


class TestClassInfo:
    """Tests for the ClassInfo dataclass."""

    def test_status_implemented_when_both_methods(self):
        """Status should be IMPLEMENTED when both methods present."""
        cls = ClassInfo(
            name="Test",
            file_path="test.py",
            line_number=1,
            has_get_extra_state=True,
            has_set_extra_state=True,
        )
        assert cls.status == "IMPLEMENTED"

    def test_status_partial_when_only_get(self):
        """Status should be PARTIAL when only get_extra_state present."""
        cls = ClassInfo(
            name="Test",
            file_path="test.py",
            line_number=1,
            has_get_extra_state=True,
            has_set_extra_state=False,
        )
        assert cls.status == "PARTIAL"

    def test_status_partial_when_only_set(self):
        """Status should be PARTIAL when only set_extra_state present."""
        cls = ClassInfo(
            name="Test",
            file_path="test.py",
            line_number=1,
            has_get_extra_state=False,
            has_set_extra_state=True,
        )
        assert cls.status == "PARTIAL"

    def test_status_missing_with_extra_attrs_no_methods(self):
        """Status should be MISSING when extra attrs but no methods."""
        cls = ClassInfo(
            name="Test",
            file_path="test.py",
            line_number=1,
            has_get_extra_state=False,
            has_set_extra_state=False,
            extra_attributes=["custom_attr"],
        )
        assert cls.status == "MISSING"

    def test_status_stateless_no_attrs_no_methods(self):
        """Status should be STATELESS when no extra attrs and no methods."""
        cls = ClassInfo(
            name="Test",
            file_path="test.py",
            line_number=1,
            has_get_extra_state=False,
            has_set_extra_state=False,
            extra_attributes=[],
        )
        assert cls.status == "STATELESS"
