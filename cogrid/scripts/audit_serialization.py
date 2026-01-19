#!/usr/bin/env python3
"""Audit script for GridObj serialization status.

This script identifies all GridObj subclasses in the cogrid codebase and
categorizes them by their serialization implementation status:
- STATELESS: No extra attributes that need serialization
- IMPLEMENTED: Has both get_extra_state and set_extra_state methods
- PARTIAL: Has only one of the two serialization methods
- MISSING: Has extra attributes but no serialization methods

Usage:
    python -m cogrid.scripts.audit_serialization
    python -m cogrid.scripts.audit_serialization --json
    python -m cogrid.scripts.audit_serialization --path /path/to/cogrid
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# Standard GridObj attributes that don't need extra serialization
# These are either handled by encode() or are set in the base GridObj.__init__
STANDARD_ATTRIBUTES = {
    # Base GridObj attributes set in __init__
    "uuid",
    "state",
    "obj_placed_on",
    "init_pos",
    "pos",
    "toggle_value",
    "inventory_value",
    "overlap_value",
    "placed_on_value",
    "picked_up_from_value",
    # Class-level attributes
    "object_id",
    "color",
    "char",
}


@dataclass
class ClassInfo:
    """Information about a GridObj subclass."""

    name: str
    file_path: str
    line_number: int
    has_get_extra_state: bool = False
    has_set_extra_state: bool = False
    extra_attributes: list[str] = field(default_factory=list)
    base_classes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Determine serialization status based on methods and attributes."""
        has_both = self.has_get_extra_state and self.has_set_extra_state
        has_one = self.has_get_extra_state or self.has_set_extra_state
        has_extra_attrs = len(self.extra_attributes) > 0

        if has_both:
            return "IMPLEMENTED"
        elif has_one:
            return "PARTIAL"
        elif has_extra_attrs:
            return "MISSING"
        else:
            return "STATELESS"


class GridObjClassFinder(ast.NodeVisitor):
    """AST visitor that finds GridObj subclasses and analyzes their serialization status."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.classes: list[ClassInfo] = []
        self._current_class: ClassInfo | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition and check if it's a GridObj subclass."""
        # Check if this class inherits from GridObj or a known GridObj subclass
        base_names = self._get_base_names(node)

        if self._is_gridobj_subclass(base_names):
            class_info = ClassInfo(
                name=node.name,
                file_path=self.file_path,
                line_number=node.lineno,
                base_classes=base_names,
            )

            # Analyze class body for methods and attributes
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "get_extra_state":
                        class_info.has_get_extra_state = True
                    elif item.name == "set_extra_state":
                        class_info.has_set_extra_state = True
                    elif item.name == "__init__":
                        # Find extra attributes in __init__
                        class_info.extra_attributes = self._find_extra_attributes(item)

            self.classes.append(class_info)

        # Continue visiting child nodes
        self.generic_visit(node)

    def _get_base_names(self, node: ast.ClassDef) -> list[str]:
        """Extract base class names from a class definition."""
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                # Handle cases like grid_object.GridObj
                base_names.append(f"{self._get_attr_string(base)}")
        return base_names

    def _get_attr_string(self, node: ast.Attribute) -> str:
        """Convert an Attribute node to a string like 'module.attr'."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _is_gridobj_subclass(self, base_names: list[str]) -> bool:
        """Check if any base class indicates GridObj inheritance."""
        gridobj_patterns = {
            "GridObj",
            "grid_object.GridObj",
        }
        return any(base in gridobj_patterns for base in base_names)

    def _find_extra_attributes(self, init_node: ast.FunctionDef) -> list[str]:
        """Find self.X = ... assignments in __init__ that are not standard attributes."""
        extra_attrs = []

        for node in ast.walk(init_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            attr_name = target.attr
                            if attr_name not in STANDARD_ATTRIBUTES:
                                extra_attrs.append(attr_name)

        return sorted(set(extra_attrs))


def find_python_files(cogrid_path: Path) -> Iterator[Path]:
    """Recursively find all Python files in the cogrid directory."""
    for py_file in cogrid_path.rglob("*.py"):
        # Skip __pycache__ directories
        if "__pycache__" in str(py_file):
            continue
        yield py_file


def find_gridobj_subclasses(cogrid_path: Path) -> list[ClassInfo]:
    """Find all GridObj subclasses in the cogrid codebase."""
    all_classes: list[ClassInfo] = []

    for py_file in find_python_files(cogrid_path):
        try:
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))
            finder = GridObjClassFinder(str(py_file))
            finder.visit(tree)
            all_classes.extend(finder.classes)
        except SyntaxError as e:
            print(f"Warning: Could not parse {py_file}: {e}")

    return all_classes


def categorize_classes(classes: list[ClassInfo]) -> dict[str, list[ClassInfo]]:
    """Categorize classes by their serialization status."""
    categories: dict[str, list[ClassInfo]] = {
        "STATELESS": [],
        "IMPLEMENTED": [],
        "PARTIAL": [],
        "MISSING": [],
    }

    for cls in classes:
        categories[cls.status].append(cls)

    # Sort each category by class name
    for category in categories.values():
        category.sort(key=lambda c: c.name)

    return categories


def format_text_output(categories: dict[str, list[ClassInfo]]) -> str:
    """Format the audit results as human-readable text."""
    lines = ["=== GridObj Serialization Audit ===", ""]

    for status in ["STATELESS", "IMPLEMENTED", "PARTIAL", "MISSING"]:
        classes = categories[status]
        lines.append(f"{status} ({len(classes)}):")

        if not classes:
            lines.append("  (none)")
        else:
            for cls in classes:
                rel_path = cls.file_path
                # Try to make path relative to cogrid
                if "cogrid" in rel_path:
                    rel_path = rel_path[rel_path.find("cogrid") :]
                lines.append(f"  - {cls.name} ({rel_path}:{cls.line_number})")

                if cls.extra_attributes:
                    lines.append(f"    Extra attributes: {', '.join(cls.extra_attributes)}")

        lines.append("")

    # Summary
    total = sum(len(c) for c in categories.values())
    lines.append("Summary:")
    lines.append(f"  Total classes: {total}")
    lines.append(f"  Stateless: {len(categories['STATELESS'])}")
    lines.append(f"  Implemented: {len(categories['IMPLEMENTED'])}")
    lines.append(f"  Partial: {len(categories['PARTIAL'])}")
    lines.append(f"  Missing: {len(categories['MISSING'])}")

    return "\n".join(lines)


def format_json_output(categories: dict[str, list[ClassInfo]]) -> str:
    """Format the audit results as JSON."""
    output = {
        "stateless": [],
        "implemented": [],
        "partial": [],
        "missing": [],
        "summary": {
            "total": 0,
            "stateless": 0,
            "implemented": 0,
            "partial": 0,
            "missing": 0,
        },
    }

    for status, classes in categories.items():
        key = status.lower()
        output[key] = [
            {
                "name": cls.name,
                "file_path": cls.file_path,
                "line_number": cls.line_number,
                "extra_attributes": cls.extra_attributes,
                "base_classes": cls.base_classes,
            }
            for cls in classes
        ]
        output["summary"][key] = len(classes)

    output["summary"]["total"] = sum(output["summary"][k] for k in ["stateless", "implemented", "partial", "missing"])

    return json.dumps(output, indent=2)


def get_cogrid_path(path_arg: str | None = None) -> Path:
    """Determine the cogrid directory path."""
    if path_arg:
        return Path(path_arg)

    # Auto-detect from script location
    script_path = Path(__file__).resolve()
    # Script is at cogrid/scripts/audit_serialization.py
    # cogrid directory is two levels up
    cogrid_path = script_path.parent.parent
    return cogrid_path


def main():
    """Main entry point for the audit script."""
    parser = argparse.ArgumentParser(
        description="Audit GridObj serialization implementation status"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to cogrid directory (default: auto-detect from script location)",
    )

    args = parser.parse_args()

    cogrid_path = get_cogrid_path(args.path)

    if not cogrid_path.exists():
        print(f"Error: cogrid directory not found at {cogrid_path}")
        return 1

    classes = find_gridobj_subclasses(cogrid_path)
    categories = categorize_classes(classes)

    if args.json:
        print(format_json_output(categories))
    else:
        print(format_text_output(categories))

    return 0


if __name__ == "__main__":
    exit(main())
