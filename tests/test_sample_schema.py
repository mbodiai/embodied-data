from typing import Any
import numpy as np
import pytest
from embdata.sample import Sample
from pydantic import Field
from embdata.ndarray import ndarray


class NewSample(Sample):
    answer: str = Field(
        default="",
        description="Short, one sentence answer to any question a user might have asked. 20 words max.",
    )
    sleep: bool = Field(
        default=False,
        description="Whether the robot should go to sleep after executing the motion.",
    )
    home: bool = Field(
        default=False,
        description="Whether the robot should go to home after executing the motion.",
    )


def test_sample_schema():
    assert NewSample().schema(include="descriptions") == {
        "title": "NewSample",
        "type": "object",
        "properties": {
            "answer": {
                "title": "Answer",
                "type": "string",
                "default": "",
                "description": "Short, one sentence answer to any question a user might have asked. 20 words max.",
            },
            "home": {
                "title": "Home",
                "type": "boolean",
                "default": False,
                "description": "Whether the robot should go to home after executing the motion.",
            },
            "sleep": {
                "title": "Sleep",
                "type": "boolean",
                "default": False,
                "description": "Whether the robot should go to sleep after executing the motion.",
            },
        },
    }


def test_sample_schema_with_numpy_array():
    class NewSample(Sample):
        answer: str = Field(
            default="",
            description="Short, one sentence answer to any question a user might have asked. 20 words max.",
        )
        sleep: bool = Field(
            default=False,
            description="Whether the robot should go to sleep after executing the motion.",
        )
        home: bool = Field(
            default=False,
            description="Whether the robot should go to home after executing the motion.",
        )
        image: ndarray = Field(
            default_factory=lambda: np.zeros((224, 224, 3)),
            description="Image data",
        )

    assert NewSample().schema(include="simple") == {
        "type": "object",
        "title": "NewSample",
        "properties": {
            "answer": {
                "default": "",
                "title": "Answer",
                "type": "string",
            },
            "home": {
                "default": False,
                "title": "Home",
                "type": "boolean",
            },
            "image": {
                "items": {
                    "type": "number",
                },
                "shape": (224, 224, 3),
                "title": "ndarray",
                "type": "array",
            },
            "sleep": {
                "default": False,
                "title": "Sleep",
                "type": "boolean",
            },
        },
    }


def test_sample_schema_with_null():
    class NewSample(Sample):
        answer: str = Field(
            default="",
            description="Short, one sentence answer to any question a user might have asked. 20 words max.",
        )
        sleep: bool | None = Field(
            description="Whether the robot should go to sleep after executing the motion.",
        )
        home: bool = Field(
            default=False,
            description="Whether the robot should go to home after executing the motion.",
        )
        image: None | ndarray = Field(
            default_factory=lambda: np.zeros((224, 224, 3)),
            description="Image data",
        )

    assert NewSample(sleep=True).schema(include="simple") == {
        "properties": {
            "answer": {
                "default": "",
                "title": "Answer",
                "type": "string",
            },
            "home": {
                "default": False,
                "title": "Home",
                "type": "boolean",
            },
            "image": {
                "items": {
                    "type": "number",
                },
                "title": "ndarray",
                "type": "array",
                "shape": (224, 224, 3),
            },
            "sleep": {
                "title": "Sleep",
                "type": "boolean",
            },
        },
        "required": [
            "sleep",
        ],
        "title": "NewSample",
        "type": "object",
    }


def test_dynamic_field():
    assert Sample(list_field=["a", "b"]).schema(include="simple") == {
        "properties": {
            "list_field": {
                "items": {
                    "type": "string",
                },
                "maxItems": 2,
                "title": "List Field",
                "type": "array",
            },
        },
        "title": "Sample",
        "type": "object",
    }


def test_nested():
    class AnotherSample(Sample):
        child_field: Sample = Field(
            default_factory=lambda: Sample(list_field=["a", "b"]),
            description="Child field",
        )

    class NewSample(Sample):
        answer: str = Field(
            default="",
            description="Short, one sentence answer to any question a user might have asked. 20 words max.",
        )
        nested: AnotherSample = Field(
            default_factory=lambda: AnotherSample(child_field=Sample(list_field=["a", "b"])),
            description="Nested sample",
        )

    assert NewSample().schema(include="simple") == {
        "properties": {
            "answer": {
                "default": "",
                "title": "Answer",
                "type": "string",
            },
            "nested": {
                "properties": {
                    "child_field": {
                        "properties": {
                            "list_field": {
                                "items": {
                                    "type": "string",
                                },
                                "maxItems": 2,
                                "title": "List Field",
                                "type": "array",
                            },
                        },
                        "title": "Sample",
                        "type": "object",
                    },
                },
                "title": "AnotherSample",
                "type": "object",
            },
        },
        "title": "NewSample",
        "type": "object",
    }


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
