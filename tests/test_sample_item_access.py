import pytest
from embdata.sample import Sample
from typing import List


class SampleSubclass(Sample):
    items: List[int] = [1, 2, 3]


@pytest.fixture
def sample_instance():
    sample = Sample(items=[1, 2, 3])
    return sample


@pytest.fixture
def subclass_instance():
    return SampleSubclass()


# def test_getitem_with_int_key_list(sample_instance):
#     assert sample_instance[0] == 1


def test_getitem_with_int_key_no_list(sample_instance):
    del sample_instance.items
    with pytest.raises(TypeError):
        _ = sample_instance[0]


def test_getitem_with_str_key(sample_instance):
    sample_instance._extra.test_attr = "value"
    assert sample_instance["test_attr"] == "value"


# def test_getitem_subclass_with_int_key(subclass_instance):
#     _ = subclass_instance[0]


def test_setitem_subclass_with_str_key(subclass_instance):
    subclass_instance["new_attr"] = "new_value"


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
