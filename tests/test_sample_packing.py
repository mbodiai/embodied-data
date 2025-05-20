import pytest
import numpy as np
from embdata.sample import Sample


@pytest.fixture
def sample_instances():
    return [Sample(attr1=1, attr2="a"), Sample(attr1=2, attr2="b"), Sample(attr1=3, attr2="c")]


@pytest.fixture
def sample_dicts():
    return [{"attr1": 4, "attr2": "d"}, {"attr1": 5, "attr2": "e"}, {"attr1": 6, "attr2": "f"}]


def test_unpack_from_with_samples(sample_instances):
    unpacked_sample = Sample.unpack_from(sample_instances)
    assert np.array_equal(unpacked_sample.attr1, [1, 2, 3])
    assert np.array_equal(unpacked_sample.attr2, ["a", "b", "c"])


def test_unpack_from_with_dicts(sample_dicts):
    unpacked = Sample.unpack_from(sample_dicts)
    assert np.array_equal(unpacked.attr1, [4, 5, 6])
    assert np.array_equal(unpacked.attr2, ["d", "e", "f"])


def test_unpack_from_with_mixed(sample_instances, sample_dicts):
    mixed_inputs = sample_instances + sample_dicts
    unpacked = Sample.unpack_from(mixed_inputs)
    assert np.array_equal(unpacked.attr1, [1, 2, 3, 4, 5, 6])
    assert np.array_equal(unpacked.attr2, ["a", "b", "c", "d", "e", "f"])


def test_unpack_from_empty():
    with pytest.raises(ValueError):
        Sample.unpack_from([])


def test_unpack_from_invalid_input():
    with pytest.raises(ValueError):
        Sample.unpack_from([None])


def test_pack_to_samples(sample_instances):
    unpacked = Sample.unpack_from(sample_instances)
    packed_samples = unpacked.pack()
    for original, packed in zip(sample_instances, packed_samples):
        assert original.attr1 == packed.attr1
        assert original.attr2 == packed.attr2


def test_pack_to_dicts(sample_instances):
    unpacked = Sample.unpack_from(sample_instances)
    packed_samples = unpacked.pack("dicts")
    for original, unpacked in zip(sample_instances, packed_samples):
        assert original.attr1 == unpacked["attr1"]
        assert original.attr2 == unpacked["attr2"]


def test_unpack_from_with_padding_truncate(sample_instances):
    # Modify one instance to test truncation
    sample_instances[-1].attr3 = True
    unpacked = Sample.unpack_from(sample_instances, padding="truncate")
    assert not hasattr(unpacked, "attr3")


def test_unpack(sample_instances):
    # Modify one instance to test truncation
    unpacked_sample = Sample.unpack_from(sample_instances)
    unpacked_sample.attr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    packed_sample = unpacked_sample.pack()
    assert packed_sample[0].attr1 == 1
    assert packed_sample[1].attr1 == 2
    assert packed_sample[2].attr1 == 3

    assert packed_sample[0].attr2 == "a"
    assert packed_sample[1].attr2 == "b"
    assert packed_sample[2].attr2 == "c"


def test_unpack_empty():
    empty_sample = Sample()  # Assuming this creates an empty Sample instance
    unpacked_samples = empty_sample.unpack()
    assert unpacked_samples == []
