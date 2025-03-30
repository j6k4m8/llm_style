import pytest
from .formality_dataset import FormalityDataset, FormalityLevel


@pytest.fixture
def sample_data():
    return [
        {"text": "Good evening, how do you do?", "label": FormalityLevel.FORMAL},
        {"text": "Hey, what's up?", "label": FormalityLevel.INFORMAL},
    ]


@pytest.fixture
def broken_data_invalid_key():
    return [{"text": "Hello", "invalid_key": "value"}]


def test_formality_dataset_schema(sample_data):
    dataset = FormalityDataset(sample_data)

    for item in dataset:
        assert isinstance(item, dict), "Each item should be a dictionary."
        assert "text" in item and "label" in item, (
            "Each item should have 'text' and 'label' keys."
        )
        assert isinstance(item["text"], str), "The 'text' field should be a string."
        assert item["label"] in (FormalityLevel.FORMAL, FormalityLevel.INFORMAL), (
            "The 'label' field should be a valid FormalityLevel."
        )


def test_formality_dataset_length(sample_data):
    dataset = FormalityDataset(sample_data)
    assert len(dataset) == len(sample_data), (
        "Dataset length should match the input data length."
    )


def test_invalid_keys_throw_error(broken_data_invalid_key):
    with pytest.raises(AssertionError):
        FormalityDataset(broken_data_invalid_key)  # Invalid key


def test_no_invalid_key_errors_without_check(broken_data_invalid_key):
    dataset = FormalityDataset(broken_data_invalid_key, check_all_keys=False)
    assert len(dataset) == len(broken_data_invalid_key), (
        "Dataset length should match the input data length."
    )  # No error should be raised
