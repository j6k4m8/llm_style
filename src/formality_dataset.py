import enum
from typing import TypedDict
from torch.utils.data import Dataset


class FormalityLevel(enum.Enum):
    INFORMAL = "informal"
    FORMAL = "formal"


class FormalityDatasetDict(TypedDict):
    text: str
    label: FormalityLevel


class FormalityDataset(Dataset):
    """
    A dataset class for handling formality style tasks.
    This class extends PyTorch's Dataset class.
    """

    def __init__(self, data: list[FormalityDatasetDict], check_all_keys: bool = True):
        """
        Initialize the dataset with data.

        Args:
            data (list): A list of data samples, where each sample is a dictionary
                         containing input and label fields.
        """
        self.data = data
        if check_all_keys:
            self._validate_data()

    def _validate_data(self):
        assert all("text" in item and "label" in item for item in self.data), (
            "Each item must contain 'text' and 'label' keys."
        )

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx) -> FormalityDatasetDict:
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the input and label for the sample.
        """
        return self.data[idx]
