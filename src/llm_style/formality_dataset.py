import enum
from typing import Literal, Optional, TypedDict
from torch.utils.data import Dataset

import datasets


class FormalityLevel(enum.Enum):
    INFORMAL = "informal"
    FORMAL = "formal"


class FormalityDatasetDict(TypedDict):
    text: str
    label: FormalityLevel | float


class ContrastiveFormalityDatasetDict(TypedDict):
    formal: str
    informal: str


class FormalityDataset(Dataset):
    """
    A dataset class for handling formality style tasks.
    This class extends PyTorch's Dataset class.
    """

    data: list[FormalityDatasetDict]

    def __init__(
        self,
        data: Optional[list[FormalityDatasetDict]] = None,
        return_float_labels: bool = True,
    ):
        """
        Initialize the dataset with data.

        Args:
            data (list): A list of data samples, where each sample is a dictionary
                         containing input and label fields.
        """
        self.data = data
        self.return_float_labels = return_float_labels

    @classmethod
    def from_contrastive_samples(
        cls, data: list[ContrastiveFormalityDatasetDict], **kwargs
    ):
        """
        Initialize the dataset with contrastive samples.

        Args:
            data (list): A list of contrastive data samples, where each sample is a dictionary
                         containing formal and informal text fields.
        """
        self = cls(**kwargs)
        self.data = []
        for item in data:
            self.data.append({"text": item["formal"], "label": FormalityLevel.FORMAL})
            self.data.append(
                {"text": item["informal"], "label": FormalityLevel.INFORMAL}
            )
        self._validate_data()
        return self

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
        if not self.return_float_labels:
            return self.data[idx]
        return {
            "text": self.data[idx]["text"],
            "label": float(self.data[idx]["label"] == FormalityLevel.FORMAL),
        }


class PavlickFormalityDataset(FormalityDataset):
    """
    A dataset class for handling Pavlick formality data.
    This class extends the FormalityDataset class.

    # https://huggingface.co/datasets/osyvokon/pavlick-formality-scores
    """

    def __init__(self, mode: Literal["float", "int", "enum"]):
        self.dataset = datasets.load_dataset(
            "osyvokon/pavlick-formality-scores", split="train"
        )
        self.mode = mode

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx) -> FormalityDatasetDict:
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the input and label for the sample.
        """
        item = self.dataset[idx]
        if self.mode == "float":
            # Return float representation
            return {
                "text": str(item["sentence"]),
                "label": float(item["avg_score"]),
            }
        elif self.mode == "enum":
            # Return string label
            return {
                "text": str(item["sentence"]),
                "label": FormalityLevel.FORMAL
                if item["avg_score"] > 0
                else FormalityLevel.INFORMAL,
            }
        elif self.mode == "int":
            """
            Return the label as a string ('formal' or 'informal') based on avg_score threshold.
            """
            return {
                "text": str(item["sentence"]),
                # Convert avg_score to 0 or 1 based on threshold (0.5)
                "label": int(item["avg_score"] > 0),  # 1 for formal, 0 for informal
            }
        elif self.mode == "filtered":
            # Only keep samples with avg_score >= 1 or <= -1
            if abs(item["avg_score"]) >= .75:
                return {
                    "text": str(item["sentence"]),
                    "label": int(item["avg_score"] > 0),  # 1 for formal, 0 for informal
                }
class ContrastiveFormalityDataset(FormalityDataset):
    """
    A dataset class for handling contrastive formality style tasks.
    This class extends the FormalityDataset class.
    """

    def __init__(
        self, data: list[ContrastiveFormalityDatasetDict], check_all_keys: bool = True
    ):
        """
        Initialize the dataset with data.

        Args:
            data (list): A list of data samples, where each sample is a dictionary
                         containing input and label fields.
        """
        # Unfold the data into formal and informal samples
        self.data = []
        for item in data:
            self.data.append({"text": item["formal"], "label": FormalityLevel.FORMAL})
            self.data.append(
                {"text": item["informal"], "label": FormalityLevel.INFORMAL}
            )

        if check_all_keys:
            self._validate_data()
