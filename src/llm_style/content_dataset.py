import enum
from typing import Literal, Optional, TypedDict
from torch.utils.data import Dataset

import datasets


class ContentType(enum.Enum):
    NEWS = "news"
    QUESTION = "question"


class ContentDatasetDict(TypedDict):
    text: str
    label: ContentType | float


class ContentDataset(Dataset):
    """
    A dataset class for handling content style tasks.
    This class extends PyTorch's Dataset class.
    """

    data: list[ContentDatasetDict]

    def __init__(
        self,
        data: Optional[list[ContentDatasetDict]] = None,
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

    def _validate_data(self):
        assert all("text" in item and "label" in item for item in self.data), (
            "Each item must contain 'text' and 'label' keys."
        )

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx) -> ContentDatasetDict:
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
            "label": float(self.data[idx]["label"] == ContentType.NEWS),
        }


class CNNContentDataset(ContentDataset):
    """
    A dataset class for handling CNN/DailyMail content data.
    This class extends the ContentDataset class.

    # https://huggingface.co/datasets/abisee/cnn_dailymail
    """

    def __init__(self):
        self.dataset = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx) -> ContentDatasetDict:
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the input and label for the sample.
        """
        item = self.dataset[idx]
        return {
            "article": str(item["article"]),
            "summary": str(item["highlights"]),
        }


class NaturalQuestionsDataset(ContentDataset):
    """
    A dataset class for handling Natural Questions content data.
    This class extends the ContentDataset class.

    # https://huggingface.co/datasets/google-research-datasets/natural_questions
    """

    def __init__(self, subset_size: Optional[int] = None):
        # Load the dataset from Hugging Face with streaming enabled to handle large datasets
        self.dataset = datasets.load_dataset("google-research-datasets/natural_questions", split="train", streaming=True)

        # If subset_size is provided, take a subset of the dataset
        if subset_size is not None:
            self.dataset = list(self.dataset.take(subset_size))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> ContentDatasetDict:
        item = self.dataset[idx]
        return {
            "question": str(item["question_text"]),
            "short_answer": str(item["short_answers"] if "short_answers" in item else ""),
        }