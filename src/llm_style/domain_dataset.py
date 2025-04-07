import enum
from typing import Literal, TypedDict
from torch.utils.data import Dataset
import datasets

class DomainType(enum.Enum):
    ANSWERS = "answers"
    BLOG = "blog"
    EMAIL = "email"
    NEWS = "news"

class DomainDatasetDict(TypedDict):
    text: str
    label: DomainType

class DomainDataset(Dataset):
    """
    A dataset class for handling domain style tasks.
    This class extends PyTorch's Dataset class.
    """
    def __init__(self):
        self.dataset = datasets.load_dataset(
            "osyvokon/pavlick-formality-scores", split="train"
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> DomainDatasetDict:
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the input and label for the sample.
        """
        item = self.dataset[idx]
        return {
            "text": str(item["sentence"]),
            "label": DomainType(item["domain"]),
        }
