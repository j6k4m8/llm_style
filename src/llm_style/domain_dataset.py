import enum
from typing import Literal, TypedDict
from torch.utils.data import Dataset
import datasets

class DomainType(enum.Enum):
    ANSWERS = "answers"
    BLOG = "blog"
    EMAIL = "email"
    NEWS = "news"

    @classmethod
    def as_int(cls, c: str):
        """
        Convert the domain type to an integer representation.
        Args:
            c (DomainType): The domain type to convert.
        Returns:
            int: The integer representation of the domain type.
        """
        return [
            cls.ANSWERS.value,
            cls.BLOG.value,
            cls.EMAIL.value,
            cls.NEWS.value,
        ].index(c)

    @classmethod
    def from_int(cls, c: int):
        """
        Convert the integer representation to a domain type.
        Args:
            c (int): The integer representation of the domain type.
        Returns:
            DomainType: The domain type corresponding to the integer.
        """
        return [
            cls.ANSWERS.value,
            cls.BLOG.value,
            cls.EMAIL.value,
            cls.NEWS.value,
        ][c]

class DomainDatasetDict(TypedDict):
    text: str
    label: DomainType

class DomainDataset(Dataset):
    """
    A dataset class for handling domain style tasks.
    This class extends PyTorch's Dataset class.
    """
    def __init__(self, mode: Literal["float", "int", "enum"]):
        self.dataset = datasets.load_dataset(
            "osyvokon/pavlick-formality-scores", split="train"
        )
        self.mode = mode

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
        if self.mode == "float":
            # Return float representation
            return {
                "text": str(item["sentence"]),
                "label": float(DomainType.as_int(item["domain"]) / 3),
            }
        elif self.mode == "int":
            # Return int representation
            return {
                "text": str(item["sentence"]),
                "label": DomainType.as_int(item["domain"]),
            }
        return {
            "text": str(item["sentence"]),
            "label": DomainType(item["domain"])
        }
