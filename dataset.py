import pandas as pd
from pandas.core.reshape.encoding import DataFrame
import torch
from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class Labeler:

    def __init__(self, df: pd.DataFrame):
        self.labels, self.label_to_index, self.index_to_label = self.process_labels(df)
        self.number_of_labels = len(self.labels)

    @staticmethod
    def process_labels(df: pd.DataFrame) -> tuple[list[str], dict[str, int], dict[int, str]]:
        labels = df['label'].unique()
        label_to_index = {}
        index_to_label = {}

        for i, label in enumerate(labels):
            label_to_index[label] = i
            index_to_label[i] = label

        return labels, label_to_index, index_to_label


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: BertTokenizer, df: DataFrame, labeler: Labeler):
        self.labels: list[int] = [labeler.label_to_index[label] for label in df['label']]

        # TODO: max_length, truncation, return_tensors?
        self.texts: list[BatchEncoding] = [
            tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            for text in df['text']]
        # tokenizer prepends [CLS]=101 and appends [SEP]=102 for each tokenized text

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> dict[str, BatchEncoding | int]:
        """ Returns 1 sample. If batch_size=n, DataLoader makes 'n' __getitem__ calls.
        https://stackoverflow.com/questions/66370250/how-does-pytorch-dataloader-interact-with-a-pytorch-dataset-to-transform-batches
        """
        return {'tokenized_text': self.texts[idx], 'label': self.labels[idx]}
