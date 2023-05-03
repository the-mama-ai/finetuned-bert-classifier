from pandas.core.reshape.encoding import DataFrame
import torch
from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class Dataset(torch.utils.data.Dataset):
    # TODO: you have to pass the labels from the WHOLE dataset !
    def __init__(self, tokenizer: BertTokenizer, df: DataFrame):
        self.number_of_labels = df['label'].unique()

        self.label_to_index: dict[str, int] = {label: i for i, label in enumerate(self.number_of_labels)}

        self.labels: list[int] = [self.label_to_index[label] for label in df['label']]

        self.texts: list[BatchEncoding] = [
            tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            for text in df['text']]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> dict[str, BatchEncoding | int]:
        """ Returns 1 sample. If batch_size=n, DataLoader makes 'n' __getitem__ calls.
        https://stackoverflow.com/questions/66370250/how-does-pytorch-dataloader-interact-with-a-pytorch-dataset-to-transform-batches
        """
        return {'tokenized_text': self.texts[idx], 'label': self.labels[idx]}