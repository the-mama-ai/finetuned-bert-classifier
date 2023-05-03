import pandas as pd
import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from dataset import Dataset


class Model(nn.Module):
    def __init__(self, transformer_model_id: str, number_of_labels: int, dropout_rate: float = 0.1) -> None:
        super(Model, self).__init__()
        self.bert: nn.Module = BertModel.from_pretrained(transformer_model_id)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(768, number_of_labels)
        self.softmax = nn.LogSoftmax(dim=1)  # dimension along which softmax will be computed

    def forward(self, input_ids, mask) -> Tensor:
        contextualized_token_embeddings, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=mask, return_dict=False)

        return self.softmax(self.linear(self.dropout(pooled_output)))



