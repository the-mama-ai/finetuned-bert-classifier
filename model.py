import torch
from torch import nn, Tensor
from transformers import BertModel, BertForSequenceClassification


class Model(nn.Module):

    def __init__(self, transformer_model_id: str, device: torch.device,
                 number_of_labels: int, dropout_rate: float = 0.1) -> None:
        print(f"Some weights of the model checkpoint at {transformer_model_id} ... "
              f"warning rendered because the classification head not trained in pre-trained model.")
        super(Model, self).__init__()
        self.bert: nn.Module = BertModel.from_pretrained(transformer_model_id)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(2048, number_of_labels)
        # self.batch_norm = nn.BatchNorm1d
        # self.softmax = nn.LogSoftmax(dim=1)  # dimension along which softmax will be computed
        # using nn.CrossEntropyLoss. This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

        self.to(device)

    def forward(self, input_ids, mask) -> Tensor:
        """ Outputs raw un-normalized logits. """
        contextualized_token_embeddings, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=mask, return_dict=False)
        # pooled output is the [CLS] token we use here for the classification of size [batch_size, H]

        return self.linear(self.dropout(pooled_output))
