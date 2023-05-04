import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BatchEncoding
from dataset import Dataset, Labeler
from tqdm import tqdm

from model import Model
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# CUDA issues
# a = torch.ones(1, device="cuda") or so.
# And then check the GPU memory usage (it’s nvidia-smi on Linux

DATA_PATH = 'army_intents.csv'
TRANSFORMER_MODEL_ID = 'bert-base-multilingual-cased'


def print_first_i_batches(data_loader: DataLoader, i: int = 1) -> None:
    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, sample_batched['label'])
        if i_batch >= i:
            break
    print()


def main():
    # TODO: move all params/hyperparams to some param dataclass ?
    # TODO: using correct transformers BertModel ? should be prob. BertModelForClassification
    np.random.seed(112)

    df = pd.read_csv(DATA_PATH)
    labeler = Labeler(df)

    train_size, val_size = 0.8, 0.1
    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=42), [int(train_size * len(df)), int((train_size + val_size) * len(df))])
    print(f"{len(df)} split into: {len(df_train)} train, {len(df_val)} validation, {len(df_test)} test")

    # TODO: check how balanced are the datasets after the split !!!!!!!

    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # if use_cuda:
    #     print("Using CUDA")
    device = torch.device('cpu')

    batch_size: int = 8
    epochs: int = 5
    warmup_steps: int
    learning_rate: float = 1e-4
    label_smoothing: float = 0.0

    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL_ID)

    train_dataloader = DataLoader(Dataset(tokenizer, df_train, labeler),
                                  batch_size=batch_size,
                                  shuffle=True)  # not using sampler, with shuffle==True
    val_dataloader = DataLoader(Dataset(tokenizer, df_val, labeler), batch_size=batch_size)
    test_dataloader = DataLoader(Dataset(tokenizer, df_test, labeler), batch_size=batch_size)
    print_first_i_batches(train_dataloader)
    print_first_i_batches(val_dataloader)
    print_first_i_batches(test_dataloader)

    model = Model(TRANSFORMER_MODEL_ID, device=device, number_of_labels=labeler.number_of_labels, )

    # CE loss between input logits and target.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing,  # todo try smoothing
                                    weight=None)  # TODO: you have unbalanced dataset

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_steps = len(train_dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    for epoch in range(epochs):
        total_acc_train, total_loss_train = 0, 0
        for batch in tqdm(train_dataloader):
            output = model(batch['tokenized_text']['input_ids'].squeeze(1).to(device),
                           batch['tokenized_text']['attention_mask'].to(device))

            batch_loss = criterion(output, batch['label'].to(device).long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == batch['label'].to(device)).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        total_acc_val, total_loss_val = 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                output = model(batch['tokenized_text']['input_ids'].squeeze(1).to(device),
                               batch['tokenized_text']['attention_mask'].to(device))

                val_label = batch['label'].to(device)
                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(f'Epochs: {epoch} \
                | Train Loss: {total_loss_train / len(df_train): .3f} \
                | Train Accuracy: {total_acc_train / len(df_train): .3f} \
                | Val Loss: {total_loss_val / len(df_val): .3f} \
                | Val Accuracy: {total_acc_val / len(df_val): .3f}')

    evaluate(model, device, test_dataloader)


def evaluate(model: Model, device: torch.device, test_dataloader: DataLoader):
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_dataloader): .3f}')


if __name__ == "__main__":
    main()

    # https://www.kaggle.com/code/akshat0007/bert-for-sequence-classification
    # https://www.tensorflow.org/text/tutorials/classify_text_with_bert
