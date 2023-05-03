import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BatchEncoding

from dataset import Dataset
from tqdm import tqdm

from model import Model

DATA_PATH = 'army_intents.csv'
TRANSFORMER_MODEL_ID = 'bert-base-multilingual-cased'


def main():
    np.random.seed(112)

    df = pd.read_csv(DATA_PATH)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
    print(f"{len(df)} split into: {len(df_train)} train, {len(df_val)} validation, {len(df_test)} test")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print("Using CUDA")

    batch_size: int = 16
    epochs: int = 5
    warmup_steps: int

    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL_ID)
    train_dataloader = DataLoader(Dataset(tokenizer, df_train), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(Dataset(tokenizer, df_val), batch_size=batch_size)

    model = Model(TRANSFORMER_MODEL_ID, number_of_labels=dataset.number_of_labels)

    # for i_batch, sample_batched in enumerate(train_dataloader):
    #     print(i_batch, sample_batched['label'])
    #     if i_batch >= 3:
    #         break
    def to_device(encoded_text: BatchEncoding):
        encoded_text['attention_mask'].to(device)
        encoded_text['input_ids'].squeeze(1).to(device)

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        for batch in tqdm(train_dataloader):

            output = model(batch['text']['input_ids'].squeeze(1).to(device), batch['text']['attention_mask'].to(device))

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')


if __name__ == "__main__":
    main()
