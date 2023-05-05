import torch
from torch import nn
import pandas as pd
from model import Model
from dataset import Labeler
from transformers import BertTokenizer

DATA_PATH = 'army_intents.csv'
TRANSFORMER_MODEL_ID = 'bert-base-multilingual-cased'
state = torch.load('./model_files/finetuned_bert-base-multilingual-cased.pt')
df = pd.read_csv(DATA_PATH)
labeler = Labeler(df)
device = torch.device("cuda")

model = Model(TRANSFORMER_MODEL_ID, device=device, number_of_labels=labeler.number_of_labels)
model.load_state_dict(state)

tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL_ID)

# text_1 = "jaké jsou výhody práce v armádě"
# text_2 = "jake jsou vyhody prace v armade"
text_1 = "bojovat ukrajinu očkov"

input_1 = tokenizer(text_1, padding='max_length', max_length=15, truncation=True, return_tensors="pt")
input_2 = tokenizer(text_2, padding='max_length', max_length=15, truncation=True, return_tensors="pt")
print(input_1)

with torch.no_grad():
    model_output = model(input_1['input_ids'].to(device),
                         input_1['attention_mask'].to(device))

    best_category_idx = model_output.argmax(dim=1)

    best_category = list(labeler.label_to_index.keys())[best_category_idx]
    for k, v in labeler.index_to_label.items():
        print(k, v)

    softmax = torch.nn.Softmax(dim=1)

    print(f"predictions: {softmax(model_output)}")

    print(
        f"confidence: {torch.topk(input=softmax(model_output), k=5, dim=1)}, best class idx={best_category_idx}, tj. {best_category}")