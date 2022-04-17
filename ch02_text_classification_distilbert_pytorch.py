import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global variables
model_ckpt = "distilbert-base-uncased"
# data_dir = "./data"
output_dir = f"{model_ckpt}-finetuned-emotion"

# hyper-parameters
num_train_epochs = 2
learning_rate = 2e-5
batch_size = 64

# ==================================================================================================== #
# Step 1: Read in data

emotions = load_dataset("emotion")
label_names = emotions["train"].features["label"].names
num_labels = len(label_names)

# ==================================================================================================== #
# Step 2: Feature engineering


def tokenize(tokenizer, batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
emotions_encoded = emotions.map(lambda batch: tokenize(tokenizer, batch), batched=True, batch_size=None)

# ==================================================================================================== #
# Step 3: Build our model

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)

# ==================================================================================================== #
# Step 4: Train & Evaluate 


def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(-1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {"accuracy": acc, "f1": f1}

logging_steps = len(emotions_encoded["train"]) // batch_size
training_args = TrainingArguments(
    output_dir=output_dir, 
    num_train_epochs=num_train_epochs, 
    learning_rate=learning_rate, 
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size, 
    evaluation_strategy="epoch", 
    disable_tqdm=False, 
    logging_steps=logging_steps, 
    push_to_hub=False, 
    log_level="error"
)

trainer = Trainer(
    tokenizer=tokenizer, 
    model=model, 
    args=training_args, 
    train_dataset=emotions_encoded["train"], 
    eval_dataset=emotions_encoded["validation"], 
    compute_metrics=compute_metrics, 
)

train_output = trainer.train()


pred_output_valid = trainer.predict(emotions_encoded["validation"])
pred_valid = pred_output_valid.predictions.argmax(-1)
y_valid = emotions_encoded["validation"]["label"]
print(classification_report(y_valid, pred_valid, target_names=label_names))


def plot_confusion_matrix(y_true, y_pred, labels, figsize=(6, 6)):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")
    plt.show()

plot_confusion_matrix(y_valid, pred_valid, label_names)

