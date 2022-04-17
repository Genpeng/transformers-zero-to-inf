import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

train_emotion = load_dataset("emotion", split="train")
valid_emotion = load_dataset("emotion", split="validation")
label_names = train_emotion.features["label"].names
num_labels = len(label_names)

# ==================================================================================================== #
# Step 2: Feature engineering


def tokenize(tokenizer, batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_input_names = tokenizer.model_input_names

train_emotion = train_emotion.map(
    lambda batch: tokenize(tokenizer, batch), 
    batched=True, 
    batch_size=None
)
train_emotion.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])
train_features = {x: train_emotion[x] for x in model_input_names}
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_features, train_emotion["label"])
).batch(batch_size)

valid_emotion = valid_emotion.map(
    lambda batch: tokenize(tokenizer, batch), 
    batched=True, 
    batch_size=None
)
valid_emotion.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])
valid_features = {x: valid_emotion[x] for x in model_input_names}
valid_dataset = tf.data.Dataset.from_tensor_slices(
    (valid_features, valid_emotion["label"])
).batch(batch_size)

# ==================================================================================================== #
# Step 3: Build our model

model = TFAutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
    metrics=tf.keras.metrics.SparseCategoricalAccuracy(), 
)

# ==================================================================================================== #
# Step 4: Train & Evaluate 

train_history = model.fit(train_dataset, validation_data=valid_dataset, epochs=num_train_epochs)

pred_output_valid = model.predict(valid_dataset, batch_size=batch_size)
pred_valid = pred_valid.logits.argmax(axis=-1)
print(classification_report(valid_emotion['label'], pred_valid, target_names=label_names))

def plot_confusion_matrix(y_true, y_pred, labels, figsize=(6, 6)):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")
    plt.show()

plot_confusion_matrix(valid_emotion["label"], pred_valid, label_names)

