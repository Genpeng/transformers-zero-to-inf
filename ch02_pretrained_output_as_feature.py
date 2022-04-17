import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel
from transformers import TFAutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==================================================================================================== #
# Step 1: Read in data

emotions = load_dataset("emotion")
label_names = emotions["train"].features["label"].names

# ==================================================================================================== #
# Step 2: Feature engineering

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)

# TF 版本
# 📢 注意：如果使用的模型没有 TF 版本的权重，可以加上参数 `from_pt=True`，transformers 会自动进行转换 
# tf_model = TFAutoModel.from_pretrained(model_ckpt)
# tf_model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)


def extract_last_hidden_states(batch):
	inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
	with torch.no_grad():
		outputs = model(**inputs)
		last_hidden_state = outputs.last_hidden_state 
	return {"last_hidden_state": last_hidden_state[:, 0, :].cpu().numpy()}

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_last_hidden_states, batched=True)  # the default value of `batch_size` is 1000

X_train = np.array(emotions_hidden["train"]["last_hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["last_hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])

# ==================================================================================================== #
# Optional: Visualizing the training set

from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
mapper = UMAP(n_components=2, metric="cosine").fit(X_train_scaled)

X_train_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
X_train_emb["label"] = y_train

fig, axes = plt.subplots(2, 3, figsize=(7,5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
for i, (label_name, cmap) in enumerate(zip(label_names, cmaps)):
    X_train_emb_sub = X_train_emb.loc[X_train_emb["label"] == i]
    axes[i].hexbin(X_train_emb_sub["X"], X_train_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
    axes[i].set_title(label_name)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.tight_layout()
plt.show()

# ==================================================================================================== #
# Step 3: Build our model

lr_clf = LogisticRegression(max_iter=3000)

# ==================================================================================================== #
# Step 4: Train & Evaluate 

lr_clf.fit(X_train, y_train)

pred_train = lr_clf.predict(X_train)
print(classification_report(y_train, pred_train, target_names=label_names))

pred_valid = lr_clf.predict(X_valid)
print(classification_report(y_valid, pred_valid, target_names=label_names))


def plot_confusion_matrix(y_true, y_pred, labels, figsize=(6, 6)):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")
    plt.show()

plot_confusion_matrix(y_valid, pred_valid, label_names)

