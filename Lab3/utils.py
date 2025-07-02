import torch
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
)


def extract_cls_features(dataset, tokenizer, model, batch_size=16, device="cuda"):
    model.to(device)
    model.eval()
    all_features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting features"):
            batch = dataset[i : i + batch_size]
            inputs = tokenizer(
                batch["text"], padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden_states = (
                outputs.last_hidden_state
            )  # (batch_size, seq_len, hidden_dim)
            cls_embeddings = hidden_states[:, 0, :]  # [CLS] token embedding

            all_features.append(cls_embeddings.cpu())

    return torch.cat(all_features, dim=0)


def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
