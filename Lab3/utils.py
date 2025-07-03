import torch
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
)
from datasets import load_dataset
import numpy as np


def flatten_report(report_dict: dict, prefix: str):
    """Flattens the classification_report dictionary for wandb logging."""
    flat_report = {}
    for key, value in report_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_report[f"{prefix}_{key.replace(' ', '_')}_{sub_key}"] = sub_value
        else:
            flat_report[f"{prefix}_{key}"] = value
    return flat_report


def get_datasets(dataset_name: str):
    """Loads and splits the specified dataset."""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    if dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.rename_column("sentence", "text")
        # -----------------------
    else:
        dataset = load_dataset(dataset_name)

    if dataset_name in ["rotten_tomatoes", "sst2"]:
        ds_train = dataset["train"]
        ds_valid = dataset["validation"]
        ds_test = dataset["test"]
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    print(
        f"Train records: {len(ds_train)}, Validation records: {len(ds_valid)}, Test records: {len(ds_test)}"
    )
    return ds_train, ds_valid, ds_test


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


def compute_metrics(eval_pred):
    """
    Computes accuracy, F1, precision, and recall for a given set of predictions.
    Automatically handles binary or multiclass classification.
    """
    predictions, labels = eval_pred
    # The 'predictions' are logits, so we take the argmax to get the predicted class
    predictions = np.argmax(predictions, axis=1)

    # --- THIS IS THE FIX ---
    # Determine if the task is binary or multiclass based on the number of unique labels
    num_unique_labels = len(np.unique(labels))

    if num_unique_labels > 2:
        # For multiclass, 'macro' is a robust averaging strategy
        average_type = "macro"
    else:
        # For binary, 'binary' is the correct setting
        average_type = "binary"
    # -----------------------

    # Calculate the metrics using the determined average type
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average_type
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
