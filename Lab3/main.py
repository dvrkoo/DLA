import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)
from tqdm import tqdm
import argparse
import wandb
from peft import get_peft_model, LoraConfig, TaskType


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


def run_baseline(device):
    # Load dataset
    dataset = load_dataset("rotten_tomatoes")
    ds_train = dataset["train"]
    ds_valid = dataset["validation"]
    ds_test = dataset["test"]

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")

    # Extract features
    Xs_train = extract_cls_features(ds_train, tokenizer, model, device=device)
    Xs_valid = extract_cls_features(ds_valid, tokenizer, model, device=device)
    Xs_test = extract_cls_features(ds_test, tokenizer, model, device=device)

    ys_train = ds_train["label"]
    ys_valid = ds_valid["label"]
    ys_test = ds_test["label"]

    # Train SVM
    svc = LinearSVC()
    svc.fit(Xs_train, ys_train)

    # Evaluation
    print("\nValidation Report:")
    print(classification_report(ys_valid, svc.predict(Xs_valid)))

    print("Test Report:")
    print(classification_report(ys_test, svc.predict(Xs_test)))


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


def run_finetune(device, args):
    dataset = load_dataset("rotten_tomatoes")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    cols = tokenized_dataset["train"].column_names
    print(f"Train columns: {cols}")
    assert all(
        x in cols for x in ["text", "label", "input_ids", "attention_mask"]
    ), "Missing required columns"

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    ).to(device)
    if args.peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.wandb:
        run_name = f"distilbert-rotten-tomatoes-{args.lr}_batch{args.batch_size}_epochs{args.epochs}_{'lora' if args.peft else ''}_{args.lora_r if args.peft else ''}_{args.lora_alpha if args.peft else ''}"
        wandb.init(
            project="distilbert-rotten-tomatoes", name=run_name, config=vars(args)
        )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=["wandb"] if args.wandb else ["console"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print("Evaluation result on validation set:")
    print(eval_result)

    if args.wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="DistilBERT Experiments")
    parser.add_argument("--baseline", action="store_true", help="Run CLS+SVM baseline")
    parser.add_argument(
        "--finetune", action="store_true", help="Run DistilBERT fine-tuning"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Hyperparameters for fine-tuning
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--peft",
        action="store_true",
        help="Use PEFT for fine-tuning (not implemented in this example)",
    )
    parser.add_argument(
        "--lora_r", type=int, default=8, help="LoRA rank (if using PEFT)"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (if using PEFT)"
    )

    args = parser.parse_args()

    if args.baseline:
        run_baseline(args.device)
    elif args.finetune:
        run_finetune(args.device, args)


if __name__ == "__main__":
    main()
