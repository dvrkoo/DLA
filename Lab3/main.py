import torch
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
)
from utils import (
    extract_cls_features,
    tokenize_function,
    compute_metrics,
    get_datasets,
    flatten_report,
)
import argparse
import wandb
from peft import get_peft_model, LoraConfig, TaskType
from datasets import DatasetDict


def run_baseline(device, args):
    # Load dataset
    ds_train, ds_valid, ds_test = get_datasets(args.dataset_name)

    if args.wandb:
        run_name = f"svm-cls-{args.dataset_name}"
        wandb.init(
            project=f"distilbert-{args.dataset_name}",
            name=run_name,
            config=vars(args),
        )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

    # Extract features
    print("Extracting features for training set...")
    Xs_train = extract_cls_features(ds_train, tokenizer, model, device=device)
    print("Extracting features for validation set...")
    Xs_valid = extract_cls_features(ds_valid, tokenizer, model, device=device)
    print("Extracting features for test set...")
    Xs_test = extract_cls_features(ds_test, tokenizer, model, device=device)

    ys_train = ds_train["label"]
    ys_valid = ds_valid["label"]
    ys_test = ds_test["label"]

    # Train SVM
    print("Training SVM...")
    svc = LinearSVC(dual="auto")  # dual="auto" handles new default behavior
    svc.fit(Xs_train, ys_train)

    # Evaluation
    print("\nValidation Report:")
    val_preds = svc.predict(Xs_valid)
    val_report = classification_report(ys_valid, val_preds, output_dict=True)
    print(classification_report(ys_valid, val_preds))

    print("\nTest Report:")
    test_preds = svc.predict(Xs_test)
    test_report = classification_report(ys_test, test_preds, output_dict=True)
    print(classification_report(ys_test, test_preds))

    if args.wandb:
        print("Logging metrics to wandb...")
        wandb.log(flatten_report(val_report, "validation"))
        wandb.log(flatten_report(test_report, "test"))
        wandb.finish()


def run_finetune(device, args):
    # Load dataset
    ds_train, ds_valid, ds_test = get_datasets(args.dataset_name)

    # Combine into a single DatasetDict for easier handling
    dataset = DatasetDict({"train": ds_train, "validation": ds_valid, "test": ds_test})

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    cols = tokenized_dataset["train"].column_names
    print(f"Train columns: {cols}")
    assert all(
        x in cols for x in ["text", "label", "input_ids", "attention_mask"]
    ), "Missing required columns"

    if args.dataset_name == "ag_news":
        num_labels = 4
        id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
    elif args.dataset_name == "emotion":
        num_labels = 6
        id2label = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise",
        }
        label2id = {
            "sadness": 0,
            "joy": 1,
            "love": 2,
            "anger": 3,
            "fear": 4,
            "surprise": 5,
        }
    else:  # Default to rotten_tomatoes or other binary tasks
        num_labels = 2
        id2label = {0: "negative", 1: "positive"}
        label2id = {"negative": 0, "positive": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
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
        peft_str = (
            f"lora-r{args.lora_r}-a{args.lora_alpha}" if args.peft else "full-finetune"
        )
        run_name = f"distilbert-{args.dataset_name}-{peft_str}-lr{args.lr}-b{args.batch_size}-e{args.epochs}"
        wandb.init(
            project=f"distilbert-{args.dataset_name}",
            name=run_name,
            config=vars(args),
        )

    training_args = TrainingArguments(
        output_dir=f"./results/{args.dataset_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=f"./logs/{args.dataset_name}",
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

    # Final evaluation on the test set
    print("\n--- Evaluating on Test Set ---")
    test_results = (
        trainer.predict(tokenized_dataset["validation"])
        if args.dataset_name == "sst2"
        else trainer.predict(tokenized_dataset["test"])
    )
    print(test_results.metrics)

    if args.wandb:
        # Log test metrics to wandb. The trainer already logged train/eval metrics.
        wandb.log(test_results.metrics)
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="DistilBERT Experiments")
    parser.add_argument("--baseline", action="store_true", help="Run CLS+SVM baseline")
    parser.add_argument(
        "--finetune", action="store_true", help="Run DistilBERT fine-tuning"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="rotten_tomatoes",
        choices=[
            "rotten_tomatoes",
            "sst2",
        ],
        help="Dataset to use for the experiment.",
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

    # PEFT/LoRA arguments
    parser.add_argument(
        "--peft",
        action="store_true",
        help="Use PEFT (LoRA) for fine-tuning",
    )
    parser.add_argument(
        "--lora_r", type=int, default=8, help="LoRA rank (if using PEFT)"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (if using PEFT)"
    )

    args = parser.parse_args()

    if args.baseline:
        run_baseline(args.device, args)
    elif args.finetune:
        run_finetune(args.device, args)
    else:
        print("Please specify a mode to run: --baseline or --finetune")


if __name__ == "__main__":
    main()
