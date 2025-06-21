# 📝 Sentiment Analysis with DistilBERT — Full & Efficient Fine-Tuning

This project demonstrates two approaches to fine-tuning [DistilBERT](https://huggingface.co/distilbert-base-uncased) for sentiment classification using the Rotten Tomatoes dataset:

- ✅ **Full fine-tuning** of the model
- ⚡ **Efficient fine-tuning** using [PEFT](https://github.com/huggingface/peft) (Low-Rank Adaptation - LoRA)

Tracking training logs is supported with both **Weights & Biases (wandb)** and **Comet ML** out-of-the-box.

---

## 📦 Installation

```
Required libraries:

    transformers

    datasets

    scikit-learn

    torch

    wandb

    peft

    accelerate

    tqdm

    comet-ml (optional)
```

## 🚀 Usage
Run Full Fine-Tuning
```
python main.py --finetune --wandb --epochs 5 --lr 2e-5 --batch_size 16
```
Run Efficient Fine-Tuning (LoRA)
```
python main.py --finetune --peft --wandb --epochs 5 --lr 2e-4 --batch_size 16
```
Run Baseline (DistilBERT CLS + Linear SVM)
```
python main.py --baseline
```
## 🧪 Model Options
Mode	Command Line	Description
Full Fine-tune	--finetune	Fine-tune all parameters of DistilBERT
LoRA (PEFT)	--peft	Fine-tune only a small number via LoRA
Baseline	--baseline	Use BERT CLS token features + Linear SVM
⚙️ Configurable Hyperparameters

You can override training configuration directly from the CLI:

python main.py --finetune --epochs 10 --lr 2e-4 --batch_size 32 --wandb

Flag	Description	Default
--epochs	Number of training epochs	3
--lr	Learning rate	2e-5
--batch_size	Batch size	16
--device	cpu or cuda	cuda if available
--wandb	Enable logging to Weights & Biases	Off by default
--peft	Use LoRA for efficient fine-tuning	Off by default
## 📊 Logging & Monitoring

By default, logs are sent to both:

    Weights & Biases

    Comet ML

To enable wandb logging:

export WANDB_API_KEY=your_key_here

## 📈 Results
More examples can be seen at: https://wandb.ai/niccolo-marini-universit-degli-studi-di-firenze/distilbert-rotten-tomatoes?nw=nwuserniccolomarini
![alt text](./plots/precision.png)
![alt text](./plots/loss.png)
Mode	Epochs	Batch Size	Accuracy (Val)	Notes
Full Fine-tune	5	16	~96%	High resource use
LoRA (PEFT)	5	16	~95-96%	Fast + efficient
SVM baseline	-	-	~90%	Only CLS features
🧠 Notes

    Efficient fine-tuning can cut training time and memory use by >90% while maintaining similar accuracy.

    LoRA introduces only a few trainable parameters (~500k) compared to the full 65M in DistilBERT.

    You can easily extend this to other datasets like IMDb, AG News, or SST2.

## 📄 License

MIT License
🙌 Acknowledgments

    Hugging Face Transformers

    Hugging Face PEFT

    Weights & Biases

    Rotten Tomatoes Dataset


---

