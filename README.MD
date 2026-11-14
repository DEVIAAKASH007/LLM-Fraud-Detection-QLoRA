# LLM-Based Fraud Detection using QLoRA

An LLM-powered fraud detection system that fine-tunes **Qwen2.5-1.5B-Instruct** using **QLoRA (Quantized Low-Rank Adaptation)** to classify financial transactions based on their fraud risk.

## 📌 Project Overview

Traditional fraud detection systems often rely on machine learning classifiers that require extensive feature engineering and labeled training data.

This project explores the use of Large Language Models (LLMs) for transaction fraud classification. A pretrained Qwen2.5-1.5B-Instruct model is fine-tuned using QLoRA to identify potentially fraudulent transactions.

The model analyzes transaction information and predicts whether a transaction belongs to a **HIGH RISK** or **LOW RISK** category.

## 🎯 Objectives

- Apply Large Language Models to financial fraud detection.
- Fine-tune a pretrained LLM using Parameter-Efficient Fine-Tuning (PEFT).
- Reduce GPU memory requirements using 4-bit quantization.
- Evaluate the model using standard classification metrics.
- Explore the potential of LLMs for structured transaction analysis.

## 🧠 Model Architecture

**Base Model:** Qwen2.5-1.5B-Instruct

**Fine-Tuning Technique:** QLoRA

The project uses the following approach:

1. Load the pretrained Qwen2.5-1.5B-Instruct model.
2. Apply 4-bit quantization using BitsAndBytes.
3. Configure LoRA adapters for parameter-efficient fine-tuning.
4. Prepare transaction data in an instruction-based format.
5. Fine-tune the model to classify transaction fraud risk.
6. Generate predictions and evaluate model performance.

### QLoRA Configuration

| Parameter | Configuration |
|---|---|
| Base Model | Qwen2.5-1.5B-Instruct |
| Quantization | 4-bit |
| Quantization Type | NF4 |
| Fine-Tuning | QLoRA |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Framework | PyTorch |

## ⚙️ Technology Stack

- **Programming:** Python
- **LLM:** Qwen2.5-1.5B-Instruct
- **Fine-Tuning:** QLoRA, LoRA, PEFT
- **Quantization:** BitsAndBytes
- **Deep Learning:** PyTorch
- **Model Training:** Hugging Face Transformers, TRL
- **Evaluation:** Scikit-learn

## 📊 Evaluation Metrics

The model is evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help assess the model's ability to identify fraudulent transactions while minimizing false positives and false negatives.

## 📁 Project Structure

```text
LLM-Fraud-Detection-QLoRA/
│
├── FRAUD_DETECTION.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/DEVIAAKASH007/LLM-Fraud-Detection-QLoRA.git
cd LLM-Fraud-Detection-QLoRA
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

Open `FRAUD_DETECTION.ipynb` using Jupyter Notebook or Visual Studio Code and execute the cells in sequence.

**Note:** Model fine-tuning with 4-bit quantization requires a compatible environment with the necessary GPU and CUDA support.

## 🔮 Future Improvements

- Experiment with larger instruction-tuned language models.
- Improve performance using larger and more diverse transaction datasets.
- Perform hyperparameter optimization for LoRA configurations.
- Compare LLM performance against traditional ML fraud detection models.
- Develop an API for real-time transaction risk prediction.
- Build an interactive fraud detection dashboard.

## 👨‍💻 Author

**Kondapalli Devi Aakash**

B.Tech – Artificial Intelligence

GitHub: [DEVIAAKASH007](https://github.com/DEVIAAKASH007)

---

⭐ If you find this project interesting, consider giving the repository a star!
