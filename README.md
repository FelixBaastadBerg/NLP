# NLP Project: PCL Detection (SemEval 2022)

This repository contains the implementation for detecting **Patronising and Condescending Language (PCL)** as part of the SemEval 2022 Task 4 (Subtask 1) [cite: 16, 17]. The goal is **Binary Classification** (PCL vs. No PCL) [cite: 36].

---

## 📂 Repository Structure [cite: 72, 73, 97, 98]
* **`BestModel/`**: Contains the trained weights (`model.safetensors`) and code/notebook [cite: 72, 73].
* **`dev.txt`**: Predictions for the official dev set (one 0/1 per line) [cite: 97, 103].
* **`test.txt`**: Predictions for the official test set (one 0/1 per line) [cite: 98, 103].
* **`requirements.txt`**: Dependencies (torch, transformers, etc.).
* **`README.md`**: Project documentation [cite: 126].

---

## ⚙️ Setup & Environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name nlp-venv --display-name "Python (nlp-venv)"
```

---

## 🤖 Model Loading
The model is built upon the **RoBERTa-base** baseline [cite: 50]. To load the weights without a local `config.json`:

```python
from transformers import AutoConfig, RobertaForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file

# Initialize config and architecture
BASE_MODEL = "roberta-base"
config = AutoConfig.from_pretrained(BASE_MODEL, num_labels=2)
model = RobertaForSequenceClassification(config)

# Load weights
state_dict = load_file("./BestModel/model.safetensors")
model.load_state_dict(state_dict, strict=False)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
```

---

## 📊 Performance Benchmarks [cite: 51, 84]
Success is measured using the **F1 score** of the positive (PCL) class [cite: 52, 84].
* **Official Dev Baseline:** 0.48 F1 [cite: 51].
* **Official Test Baseline:** 0.49 F1 [cite: 51].

---

## 📝 Submission Info
* **Deadline:** Wednesday, 4th March, 7pm [cite: 2].
* **Requirement:** Repository must be public after the deadline for assessment [cite: 71].
