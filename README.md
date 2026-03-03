# NLP Project: PCL Detection

This repository contains the implementation for detecting **Patronising and Condescending Language (PCL)**. The goal is **Binary Classification** (PCL vs. No PCL).

---

## 📂 Repository Structure.
* **`BestModel/`**: Contains the trained weights (`model.safetensors`) and code/notebook.
* **`dev.txt`**: Predictions for the official dev set (one 0/1 per line).
* **`test.txt`**: Predictions for the official test set (one 0/1 per line).
* **`requirements.txt`**: Dependencies (torch, transformers, etc.).
* **`README.md`**: Project documentation.
* **`eda.ipynb`**: Initial Data Analysis.

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
The model is built upon the **RoBERTa-base** baseline. To load the weights:

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

