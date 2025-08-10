# 🧪 EGFR Classifier - QSAR-based Activity Prediction

This project is a lightweight yet professional QSAR (Quantitative Structure–Activity Relationship) model for binary classification (Active / Inactive) of small molecules against the **EGFR** target. The goal is to demonstrate predictive modeling skills in cheminformatics, similar to freelance tasks like those from Quantori.

---

## 🚀 Features

- Calculates molecular descriptors from SMILES using **RDKit**
- Trains a **RandomForestClassifier** on cleaned and labeled dataset
- Supports binary classification: **Active** vs **Inactive**
- Tested on unseen SMILES inputs
- Well-structured and modular code, suitable for reuse and enhancement

---

## 🗂️ Project Structure

```
egfr-classifier/
│
├── data/
│   ├── raw/                # Raw dataset
│   └── processed/          # Cleaned dataset
│
├── notebooks/
│   └── egfr-classifier.ipynb  # Main Jupyter notebook

├── README.md
└── requirements.txt
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository:

```bash
git clone https://github.com/sahar-ml-lab/egfr-classifier.git
cd egfr-classifier
```

### 2. Install required packages:

```bash
pip install -r requirements.txt
```

> ⚠️ For RDKit, it's recommended to use Conda:
```bash
conda install -c rdkit rdkit
```

### 3. Run the notebook:

```bash
jupyter notebook notebooks/egfr-classifier.ipynb
```

---

## 🧠 Example Prediction

```python
from joblib import load
from utils.descriptor_calculator import calc_descriptors

model = load("models/rf_model.pkl")

smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
X = calc_descriptors([smiles])
prediction = model.predict(X)[0]

print(f"Prediction result: {'Active' if prediction == 1 else 'Inactive'}")
```

Output:
```
Prediction result: Inactive
```

---

## 📦 Dependencies

- rdkit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- xgboost *(optional for extension)*
- chembl_webresource_client *(optional for data fetching)*

---

## 📬 Contact

If you're interested in using or improving this project, feel free to reach out via GitHub Issues or contact me directly.
saharqi.h@gmail.com
---