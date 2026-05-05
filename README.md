# Network Intrusion Detector ML

A machine learning system for network intrusion detection using a hybrid model that combines **Random Forest** and **Isolation Forest** to detect both known and unknown attacks.

---

## How it works

The detector uses a hybrid approach with two complementary models:

- **Random Forest** — trained on labeled data (normal + attacks). Handles known attack patterns with high confidence.
- **Isolation Forest** — trained only on normal traffic. Detects anomalies, including unknown attacks never seen during training.

When a connection is analyzed:
1. Random Forest classifies it first
2. If the confidence for "normal" is below 95%, Isolation Forest re-evaluates it
3. The final decision combines both models

This approach solves a key limitation of traditional classifiers: detecting **zero-day attacks** that were never seen during training.

---

## Dataset

[NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) — a benchmark dataset for network intrusion detection containing 125,000+ real network connections labeled as normal or attack.

Download the dataset and place these two files in the project root:
- `KDDTrain+.txt`
- `KDDTest+.txt`

---

## Results

Evaluated on KDDTest+, which contains attack types not present in the training set:

| Metric | Normal | Suspect |
|---|---|---|
| Precision | 0.85 | 0.93 |
| Recall | 0.91 | 0.88 |
| F1-Score | 0.88 | 0.90 |
| **Accuracy** | | **89%** |

---

## Installation

```bash
git clone https://github.com/JulioVilasBoas/Network-Intrusion-Detector-ML.git
cd Network-Intrusion-Detector-ML

python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows

pip install -r requirements.txt
```

---

## Usage

```bash
python Network-Intrusion-Detection-ML.py
```

The script will:
1. Load and preprocess the NSL-KDD dataset
2. Train the Random Forest and Isolation Forest models
3. Run the hybrid detection on KDDTest+
4. Display the classification report and confusion matrix

---

## Requirements

pandas
numpy
scikit-learn
seaborn
matplotlib

---

## Tech Stack

- **Python 3**
- **Scikit-learn** — Random Forest, Isolation Forest, preprocessing and metrics
- **Pandas / NumPy** — data manipulation
- **Seaborn / Matplotlib** — confusion matrix visualization