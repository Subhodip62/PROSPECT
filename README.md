# An endoscopic ultrasound image-based Prediction and Risk Observation System for chronic Pancreatitis Evaluation using Convolutional neural network Technique (PROSPECT)

A deep learning-based solution for detecting pneumonia from chest X-rays in chronic pancreatitis patients, using InceptionV3 and GradCAM for prediction and interpretability.

---

## 🔍 Objective

Fine-tune a Inception-V3 to distinguish pneumonia from normal chest X-ray, and report your
model’s performance.
---

## 🧠 Model Overview

- Backbone: Pretrained **InceptionV3** (ImageNet)
- Fine-Tuned Layer: `Mixed_7c` (last Inception block)
- Custom Classifier Head: `2048 → 512 → 1` with Dropout(0.3)
- Loss Function: `BCEWithLogitsLoss`
- Optimizer: `Adam` (learning rate = 1e-4)
- Training Strategy: Early stopping based on validation AUC (patience = 10)
- Evaluation Metrics: **F1 Score**, **AUC-ROC**, **Accuracy**
- Interpretability: **GradCAM** applied on misclassified (false negative) cases

---

## 📊 Final Test Results

| Metric    | Value     |
|-----------|-----------|
| Accuracy  | 87.50%    |
| F1 Score  | 0.9076    |
| AUC-ROC   | 0.9553    |
| FN Rate   | 2.1%      |

> Best validation AUC: **0.9891**, saved at Epoch 37  
> Early stopping triggered at Epoch 47 (patience = 10)

---

## 📁 Repository Contents

| File/Folder                            | Description                             |
|----------------------------------------|-----------------------------------------|
| `subhodip_chakraborty_assignment.ipynb` | Full training + evaluation notebook      |
| `requirements.txt`                     | Installable packages                     |
| `figures/confusion_matrix.png`         | Test set confusion matrix                |
| `figures/gradcam_false_negative.png`   | GradCAM overlay on false negative case   |


## 🔗 Model Weights

Model is linked externally to maintain reproducibility while complying with GitHub upload constraints. you can download the full model here:

➡️ [Download best_model.pt (97MB)](https://drive.google.com/file/d/1suSwix4gSB1_UOAAWlxFMMfPYqKTGZRz/view?usp=sharing)

---

## 🛠️ How to Run

Install required packages:

```bash
pip install -r requirements.txt

To reproduce results, open and run `subhodip_chakraborty_assignment.ipynb` on Kaggle or any Jupyter-compatible environment.

