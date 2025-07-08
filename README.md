# An endoscopic ultrasound image-based Prediction and Risk Observation System for chronic Pancreatitis Evaluation using Convolutional neural network Technique (PROSPECT)

A deep learning-based solution for detecting pneumonia from chest X-rays in chronic pancreatitis patients, using InceptionV3 and GradCAM for prediction and interpretability.

---

## Objective

Fine-tune a Inception-V3 to distinguish pneumonia from normal chest X-ray, and report your
model’s performance.
---

## Model Overview

- Backbone: Pretrained **InceptionV3** (ImageNet)
- Fine-Tuned Layer: `Mixed_7c` (last Inception block)
- Custom Classifier Head: `2048 → 512 → 1` with Dropout(0.3)
- Loss Function: `BCEWithLogitsLoss`
- Optimizer: `Adam` (learning rate = 1e-4)
- Training Strategy: Early stopping based on validation AUC (patience = 10)
- Evaluation Metrics: **F1 Score**, **AUC-ROC**, **Accuracy**
- Interpretability: **GradCAM** applied on misclassified (false negative) cases

---

## Final Test Results

| Metric    | Value     |
|-----------|-----------|
| Accuracy  | 87.50%    |
| F1 Score  | 0.9076    |
| AUC-ROC   | 0.9553    |
| FN Rate   | 2.1%      |

> Best validation AUC: **0.9891**, saved at Epoch 37  
> Early stopping triggered at Epoch 47 (patience = 10)

---

## Repository Contents

| File/Folder                             | Description                              |
|---------------------------------------- |----------------------------------------- |
| `subhodip-chakraborty-assignment.ipynb` | Full training + evaluation notebook      |
| `requirements.txt`                      | Installable packages                     |
| `Confusion matrix.png`                  | Test set confusion matrix                |
| `false negative.png`                    | P(Type - 2 error) = 0.112                |
| `gradcam_false_negative_final.png`      | GradCAM overlay on false negative case   |


## Model Weights

Model is linked externally to maintain reproducibility while complying with GitHub upload constraints. you can download the full model here:

[Download best_model.pt (97MB)](https://drive.google.com/file/d/1suSwix4gSB1_UOAAWlxFMMfPYqKTGZRz/view?usp=sharing)

---

## How to Run

Install required packages:

```bash
pip install -r requirements.txt

To reproduce results, open and run `subhodip-chakraborty-assignment.ipynb` on Kaggle or any Jupyter-compatible environment.

```

## Hyperparameters

| Hyperparameter   | Value        |
|------------------|--------------|
| Learning Rate    | 1e-4         |
| Batch Size       | 32           |
| Epochs           | 50           |
| Early Stopping Patience | 10    |
| Dropout Rate     | 0.3          |
| Input Size       | 299 × 299 × 3 |
| Loss Function    | BCEWithLogitsLoss |
| Optimizer        | Adam         |
| Scheduler        | None used    |

---

## Clinical Summary

> **This model rapidly flags pneumonia risk with strong accuracy and interpretable attention heatmaps for clinicians.**  
> Leveraging InceptionV3 and GradCAM, it provides early decision support for suspected pneumonia in chronic pancreatitis patients.

---

## Dataset

- **Source**: [PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data)
- **Type**: Resized grayscale chest X-rays (28×28)
- **Classes**: 0 = Normal, 1 = Pneumonia
- **Preprocessing**: Converted to RGB and resized to 299×299 to match InceptionV3 input size

---

## Author

**Subhodip Chakraborty**  
5 years integrated MSc in Statistics, University of Kalyani
Assignment for **Project Research Scientist-I**  
Model built and evaluated using **PyTorch on Kaggle Notebooks**

---
