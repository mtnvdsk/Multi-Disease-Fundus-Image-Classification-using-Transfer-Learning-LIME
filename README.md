# 🔬 RFMiD Retinal Disease Classification with VGG16 + LIME

This project builds a **multi-label classifier** for detecting retinal diseases from fundus images using **transfer learning (VGG16)** and visual explanations using **LIME (Local Interpretable Model-Agnostic Explanations)**. The RFMiD dataset is used to train, validate, and test the model.

---

## 🧠 Model Architecture

- **Base Model:** `VGG16` pre-trained on ImageNet (frozen)
- **Input Size:** 150x150x3
- **Augmentation:**
  - Random Flip
  - Rotation
  - Zoom
  - Contrast
- **Classifier Head:**
  - GlobalAveragePooling
  - Dense (128 → 64)
  - Output: 28 sigmoid-activated neurons

---

## 🏗️ Training Details

- **Loss Function:** Binary Crossentropy
- **Metrics:** AUROC, Precision, Binary Accuracy
- **Threshold:** 0.1 for predictions
- **Optimizer:** Adam (learning rate = 0.001)
- **Epochs:** 10
- **Batch Size:** 32

---

## 📊 Evaluation

- Model is evaluated on the test set using `model.evaluate()`
- Predictions are generated using `model.predict()`
- Ground truth labels are compared to predictions
- LIME explanations are visualized for interpretability

---

## 🧠 Explainability with LIME

LIME (Local Interpretable Model-agnostic Explanations) is used to:

- Explain model predictions by highlighting important image regions
- Help validate that the model is learning meaningful features

---

## 📦 Installation

Install required packages:

```bash
pip install tensorflow-gpu==2.10.0
pip install lime
pip install pandas numpy matplotlib tqdm scikit-image
```

---

## ▶️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/rfmid-retinal-disease-vgg16.git
   cd rfmid-retinal-disease-vgg16
   ```

2. Organize dataset as shown in the folder structure.

3. Run the script:
   ```bash
   python train_and_explain.py
   ```

4. Evaluate results, check visual LIME overlays, and inspect predictions.

---

## 💾 Model Saving/Loading

```python
model.save("best_model.h5")
model = load_model("best_model.h5")
```

---

## 📈 Results

- Predicts multiple retinal diseases from a single image
- Achieves high AUROC and precision
- LIME provides clear interpretability of model focus regions

---

## 🛠 Future Work

- Hyperparameter tuning and optimization
- Fine-tuning VGG16 layers
- Deploy model as a REST API
- Add Grad-CAM for deeper visual interpretability

---
