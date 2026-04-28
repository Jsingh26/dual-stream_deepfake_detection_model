# 🧠 Dual-Stream Real/Fake Image Detector
### EfficientNet-B4 + FFT-CNN with Cross-Attention Fusion  
**📊 94.5% Accuracy on CIFAKE Dataset**

---

![Training Curves](training_curves.png)

---

## 📋 Overview

This project implements a **dual-stream deep learning architecture** to detect AI-generated (deepfake) images by combining **spatial** and **frequency-domain** features.

- **Spatial Stream**: EfficientNet-B4 extracts visual features from RGB images  
- **Frequency Stream**: CNN processes FFT magnitude spectrum to capture hidden artifacts  
- **Fusion**: Cross-attention style gating combines both streams effectively  

---

## 🧩 Architecture

- **Spatial Branch**: EfficientNet-B4 (pretrained, last 3 blocks fine-tuned)  
- **Frequency Branch**: CNN on log-scaled FFT magnitude  
- **Fusion Layer**: Learnable gating mechanism  
- **Classifier**: Fully connected layers with dropout  

---

## 🚀 Results

| Model                 | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| VGG-16               | 87.6%    | 86.8%     | 88.9%  | 87.8%    |
| ResNet-50            | 89.3%    | 88.6%     | 90.5%  | 89.5%    |
| FFT + CNN Only       | 88.7%    | 87.9%     | 90.1%  | 89.0%    |
| EfficientNet-B4 Only | 91.2%    | 90.4%     | 92.0%  | 91.2%    |
| **Proposed Model**   | **94.5%**| **93.1%** | **95.9%** | **94.5%** |

> ✅ +3.3% improvement over EfficientNet-B4 baseline

---

## 📊 Dataset

- **CIFAKE Dataset** (100,000 images)
  - 50K Real  
  - 50K AI-generated (Stable Diffusion)  
- Input Size: **224 × 224**

---

## 🛠️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/dual-stream-deepfake-detector.git
cd dual-stream-deepfake-detector
pip install -r requirements.txt
```
▶️ Usage
python inference.py --image path/to/image.jpg
📁 Project Structure
dual-stream-deepfake-detector/
├── README.md
├── model.pt
├── training_log.csv
├── training_curves.png
├── notebook/
│   └── Dual_Stream_Detector.ipynb
├── src/
│   ├── model.py
│   └── ...
├── inference.py
└── requirements.txt
📈 Training Details
Epochs: 20 (Early Stopping)
Batch Size: 64
Optimizer: AdamW
Learning Rate:
1e-5 (Spatial Stream)
1e-4 (Frequency Stream & Fusion)
Loss Function: BCEWithLogitsLoss
Hardware: Tesla T4 ×2 (Kaggle)
📚 Research Paper

Detection of AI-Generated and Deepfake Images Using Dual-Stream Deep Learning with Spatial and Frequency Domain Analysis

Authors:
Japinder Singh, Faisal Rais, Dr. Mohd Izhar

🔧 Technologies Used
PyTorch
EfficientNet-B4
Fast Fourier Transform (FFT)
DataParallel (Multi-GPU)
Cosine Annealing Scheduler
⭐ Future Work
Extend to video deepfake detection (ConvLSTM / Transformer)
Evaluate on FaceForensics++, Celeb-DF, DFDC
Improve robustness against adversarial attacks
Optimize for real-time deployment
📌 License

MIT License

💡 Author

Japinder Singh
B.Tech CSE | AI & Deep Learning
