# 🔍 ELA Guard — Hybrid Image Tamper Detection System

> **Error Level Analysis + SVM + CNN Ensemble** deployed as a Django web application

---

## 📋 Overview

ELA Guard detects image tampering using a hybrid AI pipeline:

1. **Error Level Analysis (ELA)** — preprocessing that reveals re-saved regions
2. **SVM** — trained on 256-bin ELA histograms (lightweight, fast)
3. **CNN** — trained directly on ELA image maps (deep feature learning)
4. **Ensemble Fusion** — `Final Score = 0.8 × CNN + 0.2 × SVM`

| Model    | Target Accuracy |
|----------|----------------|
| SVM      | ~83%           |
| CNN      | ~92%           |
| Ensemble | ~93%           |

---

## 🗂 Project Structure

```
ela_tamper_detection/
├── core/
│   ├── ela.py            # ELA preprocessing module
│   ├── cnn_model.py      # CNN architecture (TamperCNN)
│   ├── dataset.py        # Dataset loaders, feature extractors
│   └── ensemble.py       # Ensemble predictor (inference)
│
├── training/
│   ├── train_svm.py      # SVM training script
│   └── train_cnn.py      # CNN training script
│
├── models_saved/
│   ├── svm.pkl           # Trained SVM + scaler bundle
│   └── cnn.pt            # Trained CNN weights
│
├── django_app/
│   ├── manage.py
│   ├── tamper_project/   # Django settings, URLs, WSGI
│   ├── detector/         # Django app
│   │   ├── models.py     # AnalysisResult DB model
│   │   ├── views.py      # Upload, analyze, result, history
│   │   ├── urls.py
│   │   ├── templates/detector/
│   │   │   ├── base.html
│   │   │   ├── index.html     # Upload page
│   │   │   ├── result.html    # Results with ELA comparison
│   │   │   └── history.html   # Analysis history
│   │   └── static/detector/
│   │       ├── css/style.css
│   │       └── js/main.js
│   └── media/
│       ├── uploads/       # Original uploaded images
│       └── ela_outputs/   # Generated ELA maps
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset (CASIA v2)

Download CASIA v2 from: https://github.com/namtpham/casia2groundtruth

Expected structure:
```
CASIA_v2/
├── Au/   # ~7,500 authentic images (.jpg, .tiff)
└── Tp/   # ~5,100 tampered images (.jpg, .tiff)
```

### 3. Train SVM

```bash
cd ela_tamper_detection
python training/train_svm.py --dataset /path/to/CASIA_v2 --output models_saved/svm.pkl
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | Path to CASIA v2 root |
| `--output` | `models_saved/svm.pkl` | Output path |

### 4. Train CNN

```bash
python training/train_cnn.py \
  --dataset /path/to/CASIA_v2 \
  --output models_saved/cnn.pt \
  --epochs 25 \
  --batch_size 32
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | Path to CASIA v2 root |
| `--output` | `models_saved/cnn.pt` | Output path |
| `--epochs` | 25 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 5e-4 | Learning rate |
| `--num_workers` | 4 | DataLoader workers |

> **GPU Training**: Automatically detected. Set `CUDA_VISIBLE_DEVICES=0` to select GPU.

### 5. Run Django Web App

```bash
cd django_app

# First-time setup
python manage.py migrate
python manage.py collectstatic --noinput

# Start server
python manage.py runserver
```

Open http://localhost:8000 in your browser.

---

## 🌐 Web Application Features

- **Drag & drop** image upload (or click to browse)
- Supports: PNG, JPG, JPEG, WebP, GIF, HEIC (max 10MB)
- **Side-by-side comparison** — Original vs ELA Map
- **Score breakdown**: Ensemble, CNN, and SVM probabilities with animated bars
- **Analysis history** with filterable table
- SQLite storage (no additional DB setup needed)

---

## 🧠 Model Architecture

### CNN (TamperCNN)

```
Input: (B, 3, 224, 224) — ELA image

Conv Block 1: Conv(64) → BN → ReLU → MaxPool   → 112×112
Conv Block 2: Conv(128) → BN → ReLU → MaxPool  → 56×56
Conv Block 3: Conv(256) → BN → ReLU → MaxPool  → 28×28
Conv Block 4: Conv(512) → BN → ReLU → MaxPool  → 14×14

GlobalAvgPool → 512×4×4

FC 512 → Dropout(0.5) → FC 256 → Dropout(0.5) → FC 2 → Softmax
```

Training:
- Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
- Scheduler: Cosine Annealing
- Augmentation: RandomFlip, Rotation±15°, ColorJitter

### SVM

- Features: 256-bin normalized ELA histogram
- Kernel: RBF, C=100, gamma='scale'
- Preprocessing: StandardScaler

---

## 🔬 ELA Algorithm

```python
# 1. Recompress at JPEG quality 90
recompressed = save_as_jpeg(original, quality=90)

# 2. Compute absolute difference  
diff = |original - recompressed|

# 3. Amplify ×15 for visualization
ela_map = clip(diff * 15, 0, 255)

# 4. Extract features
histogram = 256-bin normalized histogram of grayscale ELA
```

Tampered regions appear **brighter** because they were saved at a different quality level than the rest of the image.

---

## 🔥 Ensemble Fusion

```
Final Score = (0.8 × CNN_tamper_prob) + (0.2 × SVM_tamper_prob)

If Final Score ≥ 0.50 → Tampered
Else                  → Authentic
```

---

## 📊 Evaluation Metrics

Both training scripts output:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Full classification report (Train / Val / Test splits)
- Log files: `svm_training.log`, `cnn_training.log`

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DJANGO_SECRET_KEY` | dev key | Production secret key |
| `DEBUG` | `True` | Set to `False` in production |
| `ALLOWED_HOSTS` | `localhost 127.0.0.1` | Space-separated hosts |
| `SVM_MODEL_PATH` | `models_saved/svm.pkl` | SVM model path |
| `CNN_MODEL_PATH` | `models_saved/cnn.pt` | CNN model path |

---

## 🚀 Production Deployment

```bash
# Set environment variables
export DJANGO_SECRET_KEY="your-secret-key-here"
export DEBUG=False
export ALLOWED_HOSTS="your-domain.com"

# Collect static files
python manage.py collectstatic

# Use gunicorn
pip install gunicorn
gunicorn tamper_project.wsgi:application --bind 0.0.0.0:8000 --workers 4
```

---

## 🧩 Optional Enhancements (Not Implemented)

- **Grad-CAM** visualization for CNN decision explanation
- **Vision Transformer (ViT)** as alternative backbone
- **Tampered region localization** with segmentation head
- **Video tampering detection** via frame-by-frame analysis

---

## 📄 License

MIT License — Free for research and educational use.
"# CAPSTONE" 
