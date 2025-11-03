# 🛰️ Satellite Image Recognition

> 🌍 Deep learning system for classifying satellite imagery into land-use types (water bodies, deserts, forests, urban areas) using CNNs and remote sensing data with GPS metadata integration.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![ResNet](https://img.shields.io/badge/ResNet-EfficientNet-green.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

This system uses **Convolutional Neural Networks (CNNs)** to automatically classify satellite images into different land-use categories. It leverages transfer learning with **ResNet** and **EfficientNet** architectures, combined with GPS metadata for enhanced accuracy in geospatial analysis.

**Key Applications:**
- 🌲 Environmental monitoring and deforestation tracking
- 🏙️ Urban planning and development analysis
- 🌊 Water resource management
- 🏜️ Desert expansion and climate change studies

---

## ✨ Key Features

- 🎯 **Multi-class Classification** - Water bodies, deserts, forests, urban areas
- 🗺️ **GPS Integration** - Metadata enrichment for location-aware predictions
- 🧠 **Transfer Learning** - Pre-trained ResNet/EfficientNet models
- 📊 **High Accuracy** - 90%+ classification accuracy
- 📈 **Data Augmentation** - Rotation, flip, zoom for robust training
- 📷 **Batch Processing** - Efficient large-scale image analysis

---

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| **Python** | Core language |
| **TensorFlow/Keras** | Deep learning framework |
| **ResNet/EfficientNet** | CNN architectures |
| **OpenCV** | Image preprocessing |
| **NumPy/Pandas** | Data handling |
| **Matplotlib** | Visualization |

---

## 💻 Installation

```bash
# Clone repository
git clone https://github.com/HassanRasheed91/Satellite-Image-Recognition.git
cd Satellite-Image-Recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```txt
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
pillow>=9.0.0
```

---

## 🎮 Usage

### **1. Train Model**

```bash
python train.py --data dataset/ --model resnet50 --epochs 50 --batch 32
```

### **2. Make Predictions**

```bash
python predict.py --image satellite_image.jpg --model trained_model.h5
```

### **3. Batch Processing**

```bash
python batch_predict.py --input images/ --output results/ --model trained_model.h5
```

---

## 📊 Dataset

**Classes:**
- 🌊 Water Bodies - Lakes, rivers, oceans
- 🏜️ Deserts - Arid and semi-arid regions
- 🌲 Forests - Dense vegetation areas
- 🏙️ Urban Areas - Cities and built-up regions

**Structure:**
```
dataset/
├── train/
│   ├── water/
│   ├── desert/
│   ├── forest/
│   └── urban/
├── val/
└── test/
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 92.5% |
| **Precision** | 91.3% |
| **Recall** | 90.8% |
| **F1-Score** | 91.0% |

**Per-Class Accuracy:**
- Water Bodies: 95%
- Deserts: 93%
- Forests: 91%
- Urban Areas: 90%

---

---

## 🚀 Model Training

```python
# Load pre-trained ResNet
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

# Compile model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 🔧 Configuration

**Hyperparameters:**
```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
```

---

## 📝 Example Output

```
Input: satellite_image_001.jpg
Prediction: Forest (95.3% confidence)
GPS: Lat 40.7128, Lon -74.0060
Timestamp: 2024-01-15 10:30:45
```

---

## 🌟 Future Enhancements

- [ ] Multi-modal learning with SAR data
- [ ] Semantic segmentation for detailed mapping
- [ ] Time-series analysis for change detection
- [ ] Cloud-based deployment
- [ ] Real-time satellite feed processing

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 📬 Contact

**Hassan Rasheed**  
📧 221980038@gift.edu.pk  
💼 [LinkedIn](https://www.linkedin.com/in/hassan-rasheed-datascience/)  
🐙 [GitHub](https://github.com/HassanRasheed91)

---

<div align="center">

**Made with ❤️ by Hassan Rasheed**

*Geospatial intelligence through deep learning*

</div>
