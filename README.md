# 🍅 Tomato Disease Diagnosis System

A deep learning system to classify tomato leaf diseases using 4 different models (2 from scratch, 2 transfer learning). The system can distinguish between 6 diseases and a "Not_Tomato" class.

## 📊 Models Implemented
- **SimpleCNN** (Scratch)
- **DeeperCNN** (Scratch)
- **ResNet50** (Transfer Learning)
- **EfficientNet-B0** (Transfer Learning)

## 📈 Results
The best accuracy was achieved by EfficientNet: **98.3%** on the test set.
See `comparison.csv` for complete metrics and `confusion_*.png` for confusion matrices.

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

  Run the Streamlit app

  streamlit run app.py


  
  app.py : Main UI

models.py : Model architectures

evaluate.py : Evaluation script

preprocess.py : Data preprocessing steps

data/ : Dataset (not uploaded, generate via preprocess.py)