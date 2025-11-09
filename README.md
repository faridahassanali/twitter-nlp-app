
# README.md

# LSTM Text Classification System

This project implements a **Text Classification System** using **LSTM (Long Short-Term Memory)** for NLP tasks. It includes a training script and a Streamlit web app for real-time text classification.

---

## Features
- Cleans and tokenizes text automatically.
- Builds a deep learning model (Embedding + BiLSTM).
- Handles both binary and multi-class classification.
- Saves and reloads the best-performing model weights.
- Interactive **Streamlit** web interface for live predictions.


## Folder Structure

project_folder/
│
├── training.csv              # Training dataset
├── test.csv                  # Testing dataset
├── train.py                  # Script to train and save the model
├── app.py                    # Streamlit app for deployment
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── artifacts/                # Folder created automatically after training
    ├── best_weights.h5
    ├── tokenizer.pkl
    ├── label_encoder.pkl
    └── final_model_saved/



## Installation

1. **Clone or extract the project folder**.

2. **Install dependencies:**
   
   pip install -r requirements.txt
   

3. **Add your datasets:**
   - Place your training and test CSV files in the same directory.
   - Ensure they contain the columns `text` and `label` (or similar; the script will auto-detect common names).

---

## 🚀 Training the Model

Run the training script to train the LSTM model and save the best weights:
```bash
python train.py
```

The model and artifacts will be saved in the `artifacts/` folder.

---

## Running the Streamlit App

Once training is complete, deploy the model using Streamlit:
```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## 🧩 Usage
- Enter a statement in the text box.
- Click **Classify**.
- The system will display the predicted class and confidence score.
