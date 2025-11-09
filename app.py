import streamlit as st
import re, pickle, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os

# ============================================================
# NLTK SETUP
# ============================================================
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))  

stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# ============================================================
# TEXT CLEANING FUNCTION (same as train.py)
# ============================================================
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# ============================================================
# MODEL REBUILDING FUNCTION
# ============================================================
def build_model(vocab_size, embedding_dim, maxlen, num_classes):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=maxlen),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1 if num_classes == 2 else num_classes,
              activation='sigmoid' if num_classes == 2 else 'softmax')
    ])
    loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model

# ============================================================
# UPLOAD ARTIFACTS IF MISSING
# ============================================================
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

if not os.listdir(ARTIFACTS_DIR):
    st.info("Artifacts folder is empty. Please upload your model files:")
    uploaded_files = st.file_uploader(
        "Upload model files (tokenizer.pkl, label_encoder.pkl, best_weights.keras)",
        type=["pkl", "keras"], accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(ARTIFACTS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded successfully!")

# ============================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================
@st.cache_resource
def load_model_and_artifacts():
    # Load tokenizer and label encoder
    with open(os.path.join(ARTIFACTS_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    vocab_size = min(30000, len(tokenizer.word_index) + 1)
    num_classes = len(le.classes_)
    maxlen = 150

    # Recreate model
    model = build_model(vocab_size, 128, maxlen, num_classes)
    model.build(input_shape=(None, maxlen))

    # Load best weights
    model.load_weights(os.path.join(ARTIFACTS_DIR, "best_weights.keras"))
    return tokenizer, le, model

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Twitter Sentiment Classifier")
st.title("Twitter Sentiment Classifier")
st.write("Enter a tweet below to predict its sentiment using an LSTM-based model.")

# Only load model if artifacts exist
if os.listdir(ARTIFACTS_DIR):
    tokenizer, le, model = load_model_and_artifacts()

    # Input field
    user_input = st.text_area("Tweet text:", height=120)

    if st.button("Predict Sentiment"):
        if not user_input.strip():
            st.warning("Please enter a tweet to analyze.")
        else:
            cleaned = clean_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=150)
            preds = model.predict(padded)

            if len(le.classes_) == 2:
                pred_idx = int(preds.flatten() > 0.5)
            else:
                pred_idx = np.argmax(preds, axis=1)[0]

            sentiment = le.inverse_transform([pred_idx])[0]
            st.success(f"**Predicted Sentiment:** {sentiment}")

            st.write("Model confidence:", float(np.max(preds)))
else:
    st.warning("Artifacts folder is empty. Upload required files to use the app.")

st.markdown("---")
st.caption("Built with TensorFlow & Streamlit")
