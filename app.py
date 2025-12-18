import joblib
import re
import streamlit as st
import pandas as pd

# ======================
# 1. LOAD MODEL
# ======================
svm_model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ======================
# 2. LOAD KAMUS SLANG
# ======================
kamus = pd.read_excel("kamuskatabaku.xlsx")
slang_dict = dict(zip(kamus['tidak_baku'], kamus['kata_baku']))

# ======================
# 3. STOPWORDS
# ======================
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
stop_words = set(StopWordRemoverFactory().get_stop_words())

# ======================
# 4. PREPROCESSING FUNCTIONS
# ======================
def cleaning(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalisasi_kata(text):
    words = text.split()
    hasil = []
    for w in words:
        hasil.append(slang_dict.get(w, w))
    return " ".join(hasil)

def tokenize(text):
    return text.split()

def stopword_removal(tokens):
    return [word for word in tokens if word not in stop_words]

# 👉 jika kamu pakai stemming saat training
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory().create_stemmer()

def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

# ======================
# 5. PIPELINE PREPROCESS
# ======================
def preprocess_text(text):
    text = cleaning(text)
    text = normalisasi_kata(text)
    tokens = tokenize(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return " ".join(tokens)   # WAJIB string

# ======================
# 6. PREDIKSI FUNCTION
# ======================
def predict_sentiment(text):
    clean_text = preprocess_text(text)
    vector = tfidf.transform([clean_text])
    pred = svm_model.predict(vector)
    return pred[0]

# ======================
# 7. STREAMLIT UI
# ======================
st.set_page_config(page_title="Analisis Sentimen Shopee", layout="centered")
st.title("📊 Analisis Sentimen Ulasan Shopee")
st.caption("Model: SVM + TF-IDF")

text = st.text_area("Masukkan ulasan:")

if st.button("Prediksi Sentimen"):
    if text.strip() == "":
        st.warning("⚠️ Ulasan tidak boleh kosong")
    else:
        hasil = predict_sentiment(text)
        if hasil == "positif":
            st.success("✅ Sentimen POSITIF")
        else:
            st.error("❌ Sentimen NEGATIF")
