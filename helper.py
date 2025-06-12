import joblib
import warnings
warnings.filterwarnings('ignore')
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import numpy as np
model_path = 'model.joblib'
model = joblib.load(model_path)
vectorizer_path = 'vectorizer.joblib'
vectorizer = joblib.load(vectorizer_path)

def remove_stopwords(text):
    filtered_text = ''
    for i in text.split():
        if i not in stop_words:
            filtered_text += ' ' + i
    return filtered_text.strip()

def remove_words_less_than_two_chars(text):
    filtered_text = ''
    for i in text.split():
        if len(i) > 2:
            filtered_text += ' ' + i
    return filtered_text.strip()

def stemming_text(text):
    processed_text = ''

    for i in text.split():
        stem_word = stemmer.stem(i)
        processed_text += ' ' + stem_word

    return processed_text.strip()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    text = remove_stopwords(text)
    text = remove_words_less_than_two_chars(text)
    text = stemming_text(text)
    # text = spell_correction(text)
    return text

def predict_spam(text):
    processed_text = preprocess_text(text)  # ✅ FIXED: Renamed variable to avoid confusion
    arr = vectorizer.transform([processed_text])
    pred = model.predict(arr)
    prob = np.max(model.predict_proba(arr))
    prob = round(prob, 3)
    return pred[0], prob