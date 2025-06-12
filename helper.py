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

# Load the pre-trained model and vectorizer
model_path = 'model.joblib'
model = joblib.load(model_path)
vectorizer_path = 'vectorizer.joblib'
vectorizer = joblib.load(vectorizer_path)

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def remove_words_less_than_two_chars(text):
    return ' '.join([word for word in text.split() if len(word) > 2])

def stemming_text(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = remove_stopwords(text)
    text = remove_words_less_than_two_chars(text)
    text = stemming_text(text)
    return text

def predict_spam(text):
    processed_text = preprocess_text(text)  # ✅ Variable renamed to processed_text
    arr = vectorizer.transform([processed_text])
    pred = model.predict(arr)
    prob = np.max(model.predict_proba(arr))
    prob = round(prob, 3)
    return pred[0], prob
