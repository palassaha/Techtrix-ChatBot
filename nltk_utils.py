import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download("stopwords")
# nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def tokenize(sentence):
    sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence)
    return word_tokenize(sentence)

def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [lemmatize(w) for w in tokenized_sentence if w not in stop_words]  # Remove stopwords & lemmatize
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
