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
    pass

# s1 = "Chat Bot With PyTorch - NLP And Deep Learning - Python Tutorial (Part 1)"
# print(s1)
# s1 = tokenize(s1)
# print(s1)