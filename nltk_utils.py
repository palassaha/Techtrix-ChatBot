import nltk
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')
# nltk.download('punkt_tab')

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

# s1 = "Chat Bot With PyTorch - NLP And Deep Learning - Python Tutorial (Part 1)"
# print(s1)
# s1 = tokenize(s1)
# print(s1)