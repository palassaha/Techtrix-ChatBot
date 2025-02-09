import json
from nltk_utils import tokenize, lemmatize, bag_of_words
import torch
import numpy as np
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] #hols tokenized patterns and tags

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        filtered_words = [lemmatize(word) for word in w if word not in stop_words]
        all_words.extend(w)
        xy.append((filtered_words, tag))

all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(all_words)
print(tags)



