import json
from nltk_utils import tokenize, stem, bag_of_words


with open('intents.json', 'r') as f:
    intents = json.load(f)
    
all_words = []
tags = []
xy = [] #hols tokenized patterns and tags

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        sentence = tokenize(pattern)
        all_words.extend(sentence)
        xy.append((sentence, tag))
        
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(all_words)
print(tags)



