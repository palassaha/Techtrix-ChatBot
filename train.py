import json
import torch
import numpy as np
from nltk_utils import tokenize, lemmatize, bag_of_words
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

with open("techfest_intents.json", "r") as f:
    data = json.load(f)

all_words = []
tags = []
xy = []

for event in data["events"]:
    tag = event["event"]
    tags.append(tag)

    # Generate training patterns (questions chatbot might receive)
    patterns = [
        f"When is {event['event']}?",
        f"Where is {event['event']} happening?",
        f"Who is the coordinator for {event['event']}?",
        f"Tell me about {event['event']}.",
        f"What are the rules for {event['event']}?"
    ]
    
    for pattern in patterns:
        w = tokenize(pattern)
        filtered_words = [lemmatize(word) for word in w if word not in stop_words]
        all_words.extend(filtered_words)
        xy.append((filtered_words, tag))

# Sort and remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

# Convert patterns into numerical data
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define Dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
BATCH_SIZE = 8
INPUT_SIZE = len(all_words)
HIDDEN_SIZE = 8
OUTPUT_SIZE = len(tags)
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train model
for epoch in range(NUM_EPOCHS):
    for words, labels in train_loader:
        words, labels = words.to(device), labels.to(dtype=torch.long, device=device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

print("Training complete!")

# Save model
torch.save({
    "model_state": model.state_dict(),
    "input_size": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "output_size": OUTPUT_SIZE,
    "all_words": all_words,
    "tags": tags
}, "trained_model.pth")

print("Model saved successfully!")


