import json
import torch
import random
import numpy as np
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

# Load JSON data
with open("techfest_intents.json", "r") as f:
    data = json.load(f)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_data = torch.load("trained_model.pth", map_location=device)

input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data["all_words"]
tags = model_data["tags"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_data["model_state"])
model.eval()

THRESHOLD = 0.7 

def get_event_response(tag):
    for event in data["events"]:
        if event["event"] == tag:
            return {
                "category": event["category"],
                "description": event["description"],
                "rules": event["rules"],
                "date_time": event["date_time"],
                "venue": event["venue"],
                "coordinator": event["coordinator"]
            }
    return None

def chatbot_response(user_input):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float().to(device)

    with torch.no_grad():
    #     output = model(X)
    #     _, predicted = torch.max(output, dim=0)
    #     tag = tags[predicted.item()]

    # # Fetch event details
    # event_info = get_event_response(tag)
        output = model(X)
        probs = torch.softmax(output, dim=0)  # Convert raw scores to probabilities
        confidence, predicted = torch.max(probs, dim=0)
    
    if confidence.item() < THRESHOLD:
        # return "I'm not sure. Could you clarify?"
        return "Ami dhyamna-choda bannerjee"
        

    tag = tags[predicted.item()]
    event_info = get_event_response(tag)

    if event_info:
        if "when" in user_input.lower() or "time" in user_input.lower():
            return f"The {tag} event is scheduled on {event_info['date_time']}."
        elif "where" in user_input.lower() or "venue" in user_input.lower():
            return f"The {tag} event will take place at {event_info['venue']}."
        elif "who" in user_input.lower() or "coordinator" in user_input.lower():
            return f"The coordinator for {tag} is {event_info['coordinator']['name']}, contact: {event_info['coordinator']['contact']}."
        elif "rules" in user_input.lower():
            return f"The rules for {tag} are:\n- " + "\n- ".join(event_info["rules"])
        else:
            return f"{tag}: {event_info['description']}"
    
    return "Sorry, I didn't understand that. Could you ask in a different way?"

# def chatbot_response(user_input):
#     print(f"User Input: {user_input}")  # Debugging

#     sentence = tokenize(user_input)
#     print(f"Tokenized Sentence: {sentence}")  # Debugging

#     X = bag_of_words(sentence, all_words)
#     print(f"Bag of Words: {X}")  # Debugging

#     X = torch.from_numpy(X).to(device)
#     output = model(X)
#     print(f"Model Output: {output}")  # Debugging

#     _, predicted = torch.max(output, dim=0)
#     print(f"Predicted Class Index: {predicted}")  # Debugging

#     tag = tags[predicted.item()]
#     print(f"Predicted Tag: {tag}")  # Debugging

#     for intent in data["events"]:
#         if intent["event"].lower() == tag.lower():
#             return f"Coordinator: {intent['coordinator']['name']}, Contact: {intent['coordinator']['contact']}"

#     return "Sorry, I didn't understand that."


# Start Chatbot
print("TechFest Chatbot is running! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
