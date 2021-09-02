from langdetect import detect
import random
import json
from nltk import text

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Gaia"
print("Let's chat! (type 'quit' to exit)")
res = ""
while res != "en" and res != "fr" :
    print("Please type fr for frensh or en for english")
    res = input("You : ")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    #res = detect(input(text))
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        if res == "en":
            for intent in intents['intents']:
             if tag == intent["tag"] and intent["lang"] == "en":
                print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            if res == "fr":
                for intent in intents['intents']:
                 if tag == intent["tag"] and intent["lang"] == "fr":
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")