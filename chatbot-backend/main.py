import random
import json
import logging
from flask import Flask, request, jsonify
import torch
from flask_cors import CORS, cross_origin
from model import NeuralNetwork
from nltk_process_data import bag_of_words, tokenize

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

trained_data = "trained_data.pth"
data = torch.load(trained_data)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

PROBABILITY_THRESHOLD = 0.75


@app.route("/chat", methods=["POST"])
@cross_origin()
def chat():
  
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"response": "No message provided."}), 400

        sentence = tokenize(user_message)
        if not sentence:
            return jsonify({"response": "Could not process the message."}), 400

        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > PROBABILITY_THRESHOLD:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    return jsonify({"response": response})
        else:
            return jsonify({"response": "I do not understand..."})
    
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return jsonify({"response": "Internal server error."}), 500

if __name__ == "__main__":
    app.run()
