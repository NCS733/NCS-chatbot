import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load resources
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

# Chat loop
print("Chatbot is ready! Type 'quit' to exit")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    print("Bot:", response)
