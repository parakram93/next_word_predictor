import torch
from nltk.tokenize import word_tokenize
import pickle
import torch.nn as nn
import numpy as np
from prompt_toolkit import PromptSession


# ------------------------------
# Load vocab and max_len
# ------------------------------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Reverse vocab for predictions
index_to_word = {v:k for k,v in vocab.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Define LSTMModel (same as main)
# ------------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (final_hidden_state, _) = self.lstm(embedded)
        output = self.fc(final_hidden_state.squeeze(0))
        return output

# ------------------------------
# Load trained model
# ------------------------------
model = LSTMModel(len(vocab)).to(device)
model.load_state_dict(torch.load("next_word.pth", map_location=device))
model.eval()

# ------------------------------
# Helper function: text -> indices
# ------------------------------
def text_to_indces(sentence, vocab):
    numerical_sentence = []
    for token in sentence:
        if token in vocab:
            numerical_sentence.append(vocab[token])
        else:
            numerical_sentence.append(vocab['<unk>'])
    return numerical_sentence

# ------------------------------
# Function to predict next word
# ------------------------------

def prediction(model,vocab,text):
  tokenized_text = word_tokenize(text.lower())
  numerical_text = text_to_indces(tokenized_text,vocab)
  padded_text = torch.tensor([0] * (max_len - len(numerical_text)) + numerical_text, dtype=torch.long).unsqueeze(0).to(device)
  output = model(padded_text)
  value, index = torch.max(output, dim=1)
  return index_to_word[index.item()]

# ------------------------------
# Interactive loop
# ------------------------------
print("=== Next Word Prediction Ready ===")
print("Type 'exit' to quit\n")

while True:
    text = input("You: ").strip()
    if text.lower() == "exit":
        break
    suggestion = prediction(model,vocab,text)
    print("Suggestion:", suggestion)
