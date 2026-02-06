import torch
import torch.nn as nn
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from nltk.tokenize import word_tokenize
import pickle
import re

# ------------------------------
# Load vocab and max_len
# ------------------------------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

index_to_word = {v: k for k, v in vocab.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# LSTM model
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
# Helper functions
# ------------------------------
def text_to_indces(sentence, vocab):
    numerical_sentence = []
    for token in sentence:
        if token in vocab:
            numerical_sentence.append(vocab[token])
        else:
            numerical_sentence.append(vocab['<unk>'])
    return numerical_sentence

def predict_next_word(text):
    tokenized = word_tokenize(text.lower())
    numerical = text_to_indces(tokenized, vocab)
    padded = torch.tensor([0]*(max_len - len(numerical)) + numerical, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(padded)
        _, index = torch.max(output, dim=1)
    return index_to_word[index.item()]

# ------------------------------
# Auto-complete
# ------------------------------
class NextWordCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text
        if not text.strip():
            return

        # Only predict next word if the last character is a space
        if text[-1].isspace():
            next_word = predict_next_word(text)
            # We want to append the next word after the space
            yield Completion(next_word, start_position=0)
        else:
            # Do not suggest anything if user is still typing a word
            return




# ------------------------------
# Prompt session
# ------------------------------
session = PromptSession(completer=NextWordCompleter(), complete_while_typing=True)
print("=== Type your sentence. Press TAB to accept suggestion. Type 'exit' to quit ===\n")

while True:
    text = session.prompt("> ")
    if text.strip().lower() == "exit":
        break
