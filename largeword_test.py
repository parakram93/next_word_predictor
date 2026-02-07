# largeword_prompt.py

import torch
import torch.nn as nn
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from nltk.tokenize import word_tokenize
import pickle

# ------------------------------
# Load vocab and max_len
# ------------------------------
with open("vocab_large.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("max_len_large.pkl", "rb") as f:
    max_len = pickle.load(f)

index_to_word = {v: k for k, v in vocab.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# LSTM Model
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
model.load_state_dict(torch.load("next_large_word.pth", map_location=device))
model.eval()

# ------------------------------
# Helper functions
# ------------------------------
def text_to_indices(tokens):
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if len(indices) < max_len:
        indices = [0] * (max_len - len(indices)) + indices
    else:
        indices = indices[-max_len:]
    return indices

def predict_next_word(text):
    # Only predict after a space (i.e., after completing a word)
    if not text.endswith(" "):
        return None
    
    tokens = word_tokenize(text.lower())
    if not tokens:
        return None

    indices = text_to_indices(tokens)
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred_index = torch.max(output, dim=1)

    return index_to_word[pred_index.item()]

# ------------------------------
# PromptSession completer
# ------------------------------
class NextWordCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text
        suggestion = predict_next_word(text)
        if suggestion:
            # Start position 0 because we append suggestion at the cursor
            yield Completion(suggestion, start_position=0)

# ------------------------------
# Interactive session
# ------------------------------
if __name__ == "__main__":
    session = PromptSession(completer=NextWordCompleter(), complete_while_typing=True)
    print("=== Large Next Word Auto-complete ===")
    print("Type your sentence. Press TAB to accept suggestion. Type 'exit' to quit.\n")

    while True:
        text = session.prompt("> ")
        if text.strip().lower() == "exit":
            break
