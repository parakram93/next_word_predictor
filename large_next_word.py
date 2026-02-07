import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk
import pickle
from datasets import load_dataset
import string
from datasets import load_dataset

document= """
Welcome to the general knowledge guide.
People often ask questions about daily life. What should I eat for breakfast? Where can I find fresh vegetables? How do I manage my time effectively? Giving clear answers is important.

Stories are a fun way to learn. Once upon a time, there was a little fox who lived in a forest. He was curious and wanted to explore new places. Every day, he met other animals and learned something new.

Traveling is exciting. You can visit mountains, beaches, cities, and villages. When you travel, it is helpful to know where the nearest hotel is, what food is popular, and which language people speak. Give yourself enough time to enjoy every destination.

Cooking is another essential skill. Recipes often ask you to add ingredients like salt, sugar, or butter. What temperature should the oven be? How long should you bake the cake? If you follow the instructions carefully, the dish turns out delicious.

Daily routines are also common topics. People ask: What time should I wake up? How can I exercise regularly? Where should I go for a morning walk? Giving small steps makes it easier to form habits.

Technology is everywhere. Smartphones, laptops, and smart devices help us communicate. When using technology, people often ask: How do I connect to Wi-Fi? What apps should I install? Give careful attention to privacy settings.

Education is important. Students often ask: What subjects are necessary? How should I study efficiently? Where can I find extra resources? Teachers guide students and give feedback on their progress.

Games and entertainment are popular too. People ask: What games are fun? Where can I watch movies? How do I play chess well? Giving tips and strategies makes games enjoyable.

Health and wellness are key. Questions like: What is a balanced diet? Where can I find a good gym? How do I stay motivated? People give advice to maintain a healthy lifestyle.

Shopping is part of daily life. What should I buy for groceries? Where can I get discounts? How do I compare products? Giving recommendations is helpful for buyers.

Nature and science spark curiosity. Why does the sun rise in the east? What causes rain? Where do birds migrate? Giving clear explanations helps people understand the world.

Art and creativity inspire people. How do I draw better? What colors should I use? Where can I exhibit my paintings? Giving constructive feedback is useful.

Social interactions matter. How should I greet someone? What questions are polite? Where should I sit in a meeting? Giving attention to manners improves relationships.

Travel stories mix curiosity and fun. What happened when I went hiking? Where did I see the most beautiful sunset? How did I learn new skills during the trip? Sharing stories encourages others to explore.

Pets and animals are beloved. What should I feed my dog? Where should I place a birdcage? How do I train a cat? Giving care tips ensures happy pets.

Festivals and events bring people together. What should I prepare for Diwali? Where is the New Year celebration happening? How can I join a local festival? Giving details helps everyone plan.

Historical facts are educational. Who was Alexander the Great? What caused the industrial revolution? Where did ancient civilizations live? Giving precise information enhances learning.

Science experiments are exciting. What chemicals react? How do I measure accurately? Where should I perform the experiment safely? Giving step-by-step guidance is essential.

Music and dance are joyful. What songs are trending? Where can I learn classical dance? How do I improve my singing? Giving encouragement improves skills.

Gardening teaches patience. What seeds should I plant? Where do I put the garden? How do I water plants correctly? Giving proper advice helps plants thrive.

Life tips are practical. What habits improve productivity? Where should I focus my energy? How do I deal with stress? Giving simple suggestions improves daily life.

Random facts and trivia. Did you know that honey never spoils? Where is the tallest building? What is the fastest animal on earth? Giving curious facts entertains readers.

Jokes and humor lighten the mood. Why did the chicken cross the road? What is the funniest movie scene? How do people laugh differently around the world? Giving a smile makes the day better.

Motivation and self-improvement. What are the goals to set? How do I stay disciplined? Where do I find inspiration? Giving positive reinforcement helps growth.

Daily conversations. Hello, how are you? What did you do today? Where are you going? Giving polite responses makes communication smooth.

Travel tips. What is the best way to pack? Where should I book tickets? How do I avoid long queues? Giving practical suggestions saves time.

Weather and seasons. What is the temperature today? Where will it rain tomorrow? How do I prepare for winter? Giving accurate information helps everyone plan.

Random knowledge expansion. What is the capital of France? Where is the Sahara Desert located? How do volcanoes erupt? Giving clear answers improves curiosity.

Shopping tips and advice. What is the best smartphone? Where can I get bargains? How do I check product quality? Giving guidance helps buyers make smart decisions.

Food and cooking experiences. What is the secret to perfect pasta? Where should I buy fresh vegetables? How do I preserve fruits? Giving detailed instructions helps cook efficiently.

Technology usage guidance. What apps improve productivity? Where do I find tutorials? How do I fix software issues? Giving recommendations enhances efficiency.

Sports and hobbies. What is the best training for running? Where can I join a club? How do I improve my skills? Giving tips motivates enthusiasts.

Storytelling exercises. What happened when I visited the mountains? Where did I meet interesting people? How do I describe scenes vividly? Giving engaging details inspires readers.

Pet care advice. What is the healthiest diet for pets? Where should I keep a fish tank? How do I train a puppy? Giving proper care keeps pets happy.

Health and fitness guidance. What exercises target the core? Where can I find healthy recipes? How do I stay consistent? Giving routines improves fitness.

Learning and skill development. What language should I learn next? Where do I find courses? How do I practice efficiently? Giving structured guidance improves learning.

Everyday life tips. What is the best way to organize a room? Where should I put furniture? How do I clean efficiently? Giving simple methods saves time.

Fun facts and trivia. What is the fastest car? Where is the largest library? How do penguins survive in cold climates? Giving interesting facts entertains readers.

This concludes the multi-topic general dataset example. The text above contains a mixture of questions, instructions, facts, stories, daily life tips, technical explanations, and casual conversation. 
It is around 12,000 words and ideal for training a next-word predictor or autocomplete model.
"""


tokens = word_tokenize(document.lower())

tokens = [t for t in tokens if t not in string.punctuation]

save_path = "next_large_word.pth"
nltk.download("punkt")

# ------------------------------
# Tokenize & build vocab
# ------------------------------

vocab = {'<pad>':0, '<unk>':1}

#Counter(token).keys() #provides info of how many tme same word has arrived in a data. .keys() filters uniques words form the tokens

for token in Counter(tokens).keys():
    if token not in vocab:
        vocab[token] = len(vocab)

# ------------------------------
# Prepare training sequences
# ------------------------------
input_sentences = document.split('\n')

def text_to_indces(sentence, vocab):
    numerical_sentence = []
    for token in sentence:
        if token in vocab:
            numerical_sentence.append(vocab[token])
        else:
            numerical_sentence.append(vocab['<unk>'])
    return numerical_sentence

input_numerical_sentences = []
for sentence in input_sentences:
    tkn = word_tokenize(sentence.lower())
    tkn = [t for t in tkn if t not in string.punctuation]
    input_numerical_sentences.append(text_to_indces(tkn, vocab))

training_sequence = []
for sentence in input_numerical_sentences:
    for i in range(1, len(sentence)):
        training_sequence.append(sentence[:i+1])
max_len = max(len(seq) for seq in training_sequence)
 # last token is the target

# ------------------------------
# Dataset
# ------------------------------
class CustomDataset(Dataset):
    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        padded = [0]*(self.max_len - len(seq)) + seq if len(seq) < self.max_len else seq[:self.max_len]
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[-1], dtype=torch.long)
        return x, y

        


data = CustomDataset(training_sequence, max_len=max_len)

dataloader = DataLoader(data, batch_size=32, shuffle=True)

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

model = LSTMModel(len(vocab))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------
# Training
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 50

for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ------------------------------
# Save model and vocab
# ------------------------------
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

with open("vocab_large.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("max_len_large.pkl", "wb") as f:
    pickle.dump(max_len, f)
