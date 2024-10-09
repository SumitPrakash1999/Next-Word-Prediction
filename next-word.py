import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
nltk.download('punkt')

file_path = '/kaggle/input/auguste/Auguste_Maquet.txt'
glove_path = '/kaggle/input/glove6b300dtxt/glove.6B.300d.txt'

def clean_sentence(text):   
    text = re.sub(r'[^a-zA-Z\s]', '', text)    
    text = text.lower()    
    text = re.sub(r'\s+', ' ', text).strip()
    return text


with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

sentences = sent_tokenize(text)
cleaned_sentences = [clean_sentence(sentence) for sentence in sentences]

# Tokenizing each cleaned sentence into words and adding <START> and <END> tokens
words = []
for sentence in cleaned_sentences:
    tokenized_sentence = word_tokenize(sentence)
    words.extend(['<START>'] + tokenized_sentence + ['<END>'])

# Creating a vocabulary with word counts and include special tokens
word_counts = Counter(words)
vocab = {word for word, count in word_counts.items()}
vocab.update(['<PAD>', '<UNK>', '<START>', '<END>']) 

def load_glove_embeddings(glove_path, vocab):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                embedding = np.array(values[1:], dtype='float32')
                embeddings_index[word] = embedding
    return embeddings_index

embeddings_index = load_glove_embeddings(glove_path, vocab)
print(f'Loaded {len(embeddings_index)} word vectors from GloVe.')


print(cleaned_sentences[:1]) #debug           


# Creating the N-gram class
class SixGramDataset(Dataset):
    def __init__(self, words, word_to_idx, embedding_matrix):
        self.words = words
        self.word_to_idx = word_to_idx
        self.embedding_matrix = embedding_matrix
        self.ngrams = self.create_ngrams(words)
        print(f"Total 6-grams created: {len(self.ngrams)}") 

    def create_ngrams(self, words):
        n = 6
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = words[i:i + n]  # Selects a sequence of 6 words
            ngrams.append(ngram)
        return ngrams

    def __len__(self):
        return len(self.ngrams)
    
    def generate_lstm_input(self, idx):
        ngram = self.ngrams[idx]
        input_indices = [self.word_to_idx.get(str(word), self.word_to_idx['<UNK>']) for word in ngram[:5]]
        input_embeddings = np.stack([self.embedding_matrix[idx] for idx in input_indices])
        assert input_embeddings.shape == (5, 300), f"Expected input shape (5, 300), got {input_embeddings.shape}"

        target_index = self.word_to_idx.get(str(ngram[5]), self.word_to_idx['<UNK>'])
        return torch.tensor(input_embeddings, dtype=torch.float32), torch.tensor(target_index, dtype=torch.long)

    def __getitem__(self, idx):
        # General 6-gram handling
        ngram = self.ngrams[idx]
        input_indices = [self.word_to_idx.get(str(word), self.word_to_idx['<UNK>']) for word in ngram[:5]]
        target_index = self.word_to_idx.get(str(ngram[5]), self.word_to_idx['<UNK>'])
        input_vec = np.concatenate([self.embedding_matrix[idx] for idx in input_indices])
        assert input_vec.shape[0] == 1500, f"Expected input vector size of 1500, got {input_vec.shape[0]}"
        return torch.tensor(input_vec, dtype=torch.float32), torch.tensor(target_index, dtype=torch.long)


embedding_dim = 300
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
embedding_matrix = np.random.randn(len(vocab), embedding_dim)  # Random for unknown words

# Initializing embeddings for known words
for word, idx in word_to_idx.items():
    if word in embeddings_index:
        embedding_matrix[idx] = embeddings_index[word]

def prepare_words(sentences):
    words = []
    for sentence in sentences:
        tokenized_sentence = word_tokenize(sentence)
        words.extend(['<START>'] + tokenized_sentence + ['<END>'])
    return words


total_sentences = len(cleaned_sentences)
train_size = int(0.7 * total_sentences)
test_size = int(0.1 * total_sentences)
val_size = total_sentences - train_size - test_size 

train_sentences = cleaned_sentences[:train_size]
val_sentences = cleaned_sentences[train_size:train_size + val_size]
test_sentences = cleaned_sentences[train_size + val_size:]

train_words = prepare_words(train_sentences)
val_words = prepare_words(val_sentences)
test_words = prepare_words(test_sentences)

# Creating datasets
train_dataset = SixGramDataset(train_words, word_to_idx, embedding_matrix)
val_dataset = SixGramDataset(val_words, word_to_idx, embedding_matrix)
test_dataset = SixGramDataset(test_words, word_to_idx, embedding_matrix)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f'Total number of 6-grams in training dataset: {len(train_dataset)}')
print(f'Total number of 6-grams in validation dataset: {len(val_dataset)}')
print(f'Total number of 6-grams in test dataset: {len(test_dataset)}')


#-------------------------debug--------------------------------------------------------------------------------------------------

print("Inspecting first few 6-grams from the dataset:")
num_examples_to_print = 10 

for i in range(num_examples_to_print):
 
    ngram = train_dataset.ngrams[i]
    input_words = ngram[:5]  # First 5 words for input
    target_word = ngram[5]   # 6th word as target

    input_vec, target_idx = train_dataset[i]
    input_word_indices = [train_dataset.word_to_idx.get(word, train_dataset.word_to_idx['<UNK>']) for word in input_words]
    target_word_index = train_dataset.word_to_idx.get(target_word, train_dataset.word_to_idx['<UNK>'])

    input_words_actual = [idx_to_word[idx] for idx in input_word_indices]
    target_word_actual = idx_to_word[target_word_index]

    print(f"6-gram {i + 1}:")
    print(f"  Raw 6-gram: {ngram}")
    print(f"  Input Words (first 5): {input_words_actual}")
    print(f"  Target Word (6th): {target_word_actual}")
    print(f"  Input Vector Shape: {input_vec.shape}")
    print(f"  Target Index: {target_idx}")
    print("-" * 50)

#-------------------------debug--------------------------------------------------------------------------------------------------



#Neural Language Model--------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  

class NeuralLanguageModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1,hidden_dim2,output_dim):
        super(NeuralLanguageModel, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim1)  
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2) 
        self.output = nn.Linear(hidden_dim2, output_dim)  

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x


input_dim = 5 * 300  # 5 words, each with 300-dimensional GloVe embeddings
hidden_dim1 = 300
hidden_dim2 = 300
output_dim = len(word_to_idx)  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralLanguageModel(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.001)



#-------------------------debug--------------------------------------------------------------------------------------------------
# Sample batch iteration
for i, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {i + 1}: Input shape: {inputs.shape}, Target shape: {targets.shape}")
    if i >= 5:  # Only print the first 5 batches to check
        break
#-------------------------debug--------------------------------------------------------------------------------------------------





#og code of train and test
import math
# Function to evaluate the model and calculate perplexity
def evaluate(model, data_loader, criterion):
    model.eval() 
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():  
        for inputs, targets in data_loader:
            inputs = inputs.to(device) 
            targets = targets.to(device)  
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    avg_loss = total_loss / total_count
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity



num_epochs = 20
for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)  
        targets = targets.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward() 
        optimizer.step()  
        running_loss += loss.item() * inputs.size(0)

    avg_training_loss = running_loss / len(train_loader.dataset)


    val_loss, val_perplexity = evaluate(model, val_loader, criterion)
    
#  
#         f.write(f"Epoch {epoch + 1}/{num_epochs}\n")
# #         f.write(f"  Training Loss: {avg_training_loss}, Training Perplexity: {train_perplexity}\n")
#         f.write(f"Validation Perplexity: {val_perplexity}\n")
#         f.flush() 
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_training_loss}, Validation Loss: {val_loss}, Validation Perplexity: {val_perplexity}")

# Test phase: Evaluating the model on the test set
test_loss, test_perplexity = evaluate(model, test_loader, criterion)
# f.write(f"Test Perplexity: {test_perplexity}\n")
# f.write("Model training completed and saved.\n")
print(f"Test Loss: {test_loss}, Test Perplexity: {test_perplexity}")

# Saving the model
torch.save(model.state_dict(), 'neural_language_model.pth')
print("Model training completed and saved.")





#for only txt computation
# Evaluation function to calculate and write batch-wise perplexity
def evaluate(model, data_loader, criterion, device, filepath):
    model.eval()  
    total_loss = 0.0
    total_count = 0
    perplexities = []

    with open(filepath, 'w') as f:
        f.write("Batch No\tPerplexity\n")  
        
        with torch.no_grad():
            for batch_no, (inputs, targets) in enumerate(data_loader, start=1):
                inputs, targets = inputs.to(device), targets.to(device) 
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                batch_loss = loss.item() * inputs.size(0)
                
                total_loss += batch_loss
                total_count += inputs.size(0)
            
                batch_perplexity = math.exp(loss.item())
                perplexities.append(batch_perplexity)
                
                f.write(f"{batch_no}\t{batch_perplexity:.4f}\n")


        avg_loss = total_loss / total_count
        avg_perplexity = math.exp(avg_loss)

        f.write(f"\nAverage Perplexity: {avg_perplexity:.4f}\n")
    
    return avg_loss, avg_perplexity


def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()  
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()  
            optimizer.step()  

            running_loss += loss.item() * inputs.size(0)

        avg_training_loss = running_loss / len(train_loader.dataset)        
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_training_loss:.4f}")

train_model(model, train_loader, criterion, optimizer, device, num_epochs=20)
evaluate(model, val_loader, criterion, device, 'MLP_validation_perplexities.txt')
evaluate(model, test_loader, criterion, device, 'MLP_test_perplexities.txt')




# LSTM model------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# LSTM Model Definition
class LSTMLanguageModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        super(LSTMLanguageModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))



def words_to_indices(padded_sentences, word_to_idx):
    return [[word_to_idx.get(word, word_to_idx["<PAD>"]) for word in sentence] for sentence in padded_sentences]

def sentence_to_glove_embeddings(sentence, glove_embeddings, embedding_dim=300):
    embeddings = []
    for word in sentence:
        if word == "<PAD>":  
            embeddings.append(np.zeros(embedding_dim))
        else:
            embeddings.append(glove_embeddings.get(word, np.zeros(embedding_dim)))  # Use zero vector if word not found
    return embeddings


X = []
y = []  


for sentence in cleaned_sentences:
    words = sentence.split() 
    if len(words) > 1: 
        X.append(words)  
        y.append(words[1:])  

def pad_sequence(seq, max_length, pad_token="<PAD>"):
    if len(seq) < max_length:
        seq.extend([pad_token] * (max_length - len(seq)))  
    return seq[:max_length]  


max_length = 50
X_padded = [pad_sequence(sentence, max_length) for sentence in X]
y_padded = [pad_sequence(sentence, max_length) for sentence in y]


for i in range(1):
    print(f"Input (X): {X_padded[i]}")
    print(f"Output (Y): {y_padded[i]}")



X_embeddings = [sentence_to_glove_embeddings(sentence, embeddings_index) for sentence in X_padded]
def words_to_indices(padded_sentences, word_to_idx):
    return [[word_to_idx.get(word, word_to_idx["<PAD>"]) for word in sentence] for sentence in padded_sentences]


y_indices = words_to_indices(y_padded, word_to_idx)


X_embeddings = np.array(X_embeddings)
y_embeddings = np.array(y_indices)
X_train, X_temp, y_train, y_temp = train_test_split(X_embeddings, y_indices, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Converting to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # GloVe embeddings as input
        self.y = y  # Word indices as output

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dataloaders
batch_size = 64
train_dataset = TextDataset(X_train_tensor, y_train_tensor)
val_dataset = TextDataset(X_val_tensor, y_val_tensor)
test_dataset = TextDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f'Training set size: {len(train_dataset)}')
print(f'Validation set size: {len(val_dataset)}')
print(f'Test set size: {len(test_dataset)}')






#og code for evaluation
def calculate_perplexity(loss):
    return math.exp(loss)

def evaluate_lstm(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            hidden = model.init_hidden(batch_size)
            outputs, hidden = model(inputs, hidden)

            outputs = outputs.view(-1, model.vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    avg_loss = total_loss / total_count
    perplexity = calculate_perplexity(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_count = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            hidden = model.init_hidden(batch_size)

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)

            outputs = outputs.view(-1, model.vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            total_count += batch_size

        avg_training_loss = running_loss / total_count

        val_loss, val_perplexity = evaluate_lstm(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Training Loss: {avg_training_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")


num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 300
vocab_size = len(word_to_idx)


model = LSTMLanguageModel(embedding_dim=300, hidden_dim=300, vocab_size=len(word_to_idx), num_layers=1).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

# Test evaluation
test_loss, test_perplexity = evaluate_lstm(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.4f}")

torch.save(model.state_dict(), 'lstm_language_model.pth')
print("Model training completed and saved.")




#for only txt file
import math
import torch
import numpy as np


def calculate_perplexity(loss):
    return math.exp(loss)


def evaluate_lstm(model, data_loader, criterion, device, filepath):
    model.eval() 
    total_loss = 0.0
    total_count = 0
    perplexities = []

    with open(filepath, 'w') as f:  
        f.write("Batch No\tPerplexity\n") 
        
        with torch.no_grad():  
            for batch_no, (inputs, targets) in enumerate(data_loader, start=1):
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)

                hidden = model.init_hidden(batch_size)  
                outputs, hidden = model(inputs, hidden)

                outputs = outputs.view(-1, model.vocab_size)
                targets = targets.view(-1)

                loss = criterion(outputs, targets)
                batch_loss = loss.item() * batch_size

                total_loss += batch_loss
                total_count += batch_size
                
                batch_perplexity = calculate_perplexity(loss.item())
                perplexities.append(batch_perplexity)
                
                f.write(f"{batch_no}\t{batch_perplexity:.4f}\n")

        avg_loss = total_loss / total_count
        avg_perplexity = calculate_perplexity(avg_loss)
        f.write(f"\nAverage Perplexity: {avg_perplexity:.4f}\n")

    return avg_loss, avg_perplexity


def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        total_count = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            hidden = model.init_hidden(batch_size) 

            optimizer.zero_grad()  
            outputs, hidden = model(inputs, hidden)

            outputs = outputs.view(-1, model.vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()  
            optimizer.step() 

            running_loss += loss.item() * batch_size
            total_count += batch_size

        avg_training_loss = running_loss / total_count
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_training_loss:.4f}")


num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 300
vocab_size = len(word_to_idx)

model = LSTMLanguageModel(embedding_dim=300, hidden_dim=300, vocab_size=vocab_size, num_layers=1).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

evaluate_lstm(model, val_loader, criterion, device, 'LSTM_validation_perplexities.txt')

evaluate_lstm(model, test_loader, criterion, device, 'LSTM_test_perplexities.txt')


# Transformer model------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Transformer-based Language Model
class TransformerLanguageModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=2, num_heads=8, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Positional Encoding (for sentence positions)
        self.pos_embedding = nn.Embedding(300, embedding_dim)  # assuming max length of 500 words
        
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer for vocabulary prediction
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x, memory):
        # Add positional encodings to input
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        x = x + self.pos_embedding(positions)

        # Forward pass through Transformer Decoder
        output = self.transformer_decoder(x, memory)
        output = self.fc_out(output)
        return output

    
    
def words_to_indices(padded_sentences, word_to_idx):
    return [[word_to_idx.get(word, word_to_idx["<PAD>"]) for word in sentence] for sentence in padded_sentences]

X = []
y = []  


for sentence in cleaned_sentences:
    words = sentence.split() 
    if len(words) > 1: 
        X.append(words)  
        y.append(words[1:])  

def pad_sequence(seq, max_length, pad_token="<PAD>"):
    if len(seq) < max_length:
        seq.extend([pad_token] * (max_length - len(seq)))  
    return seq[:max_length]  


max_length = 50
X_padded = [pad_sequence(sentence, max_length) for sentence in X]
y_padded = [pad_sequence(sentence, max_length) for sentence in y]


for i in range(1):
    print(f"Input (X): {X_padded[i]}")
    print(f"Output (Y): {y_padded[i]}")

def sentence_to_glove_embeddings(sentence, glove_embeddings, embedding_dim=300):
    embeddings = []
    for word in sentence:
        if word == "<PAD>":  
            embeddings.append(np.zeros(embedding_dim))
        else:
            embeddings.append(glove_embeddings.get(word, np.zeros(embedding_dim)))  # Use zero vector if word not found
    return embeddings


X_embeddings = [sentence_to_glove_embeddings(sentence, embeddings_index) for sentence in X_padded]
def words_to_indices(padded_sentences, word_to_idx):
    return [[word_to_idx.get(word, word_to_idx["<PAD>"]) for word in sentence] for sentence in padded_sentences]


y_indices = words_to_indices(y_padded, word_to_idx)


X_embeddings = np.array(X_embeddings)
y_embeddings = np.array(y_indices)
X_train, X_temp, y_train, y_temp = train_test_split(X_embeddings, y_indices, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # GloVe embeddings as input
        self.y = y  # Word indices as output

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


batch_size = 64
train_dataset = TextDataset(X_train_tensor, y_train_tensor)
val_dataset = TextDataset(X_val_tensor, y_val_tensor)
test_dataset = TextDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)




#og code
def calculate_perplexity(loss):
    return math.exp(loss)

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, inputs)
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)
    
    avg_loss = total_loss / total_count
    perplexity = calculate_perplexity(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity

def train_transformer_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, inputs)  # Using inputs as memory
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_perplexity = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")

def test_model(model, test_loader, criterion, device):
    test_loss, test_perplexity = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.4f}")

# Model Initialization and Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 300
hidden_dim = 512
vocab_size = len(word_to_idx)
num_layers = 2
num_heads = 6

model = TransformerLanguageModel(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=num_layers, num_heads=num_heads).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


train_transformer_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=2)


test_model(model, test_loader, criterion, device)

torch.save(model.state_dict(), 'transformer_model.pth')
print("Model training completed and saved.")




#only for txt file
import math

def calculate_perplexity(loss):
    return math.exp(loss)

def evaluate_model(model, data_loader, criterion, device, filepath):
    model.eval()
    total_loss = 0
    total_count = 0
    batch_perplexities = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, inputs)  # Inputs as memory
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

            batch_perplexity = calculate_perplexity(loss.item())
            batch_perplexities.append(batch_perplexity)
    
    avg_loss = total_loss / total_count
    avg_perplexity = calculate_perplexity(avg_loss) if avg_loss < 100 else float('inf')

    with open(filepath, 'w') as f:
        f.write("Batch No\tPerplexity\n")
        for idx, perplexity in enumerate(batch_perplexities, 1):
            f.write(f"{idx}\t{perplexity:.4f}\n")
        f.write(f"\nAverage Perplexity: {avg_perplexity:.4f}\n")
    
    return avg_loss, avg_perplexity

def train_transformer_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, inputs) 
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)


        avg_train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_perplexity = evaluate_model(model, val_loader, criterion, device, 'transformer_val_perplexity.txt')
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")

def test_model(model, test_loader, criterion, device):
    test_loss, test_perplexity = evaluate_model(model, test_loader, criterion, device, 'transformer_test_perplexity.txt')
    print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 300
hidden_dim = 512
vocab_size = len(word_to_idx)
num_layers = 2
num_heads = 6

model = TransformerLanguageModel(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=num_layers, num_heads=num_heads).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_transformer_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=3)

test_model(model, test_loader, criterion, device)




