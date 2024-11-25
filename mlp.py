#mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from Library import Library

class MLP(nn.Module):
    def __init__(self, vocab_size, n_gram, hidden_size, num_layers, device):
        super(MLP, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.n_gram = n_gram

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size).to(self.device)

        # Define the fully connected layers
        self.fc_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = hidden_size * n_gram if i == 0 else hidden_size * n_gram
            output_size = hidden_size * n_gram
            self.fc_layers.append(nn.Linear(input_size, output_size))

        # Output layer
        self.output_layer = nn.Linear(hidden_size * n_gram, vocab_size)

    def forward(self, x):
        x = x.to(self.device)
        # Shape: [batch_size, seq_length, n_gram]
        x = torch.flatten(self.embedding(x), 2)
        #print(f"Shape after embedding: {x.shape}")
        
        for layer in self.fc_layers:
            x = F.relu(layer(x))  # Apply the fully connected layers with ReLU
        x = self.output_layer(x)
        return F.log_softmax(x, dim=-1).to('cpu').permute(0, 2, 1)


# Hyperparameters
epochs = 16
lr = 0.0001
seq_length = 512
batch_size = 32
n_gram = 1  
hidden_size = 256
num_layers = 2
train_size = 10000

# Setup
device = torch.device('mps') 
print(f"Using device: {device}")
library = Library(encoding=27, train_size=train_size, streaming=False)
print(f"Dataset size: {len(library.dataset)}")
dataloader = library.get_train_dataloader(seq_length + 1)
print(f"Number of batches in train dataloader: {len(dataloader)}")

model = MLP(
    vocab_size=library.encoding.max_token_value,
    n_gram=n_gram,
    hidden_size=hidden_size,
    num_layers=num_layers,
    device=device
).to(device)

loss_fn = nn.NLLLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

x_batch = torch.zeros([batch_size, seq_length - n_gram + 1, n_gram])
y_batch = torch.zeros([batch_size, seq_length - n_gram + 1])
losses = torch.zeros(epochs)
perplexities = torch.zeros(epochs)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    dataloader = library.get_train_dataloader(seq_length + 1)
    print(f"Epoch {epoch + 1}: Checking dataloader...")
    for batch in dataloader:
        print(batch)  # Ensure data is being yielded
        break
    for idx, data in enumerate(dataloader):
        mod_idx = idx % batch_size

        # Generate n-grams
        ngrams = library.ngramify(data[:-1], n=n_gram)  # Shape: [num_ngrams, n_gram]

        # Pad ngrams to match [511, 2] if needed
        if ngrams.shape[0] < seq_length - n_gram + 1:  # Target size: [511, 2]
            padding_size = seq_length - n_gram + 1 - ngrams.shape[0]
            ngrams = F.pad(ngrams, (0, 0, 0, padding_size))  # Pad to target size

        # Assign to batch
        x_batch[mod_idx] = ngrams
        target = data[n_gram:]
        if target.shape[0] < seq_length - n_gram + 1:
            padding_size = seq_length - n_gram + 1 - target.shape[0]
            target = F.pad(target, (0, padding_size))
        y_batch[mod_idx] = target
        
        # Process the batch when it's full
        if mod_idx == batch_size - 1:
            # Update weights
            optim.zero_grad()
            y_pred = model(x_batch.long())
            loss = loss_fn(y_pred, y_batch.long())
            total_loss += loss.item()
            loss.backward()
            #print(f"Batch {idx}: Loss = {loss.item():.4f}")
            optim.step()

    num_batches = idx + 1 if idx else 1  # Count batches processed
    avg_loss = total_loss / num_batches
    losses[epoch] = avg_loss
    perplexities[epoch] = library.calc_perplexity(model)#, n=n_gram)
    print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Perplexity: {perplexities[epoch]:.4f}')

# Check generated tokens
sample_input = x_batch[0]  # First batch sample
generated_tokens = model(sample_input.unsqueeze(0).long()).argmax(dim=-1)
print("Generated tokens:", generated_tokens)

########
#Epoch 16/16 - Loss: 0.0700, Perplexity: 10.5483




