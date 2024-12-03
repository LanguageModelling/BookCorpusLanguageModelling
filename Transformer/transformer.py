import torch
import torch.nn as nn
from torch.nn import functional as F
from Data.Library import Library # Importing the Library class
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 200
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 27
n_head = 1
n_layer = 1
dropout = 0.2
grad_clip = 1.0  # added gradient clipping
# ------------

torch.manual_seed(1337)

# Initialize the Library class and load the BookCorpus dataset
library = Library()
train_dataloader = library.get_train_dataloader(batch_size)
test_dataloader = library.get_test_dataloader(batch_size)
vocab_size = len(library.stoi)
stoi = library.stoi
itos = library.itos

# Train and test splits
train_data_iter = iter(train_dataloader)
val_data_iter = iter(test_dataloader)

# Data loading
def get_batch(split):
    global train_data_iter, val_data_iter
    if split == 'train':
        try:
            x, y = next(train_data_iter)
        except StopIteration:
            train_data_iter = iter(train_dataloader)
            x, y = next(train_data_iter)
    else:
        try:
            x, y = next(val_data_iter)
        except StopIteration:
            val_data_iter = iter(test_dataloader)
            x, y = next(val_data_iter)
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input of size (batch, time-step, channels)
        # Output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1)  # (B, T, T)
        wei = wei * (1.0 / k.shape[-1]**0.5)  # Scaled dot-product
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Mask out future tokens
        wei = F.softmax(wei, dim=-1)  # Softmax over T
        wei = self.dropout(wei)
        # Weighted sum of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate heads
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ A simple feed-forward neural network """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # forward the GPT model
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# Initialize the model
model = SimpleTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss and optimize
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # Gradient clipping
    optimizer.step()
