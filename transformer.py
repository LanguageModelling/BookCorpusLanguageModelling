import torch
import torch.nn as nn
from torch.nn import functional as F
from Data.Library import Library  # Assuming you have the Library class
import numpy as np

# Hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 200
eval_interval = 5
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)
eval_iters = 200
n_embd = 27
n_head = 1
n_layer = 1
dropout = 0.2
grad_clip = 1.0  # added gradient clipping

# Dataset and Library setup
encoding = 72
train_size = 2**20
library = Library(encoding=encoding, train_size=train_size, streaming=False)
print("created object")
# Initialize Model
class Head(nn.Module):
    # One head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out


class MultiHeadAttention(nn.Module):
    # Multiple heads of self-attention
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    # A simple linear layer followed by a non-linearity
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
    # Transformer block: communication followed by computation
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


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(library.encoding.max_token_value, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, library.encoding.max_token_value)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(log_probs, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPTLanguageModel().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss(model, library):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        dataloader = library.get_train_dataloader(batch_size) if split == 'train' else library.get_val_dataloader(batch_size)
        for k, data in enumerate(dataloader):
            if k >= eval_iters:
                break
            xb, yb = data[:-1], data[1:]
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# Training Loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, library)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Load batch data using Library's DataLoader
    dataloader = library.get_train_dataloader(batch_size)
    print(dataloader)
    print("-----------------------")
    for idx, data in enumerate(dataloader):
        xb, yb = data[:-1], data[1:]

        logits, loss = model(xb.to(device), yb.to(device))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
    print("one interation")

    # Perplexity Evaluation
    if iter % eval_interval == 0:
        perplexity = library.calc_perplexity(model)
        print(f"Perplexity at step {iter}: {perplexity:.4f}")

# Generate output from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(library.encoding.decode(generated[0].tolist()))
