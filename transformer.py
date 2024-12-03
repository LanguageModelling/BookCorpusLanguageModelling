import torch
import torch.nn as nn
from torch.nn import functional as F
from Data.Library import Library  # Assuming you have the Library class
import numpy as np

# Hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 512  # what is the maximum context length for predictions?
max_iters = 500
eval_interval = 500
learning_rate = 3e-3
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)
eval_iters = 200
n_embd = 27
n_head = 1
n_layer = 1
dropout = 0.2
grad_clip = 10000  # added gradient clipping
seq_length = 512 

# Dataset and Library setup
encoding = 72
train_size = 2**16
test_size = 2**12
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
    def __init__(self, device):
        super().__init__()
        self.vocab_size = library.encoding.max_token_value
        self.device = device
        self.token_embedding_table = nn.Embedding(library.encoding.max_token_value, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]).to(self.device)
        self.ln_f = nn.LayerNorm(n_embd).to(self.device)
        self.lm_head = nn.Linear(n_embd, library.encoding.max_token_value).to(self.device)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02).to(self.device)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        B, T = idx.shape
        
        # print(self.token_embedding_table.device)
        tok_emb = self.token_embedding_table(idx)
        
        
        pos_emb = self.position_embedding_table(torch.arange(T))
        
        # print("token embedding ", pos_emb.shape)
        x = tok_emb + pos_emb
        x = x.to(self.device)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        B, T, C = logits.shape
        # logits = logits.view(B,T, C)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(0,2,1)
        
        


        if targets is None:
            loss = None
            return log_probs.to(torch.device('cpu'))
        
        else:
            targets = targets.to(self.device)
            
            targets = targets.view(B*T)
            
            loss = F.nll_loss(log_probs, targets)

        return log_probs.to(torch.device('cpu')), loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPTLanguageModel(device=device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

# Training Loop
for epoch in range(max_iters):
    # if iter % eval_interval == 0 or iter == max_iters - 1:
    #     losses = estimate_loss(model, library)
    #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Load batch data using Library's DataLoader
    
    dataloader = library.get_train_dataloader(seq_length+1)
    
    
    
    
    xbatch = torch.zeros((batch_size,seq_length))
    ybatch = torch.zeros((batch_size,seq_length))

    for idx, data in enumerate(dataloader):
        mod_idx = idx%batch_size
        # print("jere")

        

        xb, yb = data[:-1], data[1:]
        if(data.shape[0]!=seq_length+1):
            break
        xbatch[mod_idx] = xb
        ybatch[mod_idx] = yb
        


        if(mod_idx==batch_size-1):
            optimizer.zero_grad(set_to_none=True)

            logits = model(xbatch.long())
            loss = loss_fn(logits, ybatch.long())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
            optimizer.step()

   
        

    # Perplexity Evaluation
    
    perplexity = library.calc_perplexity(model)
    print(f"Perplexity at step {epoch}: {perplexity:.4f}")

# Generate output from the model
context = torch.zeros((1, 1), dtype=torch.long)
# generated = model.generate(context, max_new_tokens=500)
# print(library.encoding.decode(generated[0].tolist()))
