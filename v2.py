import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 8 # how many independent sequences are sampled 
block_size = 64 # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # make code device agnostic
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 6
dropout = 0.2
# -------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique chracters that occur in the text
chars  =sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# lets now encode the entire dataset and store it into a torch.Tensor
# and make train test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*.9) # first 90% will be train, the rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generates a tensor with batch size number of random ints between 0-len(data)-block_size 
    x = torch.stack([data[i:i+block_size] for i in ix]) # converts 1d tensor to a tensor stack of rows
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

# Means we will never call .backward() on this data
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # eval doesn't do anything right now because we have no dropout or batchnorm layers
    for split in ['train', 'val']:
        losses  = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Tril is not a parameter it is a buffer so we must assign it in pytorch via register buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-.5 # (B,T,C) @ (B,C,T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # Randomly dropout some affinities
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # self attention output
        out = self.dropout(self.proj(out)) # Linear transformation of the outcome of sa
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection layer going back into the residual pathway
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we would like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connections
        x = x + self.ffw(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """
    Model assumes that based on the current character we can predict the next one without any other context
    this is a simple lookup table based on what the current character is for the next character.
    """

    def __init__(self) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token froma  lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # each character in the vocabulary gets 32 tensor embed
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # each block 0-8 gets a 32 tensor 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) batch, time, channels/num embeddings 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B, T, C) this essentially is now the addition of a learned position weight and token embedding
        # At this point x holds the token identities and position they occur
        # this info is position invariant
        x = self.blocks(x) # apply one head of self attention. (B,T,C)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # Preserves the channels and just concats the two dims
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Kind of ridiculous how we are doing this because we only need a single character to predict the next not the whole sequence
        We would like this function to work in the advanced transformer architecture that utilizes the history of the characters
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            # for each batch take the last input and all of its embeddings
            # because we want to predict what comes next
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # take probability over the last dim
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    
model = BigramLanguageModel()
m = model.to(device)

# create the PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    # Crazy how much code is abstracted in these lines
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
