import torch
import torch.nn as nn
from torch.nn import functional as F


# class Head(nn.Module):
#     """One head of self attention"""

#     def __init__(self, head_size):
#         super().__init__()
#         self.key = nn.Linear(n_embd, head_size, bias=False)
#         self.query = nn.Linear(n_embd, head_size, bias=False)
#         self.value = nn.Linear(n_embd, head_size, bias=False)
#         # Tril is not a parameter it is a buffer so we must assign it in pytorch via register buffer
#         self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         B, T, C = x.shape
#         k = self.key(x)
#         q = self.query(x)
#         # compute attention scores ("affinities")
#         wei = q @ k.transpose(-2,-1) * C**-.5 # (B,T,C) @ (B,C,T) -> (B, T, T)
#         wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T, T)
#         wei = F.softmax(wei, dim=-1) # (B, T, T)
#         wei = self.dropout(wei) # Randomly dropout some affinities
#         # perform the weighted aggregation of the values
#         v = self.value(x) # (B, T, C)
#         out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
#         return out

# class MultiHeadAttention(nn.Module):
#     """Multiple heads of self-attention in parallel"""

#     def __init__(self, num_heads, head_size):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1) # self attention output
#         out = self.dropout(self.proj(out)) # Linear transformation of the outcome of sa
#         return out
    
class CausalSelfAttention(nn.Module):

    def __init__(self, block_size, n_embd, n_head, dropout):
        super().__init__()
        # Ensure that a valid embeddings are given
        assert n_embd % n_head == 0
        # Create a single linear layer to produce k, q, v
        self.attn = nn.Linear(n_embd, n_embd*3, bias=False)
        # Create a projection vector
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)).view(1, 1, block_size,block_size))
        self.n_head = n_head
        self.n_embd = n_embd
    
    def forward(self, x):
        #print(self.n_embd)
        B, T, C = x.shape
        k, q, v = self.attn(x).split(self.n_embd, dim=2) # Each one is B,T,C
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, he
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, he
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # B, nh, T, he
        wei = q @ k.transpose(-2,-1) * C**-.5 # B, nh, T, T
        wei = wei.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf')) # B, nh, T, T
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v # B, nh, T, T @ B, nh, T, he -> B, nh, T, he
        out = out.transpose(1,2).contiguous().view(B,T,C) # make the head outputs side by side again to reform into C

        # residual output connection
        out = self.proj_dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
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

    def __init__(self, block_size, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we would like
        super().__init__()
        self.sa = CausalSelfAttention(block_size, n_embd, n_head, dropout)
        self.ffw = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connections
        x = x + self.ffw(self.ln2(x))
        return x

class GPT(nn.Module):
    """
    Causal self attention GPT model for predicting the next character. Uses recurrent connections and dropout to regularize
    and improve training speed.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_layer, dropout) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token froma  lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # each character in the vocabulary gets 32 tensor embed
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # each block 0-8 gets a 32 tensor 
        self.blocks = nn.Sequential(*[Block(block_size, n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

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
            
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        We would like this function to work in the advanced transformer architecture that utilizes the history of the characters
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
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

