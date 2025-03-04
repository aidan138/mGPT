import torch
from v2 import GPT

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
chars  = sorted(list(set(text)))
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

if __name__ == '__main__':
    model = GPT(vocab_size, n_embd, block_size, n_layer, dropout)
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