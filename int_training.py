import torch
import random
import torch.nn.functional as F
from v2 import GPT
from copy import deepcopy
# hyperparameters
batch_size = 8 # how many independent sequences are sampled 
block_size = 32 # what is the maximum context length for predictions
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # make code device agnostic
eval_iters = 200
n_embd = 128
n_head = 8
n_layer = 5
dropout = 0.2
# -------------

vocab = [str(i) for i in range(0,10)]
vocab.append('+')
vocab.append('=')
vocab.append('.')
stoi = {s: i for i,s in enumerate(vocab)}
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos) 
stop_idx = stoi['.']

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def get_ints(context_length, n=32, r= 9999):
  x, y = [], []

  for _ in range(n):
    sample = []
    label = []
    while len(sample) < context_length+1:
        # Create ints for a training example
        a = random.randint(0, r)
        b = random.randint(0,r)

        ans = f'{a+b}'[::-1] + '.'
        equation = f'{a}+{b}='
        equation += ans
        non_ans = len(equation) - len(ans) # Portion that is non answer
        equation = encode(equation)
        sample += equation # Adds encoded data to non answer
        label.extend([-1] * (non_ans-1) + equation[non_ans:]) 

    x.append(sample[:context_length])
    y.append(label[:context_length])

  x = torch.tensor(x).to(device)
  y = torch.tensor(y).to(device)
  return x, y

# Means we will never call .backward() on this data
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # eval doesn't do anything right now because we have no dropout or batchnorm layers
    for split in ['train', 'val']:
        losses  = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_ints(block_size)
            # print(X[0],Y[0])
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def addition_pred(a, b, block_size):
    equation = f'{a}+{b}='
    padding_size = block_size - len(equation)
    i = len(equation)-1 # Start predicting at the equals sign pos
    equation = encode(equation ) # Use '.' as a placeholder for the unknowns as decoder will never see them
    output = deepcopy(equation)
    out_s = len(output)
    equation = torch.tensor([equation + [stoi['.']] * padding_size])
    while True:
        crop_idx = equation[:, -block_size:]
        logits, loss = model(crop_idx)
        probs = F.softmax(logits[:, i, :], dim=-1)
        idx_next = torch.multinomial(probs, 1)
        i+=1
        if idx_next.item() == stop_idx or i>=block_size:
            break
        output.append(idx_next.item())
        equation[:,i] = idx_next

    output[out_s:] = output[out_s:][::-1] # reverse the output values
    return decode(output)

if __name__ == '__main__':
    a = 10; b =70
    model = GPT(vocab_size, n_embd, block_size, n_layer, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(addition_pred(a, b, block_size))
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_ints(block_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

losses = estimate_loss()
print(f"Final loss: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

for _ in range(32):
    a = random.randint(0,9999)
    b = random.randint(0,9999)
    print(addition_pred(a,b,block_size))