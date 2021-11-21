from utils import *
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        # Set hyperparameters
        self.hidden_size = hidden_size
        # Create layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        # Return zero tensor for initial hidden tensor
        return torch.zeros(1, self.hidden_size)

    def forward(self, inp, hid):
        # Combine input tensor and hidden tensor
        combined = torch.cat((inp, hid), 1)
        # Compute new hidden tensor
        hid = self.i2h(combined)
        # Compute new output tensor and apply softmax
        out = self.i2o(combined)
        out = self.softmax(out)
        return out, hid

num_hidden = 128
rnn = RNN(num_letters, num_hidden, num_categories)
lr = 0.005
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr)

def train(y, X):
    hidden = rnn.init_hidden()
    rnn.zero_grad()
    # Pass each character through the RNN
    for i in range(X.size()[0]):
        output, hidden = rnn(X[i], hidden)
    # Compute loss, gradients, and take a step with optimizer
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # Return output tensor and loss
    return output, loss.item()

def category_from_output(output):
    # Gets top output category
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    # Returns category as string
    return all_categories[category_i], category_i

epochs = 5
print_every = 5000

Xs, ys, ls, cs = load_all()

for epoch in range(epochs):
    for i, X in enumerate(Xs):
        output, loss = train(ys[i], X)
        category = cs[i]
        line = ls[i]

        if i % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = 'CORRECT' if guess == category else f'WRONG (ans: {category})'
            print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss}, Name: {line}, Guess: {guess} --> {correct}')