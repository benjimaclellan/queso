#%%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#%%
n_qubits = 6
n_shots = 1000
n_grid = 50
n_phases = 50

n_batch = n_grid
n_seq = n_shots
n_feature = n_qubits
n_hidden = n_phases
num_layers = 3

#%%
rnn = nn.GRU(
    input_size=n_feature,
    hidden_size=n_hidden,
    num_layers=num_layers,
    batch_first=True,
    dropout=0.1,
)
n_trainable_params = sum([p.numel() for p in rnn.parameters()])
print(f"Number of parameters: {n_trainable_params}")


#%% create minimal dataset for GHZ state
def sample(phase: float, n_samples: int):
    p = np.array([np.cos(n_qubits * phase) ** 2, np.sin(n_qubits * phase) ** 2])
    p = p / np.sum(p)
    print(p, sum(p))
    logical = np.random.choice(
        [0, 1],
        n_samples,
        p=p
    )
    s = np.tile(logical, [n_qubits, 1])
    return s.T


labels = torch.arange(n_phases)
phases = 0 + labels/n_phases * np.pi/2/n_qubits
shots = torch.Tensor(np.array(list(map(lambda phase: sample(phase=phase, n_samples=n_shots), phases)))).type(torch.float32)
print(shots.shape)

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.001)

torch.manual_seed(1234)

#%%
for n_seq in (5,):
    for epoch in range(550):
        shots_batch = shots[:, torch.randperm(shots.shape[1]), :]
        output, hn = rnn(shots_batch)
        loss = criterion(hn[-1, :, :], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}], Loss: {loss.item()}')

#%%
n_seq_test = 6
index = 5
with torch.no_grad():
    shots_batch = shots[index:index+1, torch.randperm(shots.shape[1])[:n_seq_test], :]
    print(shots_batch)
    output, hn = rnn(shots_batch)
    # prob_test = torch.softmax(torch.prod(output, dim=1).squeeze(), dim=0)
    # prob = prob_test
    prob = torch.softmax(hn[-1, :, :], dim=-1)
    print(f"argmax {torch.argmax(prob)}")
    fig, axs = plt.subplots(1, 1)
    axs.plot(prob.detach().numpy().squeeze())
    plt.show()
    print(prob)

    # todo: calculate variance and bias
