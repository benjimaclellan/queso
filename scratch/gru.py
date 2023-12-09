#%%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#%%
n_qubits = 4
n_shots = 100
n_grid = 20
n_phases = 20

n_batch = n_grid
n_seq = n_shots
n_feature = n_qubits
n_hidden = n_phases
num_layers = 10

#%%
rnn = nn.GRU(
    input_size=n_feature,
    hidden_size=n_hidden,
    num_layers=num_layers,
    batch_first=True,
    dropout=0.1,
)


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
print(shots)
print(shots.shape)

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.01)

torch.manual_seed(1234)
h0 = torch.randn(num_layers, n_batch, n_hidden)
# h0 = torch.ones(size=(num_layers, n_batch, n_hidden))

#%%
# n_seq = 12
# n_batch = np.floor(n_shots / n_seq).astype('int')
# shots_batch = shots[:, torch.randperm(shots.shape[1]), :]
# shots_batch1 = shots_batch[:, :n_seq * n_batch, :].reshape(n_grid, n_batch, n_seq, n_qubits)
# shots_batch2 = torch.swapaxes(shots_batch1, 0, 1)
# shots_batch3 = shots_batch2.reshape(n_batch, n_grid * n_)
# print(shots_batch2.shape)
# output, hn = rnn(shots_batch2, h0)

#%%
for n_seq in (1, 2, 5,):
    for epoch in range(150):
        # todo: sample a sequence of length n_seq from base dataset
        # shots = torch.randint(
        #     low=0,
        #     high=2,
        #     size=[n_batch, n_seq, n_feature]
        # ).type(torch.float32)

        shots_batch = shots[:, torch.randperm(shots.shape[1]), :]
        # print(shots_batch[-5, :, :])

        # shots_batch = shots[:, :n_seq, :]

        # output, hn = rnn(shots_batch, h0)
        output, hn = rnn(shots_batch)
        # print(output.shape, hn.shape)
        loss = criterion(hn[-1, :, :], labels)

        # torch.nn.functional.cross_entropy(hn[-1, :, :], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}], Loss: {loss.item()}')

#%%
n_seq_test = 20
index = 10
with torch.no_grad():
    # h0 = torch.randn(num_layers, 1, n_hidden)
    # h0 = torch.zeros(size=(num_layers, 1, n_hidden))
    shots_batch = shots[index:index+1, torch.randperm(shots.shape[1])[:n_seq_test], :]
    output, hn = rnn(shots_batch) #, h0[:, index:index+1, :])
    # print(shots_batch.shape, hn[-1, :, :].shape)
    print(shots_batch)
    # print(shots_batch[:])

    prob = torch.softmax(hn[-1, :, :], dim=-1)
    print(f"argmax {torch.argmax(prob)}")
    fig, axs = plt.subplots(1, 1)
    # axs.plot(prob[index, :])
    axs.plot(prob.detach().numpy().squeeze())
    fig.show()
    print(prob)

    # todo: calculate variance and bias
