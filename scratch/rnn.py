import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Hyperparameters
input_size = 28  # MNIST images are 28x28 pixels
hidden_size = 64
num_classes = 10
num_epochs = 1
batch_size = 100
learning_rate = 0.001

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, dim_input: int, dim_hidden: int, dim_output: int, num_layers: int):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(dim_input, dim_hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        out, _ = self.rnn(x, None)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

# Load the MNIST dataset
# todo: create a Dataset for the calibration data, i.e., measurement bitstrings sampled from different values of phi
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# todo: create a Dataloader for providing N samples from the same ground truth phase value
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleRNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28, 28)  # Reshape input for RNN
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 28, 28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {100 * correct / total}%')
