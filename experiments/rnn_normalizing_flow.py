import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import torch.optim as optim


class PlanarFlow(nn.Module):
    def __init__(self, in_features):
        super(PlanarFlow, self).__init__()
        self.in_features = in_features
        self.u = nn.Parameter(torch.randn(1, in_features))
        self.w = nn.Parameter(torch.randn(1, in_features))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        # Transformation function
        f_z = torch.tanh(torch.mm(z, self.w.t()) + self.b)
        # Compute the Jacobian determinant
        psi = (1 - torch.tanh(torch.mm(z, self.w.t()) + self.b)**2) * self.w
        det_J = 1 + torch.mm(psi, self.u.t())
        # Apply the transformation
        z_out = z + self.u * f_z
        return z_out, det_J


class NormalizingFlow(nn.Module):
    def __init__(self, num_flows, in_features):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([PlanarFlow(in_features) for _ in range(num_flows)])

    def forward(self, z):
        log_det_J = 0
        for flow in self.flows:
            z, det_J = flow(z)
            log_det_J += torch.log(det_J.abs() + 1e-6)
        return z, log_det_J


class VAEEstimator(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        ffnn: nn.Module,
        flows: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.ffnn = ffnn
        self.flows = flows

    def forward(self, shots, z):
        _, hn = self.encoder.forward(shots)
        out = self.ffnn(hn)
        sigma, mu = out[0], out[1]


#
# rnn = nn.GRU(
#     input_size=n_feature,
#     hidden_size=n_hidden,
#     num_layers=num_layers,
#     batch_first=True,
#     dropout=0.1,
# )


# Example usage
if __name__ == '__main__':
    # Set random seed for reproducibility
    #%%
    # torch.manual_seed(42)
    n_samples = 1000
    # Define the base distribution (e.g., Gaussian)
    base_distribution = MultivariateNormal(torch.zeros(1), torch.eye(1))

    # Create a normalizing flow model
    num_flows = 4
    flow_model = NormalizingFlow(num_flows, in_features=1)

    # Sample from the base distribution
    z_sample = base_distribution.sample((n_samples,))



    # Forward pass through the normalizing flow
    z_transformed, log_det_J = flow_model(z_sample)

    # Compute the log probability of the transformed samples
    log_prob_transformed = base_distribution.log_prob(z_sample) + log_det_J

    # Compute the loss (negative log likelihood)
    loss = -log_prob_transformed.mean()

    print("Loss:", loss.item())

    def grab(x):
        return x.detach().cpu().numpy()


    def reparameterize(z, mu, sigma):
        return mu + z * sigma

    #%%
    fig, axs = plt.subplots(2, 1)
    axs[0].hist(grab(z_sample), bins=100)
    axs[1].hist(grab(z_transformed), bins=100)
    plt.show()

    #%%
    # Number of training steps
    num_steps = 10000
    mu = 0.0
    sigma = 1.0

    optimizer = optim.Adam(flow_model.parameters(), lr=1e-3)
    # Training loop
    for step in range(num_steps):
        # Sample from the base distribution
        z_sample = base_distribution.sample((1000,))
        z_sample = reparameterize(z_sample, mu, sigma)

        # Forward pass through the normalizing flow
        z_transformed, log_det_J = flow_model(z_sample)

        # Compute the log probability of the transformed samples
        log_prob_transformed = base_distribution.log_prob(z_sample) + log_det_J

        # Compute the loss (negative log likelihood)
        # loss = -log_prob_transformed.mean()
        # target = 2 * z_sample
        target = torch.sin(z_sample) * 4
        loss = torch.pow(z_transformed - target, 2).mean()

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 100 steps
        print(f"Step {step}/{num_steps}, Loss: {loss.item()}")

    #%%
    with torch.no_grad():
        z_samples = base_distribution.sample((1000,))
        z_sample = reparameterize(z_sample, mu, sigma)
        z_transformed, _ = flow_model(z_samples)

    #%%
    # Plot the original and transformed samples
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].hist(z_sample[:, 0], bins=100, label='Original Samples')
    axs[1].hist(z_transformed[:, 0], bins=100, label='Transformed Samples')
    plt.legend()
    plt.show()

    #%%