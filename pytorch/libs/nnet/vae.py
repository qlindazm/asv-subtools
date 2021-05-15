import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU

def matrix_diag(x, inverse=False):
    # 拉平，找各batch对角对应的indices，对这个indices赋值
    # easy to inverse
    N = x.shape[-1]
    shape = x.shape[:-1] + (N, N)
    device, dtype = x.device, x.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = x if inverse==False else 1./x 
    return result

class Multi_fc_relu(nn.Module):
    def __init__(self, input_dim = 512, num_layers = 2, num_units = 512):
        super(Multi_fc_relu, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_units = num_units

        hidden_dims = [num_units for i in range(num_layers+1)]
        hidden_dims[0] = input_dim

        self.layers = []
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LeakyReLU() # for there is no bn
                )
            )
        # self.layers = nn.Sequential(*self.layers)
        self.V = nn.Parameter(torch.randn(num_layers + 1))

    def forward(self, x):
        result = x * self.V[0]
        for i in range(self.num_layers):
            x = self.layers[i](x)
            result += x * self.V[i+1]
        return result


class VAE(nn.Module):
    def __init__(self, input_dim = 512, num_layers = 2, num_units = 512):
        super(VAE, self).__init__()

        # self.fc1 = nn.Linear(784, 400)
        # self.fc21 = nn.Linear(400, 20) 
        # self.fc22 = nn.Linear(400, 20)
        # self.fc3 = nn.Linear(20, 400)
        # self.fc4 = nn.Linear(400, 784)
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_units = num_units
        
        self.encoder_y = Multi_fc_relu(input_dim, num_layers, num_units)
        self.fc_mu_encoder_y = nn.Linear(num_units, input_dim)
        self.fc_log_sigma2_encoder_y = nn.Linear(num_units, input_dim)     # log(sigma^2)
        self.fc_R_encoder_y = nn.Linear(num_units, input_dim * input_dim) # torch.triu(view(input_dim, input_dim), diagonal=1) + torch.eye(input_dim)

        self.encoder_z = Multi_fc_relu(input_dim*2, num_layers, num_units)
        self.fc_mu_encoder_z = nn.Linear(num_units, input_dim)
        self.fc_log_sigma2_encoder_z = nn.Linear(num_units, input_dim)     # log(sigma^2)
        self.fc_R_encoder_z = nn.Linear(num_units, input_dim * input_dim) # torch.triu(view(input_dim, input_dim), diagonal=1) + torch.eye(input_dim)

        self.decoder = Multi_fc_relu(input_dim*2, num_layers, num_units)
        self.fc_mu_decoder = nn.Linear(num_units, input_dim)
        self.fc_log_sigma2_decoder = nn.Linear(num_units, input_dim)     # log(sigma^2)
        self.fc_R_decoder = nn.Linear(num_units, input_dim * input_dim) # torch.triu(view(input_dim, input_dim), diagonal=1) + torch.eye(input_dim)

        


    def encode(self, x):
        # x.shape = (B, d)
        # h1 = F.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        y = self.encoder_y(x)
        mu_y = self.fc_mu_encoder_y(y)   # shape=(B, d)
        log_sigma2_y = self.fc_log_sigma2_encoder_y(y)
        R_y = torch.triu(self.fc_R_encoder_y(y).view(-1, self.input_dim, self.input_dim), diagonal=1) + torch.eye(self.input_dim)

        # pooling
        var_y_inv = torch.bmm(R_y.T, torch.bmm(matrix_diag(torch.exp(log_sigma2_y), True), R_y)) # shape=(B, d, d)
        var_y_pool = torch.add(torch.sum(var_y_inv, 0), (1 - x.shape[0]) * torch.eye(self.input_dim)).inverse()  # shape=(d, d)
        mu_y_pool = var_y_pool.mm(torch.sum(torch.bmm(var_y_inv, mu_y.unsqueeze(-1)), 0))  #shape=(d,1)
        # samp

        z = self.encoder_z(torch.cat((x, y_sample), dim=))
        mu_z = self.fc_mu_encoder_z(z)
        log_sigma2_z = self.fc_log_sigma2_encoder_z(z)
        R_z = torch.triu(self.fc_R_encoder_z(z).view(-1, self.input_dim), diagonal=1) + torch.eye(self.input_dim)

        

    def reparameterize(self, mu, logvar):
        if self.training:
            # std = torch.exp(torch.mm(R.T, 0.5 * log_sigma2))
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, y, z):
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        x = self.decoder(torch.cat((y, z), dim=))
        mu_x = self.fc_mu_decoder(x)
        log_sigma2_x = self.fc_log_sigma2_decoder(x)
        R_x = torch.triu(self.fc_R_decoder(x).view(-1, self.input_dim), diagonal=1) + torch.eye(self.input_dim)

    def forward(self, x):
        # mu, logvar = self.encode(x.view(-1, 784))
        # z = self.reparameterize(mu, logvar)
        # return self.decode(z), mu, logvar