using PyCall
using Plots

optim = pyimport("torch.optim")
torch = pyimport_conda("torch", "pytorch")
dist  = pyimport("torch.distributions")
nn    = pyimport("torch.nn")
utils = pyimport("torch.nn.utils")
F     = pyimport("torch.nn.functional")
plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")

w0 = 0.1
b0 = 4.5
x_range = [-25, 55]

function load_dataset(n=150, n_tst=150)
    np.random.seed(123)

    function s(x)
        temp = (x .- x_range[1]) ./ (x_range[2] - x_range[1])
        return -3. * (0.25 .+ temp.^2.)
    end

    x = (x_range[2] - x_range[1]) .* np.random.rand(n) .+ x_range[1]
    eps = np.random.randn(n) .* s(x)
    z = (w0 .* -x .* (1. .+ np.cos(x)) .+ b0) + eps
    z = (z .- np.mean(z)) ./ np.std(z)
    idx = np.argsort(x)
    x = x[idx.+1]
    z = z[idx.+1]
    x = np.reshape(x, (n,1))
    z = np.reshape(z, (n,1))
    return z, x
end

z, x = load_dataset()

Z = torch.tensor(z, dtype=torch.float)
X = torch.tensor(x, dtype=torch.float)

@pydef mutable struct VI <: nn.Module
    
    function __init__(self)
        nn.Module.__init__(self)

        self.q_mu = nn.Sequential(
            nn.Linear(1, 25),
            nn.LeakyReLU(),
            nn.Linear(25, 25),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(25, 1)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(1, 25),
            nn.LeakyReLU(),
            nn.Linear(25, 25),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(25, 1)
        )
    end
    
    function reparameterize(self, mu, log_var)
        # log variance to avoid neg std cases
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    end
    
    function forward(self, x)
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var
    end
end


function ll_gaussian(z, mu, log_var)
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma^2) - (1 / (2 * sigma^2))* (z-mu)^2
    end

 function elbo(z_pred, z, mu, log_var)
    # likelihood z given variational mu and sigma
    likelihood = ll_gaussian(z, mu, log_var)
    
    # prior probability of z_pred
    log_prior = ll_gaussian(z_pred, 0, torch.log(torch.tensor(1.)))
    
    # variational probability of z_pred
    log_p_q = ll_gaussian(z_pred, mu, log_var)
    
    # approximate the expectation
    return (likelihood + log_prior - log_p_q).mean()
end


function det_loss(z_pred, z, mu, log_var)
    return -elbo(z_pred, z, mu, log_var)
    end

epochs = 1000

m = VI()
optim = torch.optim.Adam(m.parameters(), lr=0.005)

for epoch=1:epochs
    optim.zero_grad()
    z_pred, mu, log_var = m(X)
    loss = det_loss(z_pred, Z, mu, log_var)
    loss.backward()
    optim.step()
    end


# draw samples from q(theta)
z_pred = torch.cat([m(X)[1] for i=1:1000], dim=1)
    
mu = np.quantile(z_pred.detach().numpy(), 0.5, axis=1)
q25 = np.quantile(z_pred.detach().numpy(), 0.05, axis=1)
q95 = np.quantile(z_pred.detach().numpy(), 0.95, axis=1)

plt.figure(figsize=(15, 5))
plt.scatter(X, Z)
plt.plot(X, mu)
plt.fill_between(X.flatten(), q25, q95, alpha=0.2)
plt.xlabel("X")
plt.ylabel("Z")
plt.savefig("au.png")

@pydef mutable struct LinearVariational <: nn.Module

    function __init__(self, in_features, out_features, parent, n_batches, bias= true)
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias        
        self.parent = parent
        self.n_batches = n_batches
            
        # initialize variational parameters
        # q(w)=N(μ,σ2)
        self.w_mu = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(mean=0, std=0.002)
        )

        self.w_p = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(mean=-2.5, std=0.002)
        )
        if self.include_bias
            self.b_mu = nn.Parameter(
                torch.zeros(out_features)
            )

            self.b_p = nn.Parameter(
                torch.zeros(out_features)
            )
        end
    end
            
    function reparameterize(self, mu, p)
        sigma = torch.log(1 + torch.exp(p)) 
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma)
    end
    
    function kl_divergence(self, z, mu_theta, p_theta, prior_sd=1)
        log_prior = torch.distributions.Normal(0, prior_sd).log_prob(z) 
        log_p_q = torch.distributions.Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z) 
        return (log_p_q - log_prior).sum() / self.n_batches
    end

    function forward(self, x)
        w = self.reparameterize(self.w_mu, self.w_p)
        
        if self.include_bias
            b = self.reparameterize(self.b_mu, self.b_p)
        else
            b = 0
        end

        z = torch.mm(x, w) + b

        self.parent.accumulated_kl_div += self.kl_divergence(w, self.w_mu,self.w_p)
        if self.include_bias
            self.parent.accumulated_kl_div += self.kl_divergence(b, self.b_mu, self.b_p)
        end
        return z
    end
    end

@pydef mutable struct KL
    function __init__(self, accumulated_kl_div = 0)
        self.accumulated_kl_div = accumulated_kl_div
    end
end
    
@pydef mutable struct Model <: nn.Module
        
    function __init__(self, in_size, hidden_size, out_size, n_batches)
        nn.Module.__init__(self)
        self.kl_loss = KL(0)
        
        self.layers = nn.Sequential(
            LinearVariational(in_size, hidden_size, self.kl_loss, n_batches),
            nn.LeakyReLU(),
            LinearVariational(hidden_size, hidden_size, self.kl_loss, n_batches),
            nn.LeakyReLU(),
            LinearVariational(hidden_size, out_size, self.kl_loss, n_batches)
        )
    end
    

    function accumulated_kl_div(self)
        return self.kl_loss.accumulated_kl_div
    end
    
    function reset_kl_div(self)
        self.kl_loss.accumulated_kl_div = 0
    end
            
    function forward(self, x)
        out = self.layers(x)
        return out
    end
    end



epochs = 1000

function det_loss2(z, z_pred, model)
    batch_size = z.shape[1]
    reconstruction_error = -torch.distributions.Normal(z_pred, .1).log_prob(z).sum()
    kl = model.accumulated_kl_div()
    model.reset_kl_div()
    return reconstruction_error + kl
    end

m2 = Model(1, 25, 1, 1)
optim = torch.optim.Adam(m2.parameters(), lr=0.01)


for epoch=1:epochs
    optim.zero_grad()
    z_pred2 = m2(X)
    loss2 = det_loss2(z_pred2, Z, m2)
    loss2.backward()
    optim.step()
    end

transposed = np.transpose(np.array([m2(X).flatten().detach().numpy() for i=1:1000]))
q_25 = np.quantile(transposed, 0.05, axis=1)
q_95 = np.quantile(transposed, 0.95, axis=1)

plt.figure(figsize=(15, 5))
plt.plot(X.detach().numpy(), np.mean(transposed, axis = 1))
plt.scatter(X.detach().numpy(), Z.detach().numpy())
plt.fill_between(X.flatten().detach().numpy(), q_25, q_95, alpha=0.2)
plt.xlabel("X")
plt.ylabel("Z")
plt.savefig("eu.png")