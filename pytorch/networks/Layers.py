"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Custom Layers

"""
import torch
from torch import nn
from torch.nn import functional as F

# Flatten layer
class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)

# Reshape layer    
class Reshape(nn.Module):
  def __init__(self, outer_shape):
    super(Reshape, self).__init__()
    self.outer_shape = outer_shape
  def forward(self, x):
    return x.view(x.size(0), *self.outer_shape)

# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, c_dim):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(f_dim, c_dim)
    self.f_dim = f_dim
    self.c_dim = c_dim
     
  def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
      U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard 
  
  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y

# Sample from a Gaussian distribution
class Gaussian(nn.Module):
  def __init__(self, in_dim, z_dim):
    super(Gaussian, self).__init__()
    self.mu = nn.Linear(in_dim, z_dim)
    self.var = nn.Linear(in_dim, z_dim)

  def reparameterize(self, mu, var):
    std = torch.sqrt(var + 1e-10)
    noise = torch.randn_like(std)
    z = mu + noise * std
    return z      

  def forward(self, x):
    mu = self.mu(x)
    var = F.softplus(self.var(x))
    z = self.reparameterize(mu, var)
    return mu, var, z 


############## transformers ##############


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                                dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        # we made x [batch, seq_len, embed_dim]
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.layernorm1(x + self.dropout(attn_output))

        # Cross-attention (if context is provided)
        if context is not None:
            cross_attn_output, _ = self.cross_attn(x, context, context, attn_mask=mask)
            x = self.layernorm2(x + self.dropout(cross_attn_output))

        # Feedforward
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout(ffn_output))

        return x


# this will generate flux, in decoder
class fluxTransformerModel(nn.Module):
    def __init__(self, spectra_length,
                 flux_embd_dim, 
                 wavelength_embd_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 bottleneck_dim,
                 dropout=0.1):
        super(fluxTransformerModel, self).__init__()
        self.init_flux_embd = nn.Parameter(torch.randn(spectra_length, flux_embd_dim))
        self.transformerblocks = nn.ModuleList( [TransformerBlock(flux_embd_dim + wavelength_embd_dim, 
                                                 num_heads, ff_dim, dropout) 
                                                    for _ in range(num_layers)] 
                                                )
        self.contextfc = nn.Linear(bottleneck_dim, flux_embd_dim + wavelength_embd_dim ) # expand bottleneck to flux and wavelength
    def forward(self, wavelength_embd, bottleneck, mask=None):
        x = torch.cat([self.init_flux_embd.init_flux_embd[None, :, :], wavelength_embd], dim=-1)
        bottleneck = self.contextfc(bottleneck)
        for transformerblock in self.transformerblocks:
            x = transformerblock(x, bottleneck, mask=mask)
        return x

# this will generate bottleneck, in encoder
class bottleneckTransformerModel(nn.Module):
    def __init__(self, bottleneck_length,
                 flux_embd_dim, 
                 wavelength_embd_dim,
                 num_heads, 
                 num_layers,
                 bottleneck_dim,
                 ff_dim, dropout=0.1):
        super(bottleneckTransformerModel, self).__init__()
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, flux_embd_dim + wavelength_embd_dim))
        self.bottleneckfc = nn.Linear(flux_embd_dim + wavelength_embd_dim, bottleneck_dim)
        self.transformerblocks =  nn.ModuleList( [TransformerBlock(flux_embd_dim + wavelength_embd_dim, 
                                                    num_heads, ff_dim, dropout) 
                                                 for _ in range(num_layers)] )
    def forward(self, wavelength_embd, flux_embd, mask=None):
        flux = torch.cat([flux_embd, wavelength_embd], dim=-1)
        x = self.initbottleneck[None, :, :]
        for transformerblock in self.transformerblocks:
            x = transformerblock(x, flux, key_padding_mask=mask)
        return self.bottleneckfc(x)
        



class TransformerModel(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x, context=None):
        for layer in self.layers:
            x = layer(x, context)
        return x
