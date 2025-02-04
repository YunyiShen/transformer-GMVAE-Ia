"""
---------------------------------------------------------------------
-- some useful layers by Jhosimar George Arias Figueroa
---------------------------------------------------------------------
"""
import torch
from torch import nn
from torch.nn import functional as F


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class MultiHeadAttentionLayer_relative(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x



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



"""
---------------------------------------------------------------------
-- ############## our transformers ##############
---------------------------------------------------------------------
"""

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                                dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
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
                 phase_embd_dim,
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
        self.phasefc = nn.Linear(phase_embd_dim, bottleneck_dim) # expand phase to bottleneck
        self.contextfc = nn.Linear(bottleneck_dim, flux_embd_dim + wavelength_embd_dim ) # expand bottleneck to flux and wavelength
    def forward(self, wavelength_embd, phase_embd,bottleneck, mask=None):
        x = torch.cat([self.init_flux_embd.init_flux_embd[None, :, :], wavelength_embd], dim=-1)
        h = x
        bottleneck = torch.cat([bottleneck, self.phasefc(phase_embd)], dim=1) # concate phase embd to bottleneck, in sequence dimension
        bottleneck = self.contextfc(bottleneck)
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, bottleneck, mask=mask)
        return x + h # residual connection

# this will generate bottleneck, in encoder
class bottleneckTransformerModel(nn.Module):
    def __init__(self, bottleneck_length,
                 flux_embd_dim, 
                 wavelength_embd_dim,
                 phase_embd_dim,
                 num_heads, 
                 num_layers,
                 bottleneck_dim,
                 ff_dim, dropout=0.1):
        super(bottleneckTransformerModel, self).__init__()
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, flux_embd_dim + wavelength_embd_dim))
        self.bottleneckfc = nn.Linear(flux_embd_dim + wavelength_embd_dim, bottleneck_dim)
        self.phasefc = nn.Linear(phase_embd_dim, flux_embd_dim + wavelength_embd_dim) # expand phase to bottleneck
        self.transformerblocks =  nn.ModuleList( [TransformerBlock(flux_embd_dim + wavelength_embd_dim, 
                                                    num_heads, ff_dim, dropout) 
                                                 for _ in range(num_layers)] )
    def forward(self, wavelength_embd, flux_embd, phase_embd, mask=None):
        flux = torch.cat([flux_embd, wavelength_embd], dim=-1)
        flux = torch.cat([flux, self.phasefc(phase_embd)], dim=1) # concate phase embd to flux, in sequence
        if mask is not None:
           # add a false at end to account for the added phase embd
           mask = torch.cat([mask, torch.zeros(mask.shape[0], 1).bool()], dim=1)
        x = self.initbottleneck[None, :, :]
        h = x
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, flux, key_padding_mask=mask)
        return self.bottleneckfc(x+h) # residual connection
        



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
