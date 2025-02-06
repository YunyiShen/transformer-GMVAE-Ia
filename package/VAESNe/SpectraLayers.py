import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import * # useful base layers


"""
---------------------------------------------------------------------
-- ############## our transformers ##############
---------------------------------------------------------------------
"""

# this will generate flux, in decoder
class spectraTransformerDecoder(nn.Module):
    def __init__(self, spectra_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout=0.1):
        super(spectraTransformerDecoder, self).__init__()
        self.init_flux_embd = nn.Parameter(torch.randn(spectra_length, model_dim))
        self.transformerblocks = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout) 
                                                    for _ in range(num_layers)] 
                                                )
        self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)
        self.phase_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)
        self.contextfc = MLP(bottleneck_dim, model_dim, [model_dim]) # expand bottleneck to flux and time
        #self.get_photo = singlelayerMLP(model_dim, 1) # expand bottleneck to flux and wavelength
        self.get_flux = singlelayerMLP(model_dim, 1)
    
    def forward(self, wavelength, phase, bottleneck, mask=None):
        '''
        wavelength: real wavelength of the spectra being taken (batch_size, spectra_length)
        phase: phase of the spectra being taken (batch_size, 1)
        bottleneck: bottleneck from the encoder (batch_size, bottleneck_length, bottleneck_dim)
        '''
        wavelength_embd = self.wavelength_embd_layer(wavelength)
        phase_embd = self.phase_embd_layer(phase[:, None])
        x =  self.init_flux_embd[None,:,:] + wavelength_embd #+ phase_embd
        h = x
        bottleneck = self.contextfc(bottleneck)
        bottleneck = torch.concat([bottleneck, phase_embd], dim=1)
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, bottleneck, mask=mask)
        return self.get_flux(x + h).squeeze(-1) # residual connection

# this will generate bottleneck, in encoder
class spectraTransformerEncoder(nn.Module):
    def __init__(self, bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, dropout=0.1):
        super(spectraTransformerEncoder, self).__init__()
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        
        self.phase_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand phase to bottleneck
        self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand wavelength to bottleneck
        self.flux_embd = nn.Linear(1, model_dim)
        self.transformerblocks =  nn.ModuleList( [TransformerBlock(model_dim, 
                                                    num_heads, ff_dim, dropout) 
                                                 for _ in range(num_layers)] )
        
        self.bottleneckfc = singlelayerMLP(model_dim, bottleneck_dim)

    def forward(self, wavelength, flux, phase, mask=None):
        
        flux_embd = self.flux_embd(flux[:, :, None]) + self.wavelength_embd_layer(wavelength)
        phase_embd = self.phase_embd_layer(phase[:, None])
        context = torch.cat([flux_embd, phase_embd], dim=1) # concatenate flux and phase embd
        if mask is not None:
           # add a false at end to account for the added phase embd
           mask = torch.cat([mask, torch.zeros(mask.shape[0], 1).bool().to(mask.device) ], dim=1)
        x = self.initbottleneck[None, :, :]
        x = x.repeat(context.shape[0], 1, 1)
        h = x
        
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, context, context_mask=mask)
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
