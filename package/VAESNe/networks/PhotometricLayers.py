"""
---------------------------------------------------------------------
-- some useful layers by Jhosimar George Arias Figueroa
---------------------------------------------------------------------
"""
import torch
from torch import nn
from torch.nn import functional as F
from networks.util_layers import *




"""
---------------------------------------------------------------------
-- ############## our transformers ##############
---------------------------------------------------------------------
"""


# this will generate flux, in decoder
class photometricTransformerDecoder(nn.Module):
    def __init__(self, photometry_length,
                 bottleneck_dim,
                 num_bands,
                 model_dim,
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout=0.1):
        super(photometricTransformerDecoder, self).__init__()
        self.init_flux_embd = nn.Parameter(torch.randn(photometry_length, model_dim))
        self.transformerblocks = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout) 
                                                    for _ in range(num_layers)] 
                                                )
        self.model_dim = model_dim
        self.solid_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        self.bandembd = nn.Embedding(num_bands, model_dim)
        self.contextfc = nn.Linear(bottleneck_dim, model_dim) # expand bottleneck to flux and time
        self.get_photo = singlelayerMLP(model_dim, 1)
    
    def forward(self, time, band, bottleneck, mask=None):
        '''
        time: real time of the photometry being taken (batch_size, photometry_length)
        band: band of the photometry being taken (batch_size, photometry_length)

        bottleneck: bottleneck from the encoder (batch_size, bottleneck_length, bottleneck_dim)
        '''
        time_embd = self.solid_embd(time)
        band_embd = self.bandembd(band)
        x = self.init_flux_embd.init_flux_embd[None, :, :] + time_embd + band_embd
        h = x
        bottleneck = self.contextfc(bottleneck)
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, bottleneck, mask=mask)
        x = x + h # residual connection
        return self.get_photo(x).squeeze(-1) # get flux

# this will generate bottleneck, in encoder
class photometricTransformerEncoder(nn.Module):
    def __init__(self,
                 num_bands, 
                 bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 ff_dim,
                 num_layers,
                 dropout=0.1):
        super(photometricTransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.time_embd = SinusoidalPositionalEmbedding(model_dim)
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        self.bottleneckfc = nn.Linear(model_dim, bottleneck_dim)
        self.transformerblocks =  nn.ModuleList( [TransformerBlock(model_dim, 
                                                    num_heads, ff_dim, dropout) 
                                                 for _ in range(num_layers)] )
        
        self.bandembd = nn.Embedding(num_bands, model_dim)
        self.fluxfc = nn.Linear(1, model_dim)


    def forward(self, flux, time, band, mask=None):
        '''
        flux: real flux of the photometry being taken (batch_size, photometry_length)
        time: real time of the photometry being taken (batch_size, photometry_length)
        band: band of the photometry being taken (batch_size, photometry_length)

        '''
        photomety_embdding = (self.fluxfc(flux[:, :, None]) + 
                              self.time_embd(time) + 
                              self.bandembd(band))

        x = self.initbottleneck[None, :, :]
        h = x
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, photomety_embdding, 
                                 key_padding_mask=mask)
        return self.bottleneckfc(x+h) # residual connection
        
