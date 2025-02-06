import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from .PhotometricLayers import photometricTransformerEncoder, photometricTransformerDecoder

class vanillaPhotometricInferenceNet(nn.Module):
    def __init__(self, 
                num_bands,
                latent_len,
                latent_dim,
                model_dim,
                num_heads, 
                ff_dim, 
                num_layers,
                dropout=0.1):
        super(vanillaPhotometricInferenceNet, self).__init__()

        # q(y|x) and q(z|y,x) before GumbelSoftmax and Gaussian
        
        self.inference_transformer = photometricTransformerEncoder(
                                 num_bands, # mean, variance, class
                                 2 * latent_len,
                                 latent_dim,
                                 model_dim, 
                                 num_heads, 
                                 ff_dim, 
                                 num_layers,
                                 dropout)
        '''
        self.inference_transformer_mu = photometricTransformerEncoder(
                                 num_bands, # mean, variance, class
                                 latent_len,
                                 latent_dim,
                                 model_dim, 
                                 num_heads, 
                                 ff_dim, 
                                 num_layers,
                                 dropout)
        self.inference_transformer_sigma = photometricTransformerEncoder(
                                 num_bands, # mean, variance, class
                                 latent_len,
                                 latent_dim,
                                 model_dim, 
                                 num_heads, 
                                 ff_dim, 
                                 num_layers,
                                 dropout)

        '''

        self.latent_dim = latent_dim
        self.latent_len = latent_len
                                 

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z    

    def forward(self, flux, time, band, mask = None):
        
        bottleneck = self.inference_transformer(flux, 
                                                time, 
                                                band,
                                                mask)

        
        # q(z|x,y)
        #breakpoint()
        #mu = bottleneck[:,:,:self.latent_dim] # should it be dimension or should it be length??
        #var = F.softplus( bottleneck[:,:,self.latent_dim:])
        mu = bottleneck[:,:self.latent_len,:]
        var = F.softplus( bottleneck[:,self.latent_len:,:])
        '''
        mu = self.inference_transformer_mu(flux, 
                                            time, 
                                            band,
                                            mask)
        var = F.softplus(self.inference_transformer_sigma(flux,
                                            time,
                                            band,
                                            mask))
                                           
        '''
        z = self.reparameterize(mu, var)

        output = {'mean': mu, 'var': var, 'gaussian': z}
        return output

class vanillaPhotometricGenerativeNet(nn.Module):
    def __init__(self, photometric_length,
                 latent_dim,
                num_bands,
                model_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                dropout=0.1):
        super(vanillaPhotometricGenerativeNet, self).__init__()

        # p(x|z)
        self.generativetransformer = photometricTransformerDecoder(
                photometric_length,
                latent_dim,
                num_bands,
                model_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                dropout)

    
    # p(x|z)
    def pxz(self, time, band, z, mask = None):
        flux = self.generativetransformer(time, band, z, mask)
        return flux

    def forward(self, time, band, z, mask = None):
        x_rec = self.pxz(time, band, z, mask)

        output = {'reconstruction': x_rec}
        return output

class PhotometricVAENet(nn.Module):
    def __init__(self, photometric_length = 60,
                num_bands = 6,
                latent_len = 8,
                latent_dim = 4,
                model_dim = 64, 
                num_heads = 4, 
                ff_dim = 64, 
                num_layers = 2,
                dropout = 0.1):
        super(PhotometricVAENet, self).__init__()

        self.inference = vanillaPhotometricInferenceNet(num_bands,
                latent_len,
                latent_dim,
                model_dim,
                num_heads, 
                ff_dim, 
                num_layers,
                dropout)
        self.generative = vanillaPhotometricGenerativeNet(photometric_length,
                latent_dim,
                num_bands,
                model_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                dropout)
    
    def encode(self, flux, time, band, mask = None):
        return self.inference(flux, time, band, mask)['mean']
    
    def reconstruct(self, flux, time, band, mask = None):
        return self.generative(time, band, 
                               self.inference(flux, time, band, mask)['gaussian'], 
                               mask)
    
    def decode(self, time, band, z, mask = None):
        return self.generative(time, band, z, mask)

    def forward(self, flux, time, band, mask = None):
        #x = x.view(x.size(0), -1)
        out_inf = self.inference(flux, time, band, mask)
        z = out_inf['gaussian']
        out_gen = self.generative(time, band, z,mask)
        
        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output