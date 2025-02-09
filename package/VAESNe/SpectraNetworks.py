import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from .SpectraLayers import spectraTransformerDecoder, spectraTransformerEncoder

'''
Adaptations of GMVAE for spectra
'''

class vanillaSpectraInferenceNet(nn.Module):
    def __init__(self, 
                 latent_length,
                 latent_dim,
                 model_dim, 
                num_heads, 
                num_layers,
                ff_dim, 
                dropout=0.1):
        super(vanillaSpectraInferenceNet, self).__init__()

        # q(y|x) and q(z|y,x) before GumbelSoftmax and Gaussian
        self.inference_transformer = spectraTransformerEncoder(
                2 * latent_length,
                 latent_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, dropout)
        self.latent_dim = latent_dim
        self.latent_length = latent_length

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z    

    def forward(self, flux, wavelength, phase, mask = None):
        bottleneck = self.inference_transformer(flux, 
                                                wavelength, 
                                                phase,
                                                mask)

        
        # q(z|x,y)
        mu = bottleneck[:,:self.latent_length,:]
        var = F.softplus( bottleneck[:,self.latent_length:,:])
        z = self.reparameterize(mu, var)

        output = {'mean': mu, 'var': var, 'gaussian': z}
        return output

class vanillaSpectraGenerativeNet(nn.Module):
    def __init__(self, spectra_length,
                 latent_dim,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout=0.1):
        super(vanillaSpectraGenerativeNet, self).__init__()

        # p(x|z)
        self.generativetransformer = spectraTransformerDecoder(
                spectra_length,
                 latent_dim,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout)

    
    # p(x|z)
    def pxz(self, wavelength, phase, z, mask = None):
        flux = self.generativetransformer(wavelength, phase, z, mask)
        return flux

    def forward(self, wavelength, phase,z, mask = None):
        x_rec = self.pxz(wavelength, phase, z, mask)

        output = {'reconstruction': x_rec}
        #breakpoint()
        return output

class vanillaSpectraVAENet(nn.Module):
    def __init__(self, spectra_length,
                latent_length,
                latent_dim,
                model_dim, 
                num_heads, 
                num_layers,
                ff_dim, 
                dropout=0.1):
        super(vanillaSpectraVAENet, self).__init__()

        self.inference = vanillaSpectraInferenceNet(latent_length,
                 latent_dim,
                 model_dim, 
                num_heads, 
                num_layers,
                ff_dim, 
                dropout)
        self.generative = vanillaSpectraGenerativeNet(spectra_length,
                 latent_dim,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0) 

    def encode(self, flux, wavelength, phase, mask = None):
        return self.inference(flux, wavelength, phase, mask)
    
    def decode(self, wavelength, phase, z, mask = None):
        return self.generative(wavelength, phase, z, mask)
    
    def reconstruct(self, flux, wavelength, phase, mask = None):
        out_inf = self.inference(flux, wavelength, phase, mask)
        z = out_inf['gaussian']
        out_gen = self.generative(wavelength, phase, z, mask)
        return out_gen['reconstruction']

    def forward(self, flux, wavelength, phase, mask = None):
        #x = x.view(x.size(0), -1)
        out_inf = self.inference(flux, wavelength, phase, mask)
        z = out_inf['gaussian']
        out_gen = self.generative(wavelength, phase, z, mask)
        
        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output
