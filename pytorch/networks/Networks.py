import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from networks.Layers import *


"""
---------------------------------------------------------------------
-- Below are taken from code Authored by Jhosimar George Arias Figueroa under MIT license
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""


# Inference Network


class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(InferenceNet, self).__init__()

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList([
                nn.Linear(x_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                GumbelSoftmax(512, y_dim)
        ])

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList([
                nn.Linear(x_dim + y_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                Gaussian(512, z_dim)
        ])

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                #last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)    
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat
    
    def forward(self, x, temperature=1.0, hard=0):
        #x = Flatten(x)

        # q(y|x)
        logits, prob, y = self.qyx(x, temperature, hard)
        
        # q(z|x,y)
        mu, var, z = self.qzxy(x, y)

        output = {'mean': mu, 'var': var, 'gaussian': z, 
                            'logits': logits, 'prob_cat': prob, 'categorical': y}
        return output


# Generative Network
class GenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GenerativeNet, self).__init__()

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList([
                nn.Linear(z_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, x_dim),
                torch.nn.Sigmoid()
        ])

    # p(z|y)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var
    
    # p(x|z)
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)
        
        # p(x|z)
        x_rec = self.pxz(z)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output


# GMVAE Network
class GMVAENet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GMVAENet, self).__init__()

        self.inference = InferenceNet(x_dim, z_dim, y_dim)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0) 

    def forward(self, x, temperature=1.0, hard=0):
        x = x.view(x.size(0), -1)
        out_inf = self.inference(x, temperature, hard)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative(z, y)
        
        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output


'''
Adaptations
'''

class vanillaSpectraInferenceNet(nn.Module):
    def __init__(self, 
                flux_embd_dim, 
                wavelength_embd_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                bottleneck_dim,
                dropout=0.1):
        super(vanillaSpectraInferenceNet, self).__init__()
        self.flux_embd = nn.Linear(1, flux_embd_dim)
        self.wavelength_embd = nn.Linear(1, wavelength_embd_dim)

        # q(y|x) and q(z|y,x) before GumbelSoftmax and Gaussian
        self.inference_transformer = bottleneckTransformerModel(2, # mean, variance, class
                                 flux_embd_dim, 
                                 wavelength_embd_dim, 
                                 num_heads, 
                                 ff_dim, 
                                 num_layers,
                                 bottleneck_dim,
                                 dropout)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z    

    def forward(self, flux, wavelength, mask = None):
        flux_embded = self.flux_embd(flux)
        wavelength_embded = self.wavelength_embd(wavelength)
        bottleneck = self.inference_transformer(flux_embded, 
                                                wavelength_embded, mask)

        
        # q(z|x,y)
        mu = bottleneck[:,0,:]
        var = F.softplus( bottleneck[:,1,:])
        z = self.reparameterize(mu, var)

        output = {'mean': mu, 'var': var, 'gaussian': z}
        return output

class vanillaSpectraGenerativeNet(nn.Module):
    def __init__(self, spectra_length,
                 flux_embd_dim, 
                wavelength_embd_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                bottleneck_dim,
                dropout=0.1):
        super(vanillaSpectraGenerativeNet, self).__init__()
        self.wavelength_embd = nn.Linear(1, wavelength_embd_dim)

        # p(x|z)
        self.generativetransformer = fluxTransformerModel(
                spectra_length,
                flux_embd_dim, 
                wavelength_embd_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                bottleneck_dim,
                dropout)
        self.makeflux = nn.Linear(flux_embd_dim + wavelength_embd_dim, 1)

    
    # p(x|z)
    def pxz(self, wavelength_embded, z, mask = None):
        flux = self.generativetransformer(wavelength_embded, z, mask)
        flux = self.makeflux(flux)
        return flux

    def forward(self, z, y):
        
        # p(x|z)
        x_rec = self.pxz(z)

        output = {'x_rec': x_rec}
        return output



# reduce dimension
class SpectraInferenceNet(nn.Module):
    def __init__(self, 
                flux_embd_dim, 
                wavelength_embd_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                bottleneck_dim,
                num_classes,
                dropout=0.1):
        super(SpectraInferenceNet, self).__init__()
        self.flux_embd = nn.Linear(1, flux_embd_dim)
        self.wavelength_embd = nn.Linear(1, wavelength_embd_dim)

        # q(y|x) and q(z|y,x) before GumbelSoftmax and Gaussian
        self.inference_transformer = bottleneckTransformerModel(3, # mean, variance, class
                                 flux_embd_dim, 
                                 wavelength_embd_dim, 
                                 num_heads, 
                                 ff_dim, 
                                 num_layers,
                                 bottleneck_dim,
                                 dropout)


        self.inference_qyx = torch.nn.ModuleList([
                nn.Linear(bottleneck_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                GumbelSoftmax(512, num_classes)
        ])

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z    


    # q(y|x)
    def qyx(self, class_embd, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                #last layer is gumbel softmax
                class_embd = layer(class_embd, temperature, hard)
            else:
                class_embd = layer(class_embd)
        return class_embd

    # q(z|x,y)
    '''
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)    
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat
'''
    
    def forward(self, flux, wavelength, mask = None, temperature=1.0, hard=0):
        flux_embded = self.flux_embd(flux)
        wavelength_embded = self.wavelength_embd(wavelength)
        bottleneck = self.inference_transformer(flux_embded, 
                                                wavelength_embded, mask)

        # q(y|x)
        logits, prob, y = self.qyx(bottleneck[:,-1,:], temperature, hard) 
        
        # q(z|x,y)
        mu = bottleneck[:,0,:]
        var = F.softplus( bottleneck[:,1,:])
        z = self.reparameterize(mu, var)

        output = {'mean': mu, 'var': var, 'gaussian': z, 
                            'logits': logits, 'prob_cat': prob, 'categorical': y}
        return output


# Generative Network
class SpectraGenerativeNet(nn.Module):
    def __init__(self, spectra_length,
                 flux_embd_dim, 
                wavelength_embd_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                bottleneck_dim,
                num_classes,
                dropout=0.1):
        super(SpectraGenerativeNet, self).__init__()
        super(vanillaSpectraGenerativeNet, self).__init__()
        self.wavelength_embd = nn.Linear(1, wavelength_embd_dim)

        # p(x|z)
        self.generativetransformer = fluxTransformerModel(
                spectra_length,
                flux_embd_dim, 
                wavelength_embd_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                bottleneck_dim,
                dropout)
        self.makeflux = nn.Linear(flux_embd_dim + wavelength_embd_dim, 1)
        self.y_mu = nn.Linear(num_classes, bottleneck_dim)
        self.y_var = nn.Linear(num_classes, bottleneck_dim)

    
    # p(x|z)
    def pxz(self, wavelength_embded, z, mask = None):
        flux = self.generativetransformer(wavelength_embded, z, mask)
        flux = self.makeflux(flux)
        return flux
        # p(z|y)

    # p(z|y)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var
    

    def forward(self, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)
        
        # p(x|z)
        x_rec = self.pxz(z)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output



# vanillaVAE Network
class vanillaVAENet(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(vanillaVAENet, self).__init__()

        self.inference = InferenceNet(x_dim, z_dim)
        self.generative = GenerativeNet(x_dim, z_dim)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0) 

    def forward(self, x, temperature=1.0, hard=0):
        x = x.view(x.size(0), -1)
        out_inf = self.inference(x, temperature, hard)
        z= out_inf 
        out_gen = self.generative(z)
        
        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output

