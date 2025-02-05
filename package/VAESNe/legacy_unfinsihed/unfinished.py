import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from networks.SpectraLayers import spectraTransformerDecoder, spectraTransformerEncoder
from networks.util_layers import MLP

#GMVAE
# reduce dimension
class SpectraGMInferenceNet(nn.Module):
    def __init__(self, 
                 latent_length,
                 latent_dim,
                 num_classes,
                 model_dim, 
                num_heads, 
                num_layers,
                ff_dim, 
                dropout=0.1, 
                class_net_hidden = [64, 64]):
        super(SpectraGMInferenceNet, self).__init__()

        # q(y|x) and q(z|y,x) before GumbelSoftmax and Gaussian
        self.inference_transformer = spectraTransformerEncoder(latent_length, 
                 2 * latent_dim, 
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, dropout)
        self.latent_dim = latent_dim
        self.latent_length = latent_length
        self.logitsnet = MLP(2 * latent_dim * latent_length, 
                             num_classes, class_net_hidden)

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
class SpectraGMGenerativeNet(nn.Module):
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
    

    def forward(self, wavelength, z, y, mask = None):
        wavelength_embded = self.wavelength_embd(wavelength)
        # p(z|y)
        y_mu, y_var = self.pzy(y)
        
        # p(x|z)
        x_rec = self.pxz(wavelength_embded, z, mask)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output


class SpectraGMVAENet(nn.Module):
    def __init__(self, spectra_length,
                 flux_embd_dim, 
                wavelength_embd_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                bottleneck_dim,
                num_classes,
                dropout=0.1):
        super(SpectraGMVAENet, self).__init__()

        self.inference = SpectraInferenceNet(flux_embd_dim, 
                wavelength_embd_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                bottleneck_dim,
                num_classes,
                dropout)
        self.generative = SpectraGenerativeNet(spectra_length,
                 flux_embd_dim, 
                wavelength_embd_dim, 
                num_heads, 
                ff_dim, 
                num_layers,
                bottleneck_dim,
                num_classes,
                dropout)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0) 

    def forward(self, flux, wavelength, mask = None, temperature=1.0, hard=0):
        #x = x.view(x.size(0), -1)
        out_inf = self.inference(flux, wavelength, mask, temperature, hard)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative(wavelength, z, y, mask)
        
        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
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

