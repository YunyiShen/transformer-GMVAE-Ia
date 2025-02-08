import torch
import numpy as np
from torch import nn, optim
from .losses import VAEloss
from .Metrics import *
import matplotlib.pyplot as plt
import math
import gc


def safelog10(x):
    tmp = max(1e-10, x)
    return math.log10(tmp)


def training_step(network, optimizer, data_loader, 
                  loss_fn = VAEloss(beta = 1.), release_memory = False):
    """Train the model for one epoch

    Args:
        network: (nn.Module) the neural network
        optimizer: (Optim) optimizer to use in backpropagation
        data_loader: (DataLoader) corresponding loader containing the training data
        loss_fn: (function) loss function to use

    Returns:
        average of all loss values, accuracy, nmi
    """
    network.train()
    total_loss = 0.
    num_batches = 0.
    rec_loss = 0.
    kl_loss = 0.
    device = next(network.parameters()).device
    for (flux, time, band, mask) in data_loader: # flux, time, band, mask for photometry and flux, wavelength, phase, mask for spectra
        optimizer.zero_grad()
        flux = flux.to(device)
        time = time.to(device)
        band = band.to(device)
        mask = mask.to(device)
        out_net = network(flux, 
                          time, 
                          band, 
                          mask)
        
        if release_memory:
            del time, band
            gc.collect()
            torch.cuda.empty_cache()

        rec_loss, kl_loss = loss_fn(flux, 
                                    out_net['reconstruction'], 
                                    out_net['mean'],
                                    out_net['var'],
                                    mask)
        loss = rec_loss + kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().item()
        rec_loss += rec_loss.detach().cpu().item()
        kl_loss += kl_loss.detach().cpu().item()
        num_batches += 1.
        if release_memory:
            del out_net, loss, flux, mask # Free memory
            gc.collect()
            torch.cuda.empty_cache()

    return [rec_loss / num_batches, kl_loss / num_batches]


def validation_step(network, data_loader, loss_fn = VAEloss(beta = 1.), release_memory = False):
    """validate the model for one epoch

    Args:
        network: (nn.Module) the neural network
        optimizer: (Optim) optimizer to use in backpropagation
        data_loader: (DataLoader) corresponding loader containing the training data
        loss_fn: (function) loss function to use

    Returns:
        average of all loss values, accuracy, nmi
    """
    network.eval()
    device = next(network.parameters()).device
    total_loss = 0.
    num_batches = 0.
    rec_loss = 0.
    kl_loss = 0.
    for (flux, time, band, mask) in data_loader:
        flux = flux.to(device)
        time = time.to(device)
        band = band.to(device)
        mask = mask.to(device)
        out_net = network(flux, 
                          time, 
                          band, 
                          mask)
        if release_memory:
            del time, band
            gc.collect()
            torch.cuda.empty_cache()

        rec_loss, kl_loss = loss_fn(flux, 
                                    out_net['reconstruction'], 
                                    out_net['mean'],
                                    out_net['var'],
                                    mask)
        loss = rec_loss + kl_loss
        total_loss += loss.detach().cpu().item()
        rec_loss += rec_loss.detach().cpu().item()
        kl_loss += kl_loss.detach().cpu().item()
        if release_memory:
            del out_net, loss, flux, mask # Free memory
            gc.collect()
            torch.cuda.empty_cache()

        num_batches += 1.
    return [rec_loss / num_batches, kl_loss / num_batches]



def train(network, 
          train_loader, 
          val_loader,
          learning_rate=1e-3,
          num_epochs=100,
          loss_fn = None,
          device = 'cuda' if torch.cuda.is_available() else 'cpu',
          release_memory = False
          ):
    """Train the model

    Args:
        train_loader: (DataLoader) corresponding loader containing the training data
        val_loader: (DataLoader) corresponding loader containing the validation data

    Returns:
        output: (dict) contains the history of train/val loss
    """
    network.to(device)
    if loss_fn is None:
        loss_fn = VAEloss(beta = 1.).to(device)
    optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
    train_history_lost, val_history_loss = [], []
    for epoch in range(1, num_epochs + 1):
        train_loss = training_step(network, optimizer, train_loader, loss_fn, release_memory)
        with torch.no_grad():
            val_loss = validation_step(network, val_loader, loss_fn, release_memory)
        train_history_lost.append(train_loss)
        val_history_loss.append(val_loss)
        print('(Epoch %d / %d losses) Train_recon: %.3lf; Train_kl: %.3lf; Val_recon: %.3lf; Val_kl: %.3lf  ' % \
              (epoch, num_epochs, 
               safelog10(train_loss[0]), 
               safelog10(train_loss[1]), 
               safelog10(val_loss[0] ),
               safelog10(val_loss[1]) 
               ))


    return {'train_loss': train_history_lost, 'val_loss': val_history_loss}
  

def latent_features(network, data_loader):
    """Obtain latent features learnt by the model

    Args:
        data_loader: (DataLoader) loader containing the data
        return_labels: (boolean) whether to return true labels or not

    Returns:
       features: (array) array containing the features from the data
    """
    network.eval()
    device = next(network.parameters()).device
    features = []
    with torch.no_grad():
        for (flux, time, band, mask) in data_loader:
            encode = network.encode(flux.to(device), 
                          time.to(device), 
                          band.to(device), 
                          mask.to(device))

        # return true labels
        features += [encode.cpu().detach().numpy()]
    return np.array(features)




def plot_latent_space(network, data_loader, save=False):
    """Plot the latent space learnt by the model

    Args:
        data: (array) corresponding array containing the data
        labels: (array) corresponding array containing the labels
        save: (bool) whether to save the latent space plot

    Returns:
        fig: (figure) plot of the latent space
    """
    network.eval()
    # obtain the latent features
    features = latent_features(data_loader)
    
    # plot only the first 2 dimensions
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0, 0], features[:, 0, 1], marker='o',
            edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
    plt.colorbar()
    if(save):
        fig.savefig('latent_space.png')
    return fig