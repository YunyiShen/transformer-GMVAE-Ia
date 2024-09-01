"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder for Unsupervised Clustering

"""
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from networks.Networks import *
from losses.LossFunctions import *
from metrics.Metrics import *
import matplotlib.pyplot as plt

class vanillaVAE:

  def __init__(self, args):
    self.num_epochs = args.epochs
    self.cuda = args.cuda
    self.verbose = args.verbose

    self.batch_size = args.batch_size
    self.batch_size_val = args.batch_size_val
    self.learning_rate = args.learning_rate
    self.decay_epoch = args.decay_epoch
    self.lr_decay = args.lr_decay
    self.w_cat = args.w_categ
    self.w_gauss = args.w_gauss    
    self.w_rec = args.w_rec
    self.rec_type = args.rec_type 

    self.num_classes = args.num_classes
    self.gaussian_size = args.gaussian_size
    self.input_size = args.input_size

    # gumbel
    self.init_temp = args.init_temp
    self.decay_temp = args.decay_temp
    self.hard_gumbel = args.hard_gumbel
    self.min_temp = args.min_temp
    self.decay_temp_rate = args.decay_temp_rate
    self.gumbel_temp = self.init_temp

    self.network = GMVAENet(self.input_size, self.gaussian_size, self.num_classes)
    self.losses = LossFunctions()
    self.metrics = Metrics()

    if self.cuda:
      self.network = self.network.cuda() 
  

  def unlabeled_loss(self, flux, out_net, mask=None):
    """Method defining the loss functions derived from the variational lower bound
    Args:
        flux: (array) corresponding array containing the input data
        out_net: (dict) contains the graph operations or nodes of the network output

    Returns:
        loss_dic: (dict) contains the values of each loss function and predictions
    """
    # obtain network variables
    z, flux_recon = out_net['gaussian'], out_net['x_rec'] 
    #logits, prob_cat = out_net['logits'], out_net['prob_cat']
    y_mu, y_var = out_net['y_mean'], out_net['y_var']
    mu, var = out_net['mean'], out_net['var']
    
    # reconstruction loss
    loss_rec = self.losses.reconstruction_loss(flux, flux_recon, self.rec_type, mask)

    # gaussian loss
    loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

    # categorical loss
    #loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

    # total loss
    loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss #+ self.w_cat * loss_cat

    # obtain predictions
    #_, predicted_labels = torch.max(logits, dim=1)

    loss_dic = {'total': loss_total, 
                #'predicted_labels': predicted_labels,
                'reconstruction': loss_rec,
                'gaussian': loss_gauss}#,
                #'categorical': loss_cat}
    return loss_dic
    
    
  def train_epoch(self, optimizer, data_loader):
    """Train the model for one epoch

    Args:
        optimizer: (Optim) optimizer to use in backpropagation
        data_loader: (DataLoader) corresponding loader containing the training data

    Returns:
        average of all loss values, accuracy, nmi
    """
    self.network.train()
    total_loss = 0.
    recon_loss = 0.
    cat_loss = 0.
    gauss_loss = 0.

    accuracy = 0.
    nmi = 0.
    num_batches = 0.
    
    true_labels_list = []
    predicted_labels_list = []

    # iterate over the dataset
    for (flux, wavelength, mask) in data_loader:
      if self.cuda == 1:
        flux = flux.cuda()
        wavelength = wavelength.cuda()
        mask = mask.cuda()

      optimizer.zero_grad()

      # flatten data
      #flux = flux.view(flux.size(0), -1)
      
      # forward call
      out_net = self.network(flux, wavelength, mask) 
      unlab_loss_dic = self.unlabeled_loss(flux, out_net, mask) 
      total = unlab_loss_dic['total']

      # accumulate values
      total_loss += total.item()
      recon_loss += unlab_loss_dic['reconstruction'].item()
      gauss_loss += unlab_loss_dic['gaussian'].item()

      # perform backpropagation
      total.backward()
      optimizer.step()  

      # save predicted and true labels
      #predicted = unlab_loss_dic['predicted_labels']
      #true_labels_list.append(labels)
      #predicted_labels_list.append(predicted)   
   
      num_batches += 1. 

    # average per batch
    total_loss /= num_batches
    recon_loss /= num_batches
    gauss_loss /= num_batches
    #cat_loss /= num_batches
    
    # concat all true and predicted labels
    #true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
    #predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()

    # compute metrics
    #accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
    #nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)

    return total_loss, recon_loss, gauss_loss#, cat_loss, accuracy, nmi


  def test(self, data_loader, return_loss=False):
    """Test the model with new data

    Args:
        data_loader: (DataLoader) corresponding loader containing the test/validation data
        return_loss: (boolean) whether to return the average loss values
          
    Return:
        accuracy and nmi for the given test data

    """
    self.network.eval()
    total_loss = 0.
    recon_loss = 0.
    
    gauss_loss = 0.

    
    num_batches = 0.
    

    with torch.no_grad():
      for (flux, wavelength, mask) in data_loader:
        if self.cuda == 1:
          flux = flux.cuda()
          wavelength = wavelength.cuda()
          mask = mask.cuda()
      
        # flatten data
        #flux = flux.view(flux.size(0), -1)
      
        # forward call
        out_net = self.network(flux, wavelength, mask) 
        unlab_loss_dic = self.unlabeled_loss(flux, out_net, mask) 
        total = unlab_loss_dic['total']

        # accumulate values
        total_loss += total.item()
        recon_loss += unlab_loss_dic['reconstruction'].item()
        gauss_loss += unlab_loss_dic['gaussian'].item()

   
        num_batches += 1. 

    # average per batch
    if return_loss:
      total_loss /= num_batches
      recon_loss /= num_batches
      gauss_loss /= num_batches
    

      return total_loss, recon_loss, gauss_loss


  def train(self, train_loader, val_loader):
    """Train the model

    Args:
        train_loader: (DataLoader) corresponding loader containing the training data
        val_loader: (DataLoader) corresponding loader containing the validation data

    Returns:
        output: (dict) contains the history of train/val loss
    """
    optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
    train_history_lost, val_history_loss = [], []
    for epoch in range(1, self.num_epochs + 1):
      train_loss, train_rec, train_gauss = self.train_epoch(optimizer, train_loader)
      val_loss, val_rec, val_gauss = self.test(val_loader, True)
      train_history_lost.append(train_loss)
      val_history_loss.append(val_loss)
      # if verbose then print specific information about training
      if self.verbose == 1:
        print("(Epoch %d / %d)" % (epoch, self.num_epochs) )
        print("Train - REC: %.5lf;  Gauss: %.5lf;" % \
              (train_rec, train_gauss))
        print("Valid - REC: %.5lf;  Gauss: %.5lf;" % \
              (val_rec, val_gauss))
        print(" Total Loss=Train: %.5lf; Val: %.5lf" % \
              ( train_loss, val_loss))
      else:
        print('(Epoch %d / %d) Train_Loss: %.3lf; Val_Loss: %.3lf  ' % \
              (epoch, self.num_epochs, train_loss, val_loss))


    return {'train_loss': train_history_lost, 'val_loss': val_history_loss}
  

  def latent_features(self, data_loader):
    """Obtain latent features learnt by the model

    Args:
        data_loader: (DataLoader) loader containing the data
        return_labels: (boolean) whether to return true labels or not

    Returns:
       features: (array) array containing the features from the data
    """
    self.network.eval()
    N = len(data_loader.dataset)
    features = np.zeros((N, self.gaussian_size))
    start_ind = 0
    with torch.no_grad():
      for (flux, wavelength, mask) in data_loader:
        if self.cuda == 1:
          flux = flux.cuda()
          wavelength = wavelength.cuda()
          mask = mask.cuda()
      
        out = self.network.inference(flux, wavelength, mask)
        latent_feat = out['mean']
        end_ind = min(start_ind + flux.size(0), N+1)

        # return true labels
        features[start_ind:end_ind] = latent_feat.cpu().detach().numpy()  
        start_ind += flux.size(0)
    return features


  def reconstruct_data(self, data_loader, sample_size=-1):
    """Reconstruct Data

    Args:
        data_loader: (DataLoader) loader containing the data
        sample_size: (int) size of random data to consider from data_loader
      
    Returns:
        reconstructed: (array) array containing the reconstructed data
    """
    self.network.eval()

    # sample random data from loader
    indices = np.random.randint(0, len(data_loader.dataset), size=sample_size)
    test_random_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=sample_size, sampler=SubsetRandomSampler(indices))
  
    # obtain values
    it = iter(test_random_loader)
    test_batch_data_flux, test_batch_data_wavelength, test_batch_data_mask  = it.next()
    original = test_batch_data.data.numpy()
    if self.cuda:
      test_batch_data = test_batch_data.cuda()  
      test_batch_data_wavelength = test_batch_data_wavelength.cuda()
      test_batch_data_mask = test_batch_data_mask.cuda()

    # obtain reconstructed data  
    out = self.network(test_batch_data, test_batch_data_wavelength, test_batch_data_mask) 
    reconstructed = out['x_rec']
    return original, reconstructed.data.cpu().numpy()


  def plot_latent_space(self, data_loader, save=False):
    """Plot the latent space learnt by the model

    Args:
        data: (array) corresponding array containing the data
        labels: (array) corresponding array containing the labels
        save: (bool) whether to save the latent space plot

    Returns:
        fig: (figure) plot of the latent space
    """
    # obtain the latent features
    features = self.latent_features(data_loader)
    
    # plot only the first 2 dimensions
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], marker='o',
            edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
    plt.colorbar()
    if(save):
        fig.savefig('latent_space.png')
    return fig
  
  
  def random_generation(self, wavelength, mask, num_elements=1):
    """Random generation for each category

    Args:
        num_elements: (int) number of elements to generate

    Returns:
        generated data according to num_elements
    """ 
    

  
    # infer the gaussian distribution according to the category
    mean, var = self.network.generative.pzy(wavelength, mask)

    # gaussian random sample by using the mean and variance
    noise = torch.randn_like(var)
    std = torch.sqrt(var)
    gaussian = mean + noise * std

    # generate new samples with the given gaussian
    generated = self.network.generative.pxz(gaussian)

    return generated.cpu().detach().numpy() 

