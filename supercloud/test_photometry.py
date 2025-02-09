import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from VAESNe.PhotometryNetworks import PhotometricVAENet
from VAESNe.VanillaVAE_trainer import train
from VAESNe.losses import VAEloss


data = np.load('../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_trunc15_phase.npz')
training_idx = data['training_idx']
testing_idx = data['testing_idx']
photoflux, phototime, photomask = data['photoflux'][training_idx,:], data['phototime'][training_idx,:], data['photomask'][training_idx,:]
photoband = data['photowavelength'][training_idx,:]

photo_flux_test, phototime_test, photomask_test = data['photoflux'][testing_idx], data['phototime'][testing_idx], data['photomask'][testing_idx]
photoband_test = data['photowavelength'][testing_idx]


photoflux = torch.tensor(photoflux, dtype=torch.float32)
phototime = torch.tensor(phototime, dtype=torch.float32)
photomask = torch.tensor(photomask == 0)
photoband = torch.tensor(photoband, dtype=torch.long)

photoflux_test = torch.tensor(photo_flux_test, dtype=torch.float32)
phototime_test = torch.tensor(phototime_test, dtype=torch.float32)
photomask_test = torch.tensor(photomask_test == 0)
photoband_test = torch.tensor(photoband_test, dtype=torch.long)


# do some data augmentation on flux and time, the data is already repeated multiple times 
photoflux = photoflux + 0.02 * torch.randn_like(photoflux)
phototime = phototime + 0.1 * torch.randn(phototime.shape[0])[:,None] # shift all time in a single light curve by the same amount
# randomly set some masks to be True
photomask = torch.logical_or(photomask, torch.rand_like(photoflux) < 0.025)
#breakpoint()

# split loaded data into training and validation
train_dataset = TensorDataset(photoflux, phototime, photoband, photomask)
test_dataset = TensorDataset(photoflux_test, phototime_test, photoband_test, photomask_test)
test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


lr = 5e-4
epochs = 300

loss_fn = VAEloss(beta = 1.).to(device)

my_vaesne = PhotometricVAENet(
    # data parameters
    photometric_length = 90,
    num_bands = 6,

    # model parameters
    latent_len = 3,
    latent_dim = 1,
    model_dim = 32, 
    num_heads = 4, 
    ff_dim = 32, 
    num_layers = 4,
    dropout = 0.1).to(device)

losses = train(my_vaesne,
               train_loader, 
               val_loader,
              lr,
              epochs,
             loss_fn = loss_fn)

torch.save(my_vaesne, '../ckpt/first_vaesne_3-1.pth')

