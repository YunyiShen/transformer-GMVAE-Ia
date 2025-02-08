import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from VAESNe.SpectraNetworks import vanillaSpectraVAENet
from VAESNe.VanillaVAE_trainer import train
from VAESNe.losses import VAEloss

torch.manual_seed(0)


data = np.load('../data/goldstein_processed/preprocessed_midfilt_3_centeringFalse_realisticLSST_phase.npz')
training_idx = data['training_idx']
testing_idx = data['testing_idx']


flux, wavelength, mask = data['flux'][training_idx,:], data['wavelength'][training_idx,:], data['mask'][training_idx,:]
phase = data['phase'][training_idx]

flux_test, wavelength_test, mask_test = data['flux'][testing_idx], data['wavelength'][testing_idx], data['mask'][testing_idx]
phase_test = data['phase'][testing_idx]


flux = torch.tensor(flux, dtype=torch.float32)
wavelength = torch.tensor(wavelength, dtype=torch.float32)
mask = torch.tensor(mask == 0)
phase = torch.tensor(phase, dtype=torch.float32)

flux_test = torch.tensor(flux_test, dtype=torch.float32)
wavelength_test = torch.tensor(wavelength_test, dtype=torch.float32)
mask_test = torch.tensor(mask_test == 0)
phase_test = torch.tensor(phase_test, dtype=torch.float32)

# do some data augmentation on flux and time, the data is already repeated multiple times 
flux = flux + 0.02 * torch.randn_like(flux)
#wavelength = wavelength + 0.1 * torch.randn(wavelength.shape[0])[:,None] # shift all time in a single light curve by the same amount
# randomly set some masks to be True
mask = torch.logical_or(mask, torch.rand_like(flux) < 0.025)
#breakpoint()

# split loaded data into training and validation
train_dataset = TensorDataset(flux, wavelength, phase, mask)
test_dataset = TensorDataset(flux_test, wavelength_test, phase_test, mask_test)
test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, pin_memory=True)


lr = 1e-4
epochs = 300

loss_fn = VAEloss(beta = 1.).to(device)
#breakpoint()
my_vaesne = vanillaSpectraVAENet(
    # data parameters
    spectra_length = 982,

    # model parameters
    latent_length = 4,
    latent_dim = 2,
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
             loss_fn = loss_fn, 
             device = device,
             release_memory = False
             )

torch.save(my_vaesne, '../ckpt/first_spectra_vaesne.pth')

