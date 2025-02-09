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

flux_test, wavelength_test, mask_test = data['flux'][testing_idx], data['wavelength'][testing_idx], data['mask'][testing_idx]
phase_test = data['phase'][testing_idx]


idx = 17

flux_test = torch.tensor(flux_test, dtype=torch.float32)
wavelength_test = torch.tensor(wavelength_test, dtype=torch.float32)
mask_test = torch.tensor(mask_test == 0)
phase_test = torch.tensor(phase_test, dtype=torch.float32)

trained_vae = torch.load("../ckpt/first_spectra_vaesne.pth",
                         map_location=torch.device('cpu'))

#breakpoint()
reconstruction = trained_vae.reconstruct(flux_test[idx][None,:],
                                         wavelength_test[idx][None,:],
                                            phase_test[idx][None],
                                            mask_test[idx][None,:])


import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
axs.plot(wavelength_test[idx], flux_test[idx], label='ground truth')
axs.plot(wavelength_test[idx], reconstruction[0].detach().numpy(), label='reconstruction')



# invert y
axs.set_ylim(-2, 2)
axs.legend()



#plt.tight_layout()

plt.show()
plt.savefig("spectra_reconstruction.png")
plt.close()





