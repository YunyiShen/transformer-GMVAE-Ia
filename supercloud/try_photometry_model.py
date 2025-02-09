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

photo_flux_test, phototime_test, photomask_test = data['photoflux'][testing_idx], data['phototime'][testing_idx], data['photomask'][testing_idx]
photoband_test = data['photowavelength'][testing_idx]

idx = 15

photoflux_test = torch.tensor(photo_flux_test, dtype=torch.float32)
phototime_test = torch.tensor(phototime_test, dtype=torch.float32)
photomask_test = torch.tensor(photomask_test == 0)
photoband_test = torch.tensor(photoband_test, dtype=torch.long)

trained_vae = torch.load("../ckpt/first_vaesne_3-1.pth",
                         map_location=torch.device('cpu'))

reconstruction = trained_vae.reconstruct(photoflux_test[idx][None,:],
                                         phototime_test[idx][None,:],
                                            photoband_test[idx][None,:],
                                            photomask_test[idx][None,:])


import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for i in range(6):
    thisband_gt = photoflux_test[idx][torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]
    thisband_time = phototime_test[idx][torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]
    thisband_rec = reconstruction['reconstruction'][0].detach()
    
    thisband_rec = thisband_rec[torch.logical_and(
        photoband_test[idx] == i,
        torch.logical_not(photomask_test[idx]))]
    axs[0].plot(thisband_time, thisband_gt)
    axs[0].scatter(thisband_time, thisband_gt, s=20, marker='o')

    axs[1].plot(thisband_time, thisband_rec)
    axs[1].scatter(thisband_time, thisband_rec, s=20, marker='x')

# invert y
axs[0].set_ylim(-2, 6)
axs[1].set_ylim(-2, 6)
axs[0].invert_yaxis()
axs[0].set_title("Ground truth")
axs[1].invert_yaxis()
axs[1].set_title("Reconstruction")


#plt.tight_layout()

plt.show()
plt.savefig("reconstruction.png")
plt.close()





