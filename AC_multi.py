#%%

import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from tensorflow.keras.utils import to_categorical

#%%

dt = 0.005
kappa = 3.0
L = 10
Nx, Ny = 64, 64
dx, dy = 1.0, 1.0
num_regions_in = 5
ntimestep = 8000
nprint = 50
coefA = 1.0
dexp = -5
IMG_DIM = 64
seed = 42

np.random.seed(seed)
#%%
# Set random seed for reproducibility
np.random.seed(seed)

class GenerateEtas:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def create_etas(self):
        img = cv2.resize(self.snapshot, (IMG_DIM, IMG_DIM))
        img[img != 255] = 0  # Thresholding
        # mask = img == 255
        
        # cv2.imshow('Resized Grayscale Image', resized_img)
        _, thresholded_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        mask = thresholded_img == 255
        structure = np.ones((3, 3), dtype=int)
        labeled_mask, num_labels = ndimage.label(mask, structure=structure)
        n_classes = len(np.unique(labeled_mask))
        s=to_categorical(labeled_mask, num_classes=n_classes)[:, :, 1:n_classes]
        return s
        
        
        
        # structure = np.ones((3, 3), dtype=int)
        # labeled_mask, num_labels = ndimage.label(mask, structure=structure)
        # n_classes = len(np.unique(labeled_mask))
        # return to_categorical(labeled_mask, num_classes=n_classes)[:, :, 1:n_classes]


class ReadInput:
    def __init__(self, dim=IMG_DIM, input_folder="Input"):
        self.dim = dim
        self.direct = os.path.join(os.getcwd(), input_folder)

    def read(self):
        if not os.path.exists(self.direct):
            raise FileNotFoundError(f"Directory {self.direct} does not exist.")
        image_files = [f for f in os.listdir(self.direct) if f.endswith('.png')]
        if not image_files:
            raise ValueError(f"No PNG files found in directory {self.direct}.")
        return [cv2.resize(cv2.imread(os.path.join(self.direct, f), 0), (self.dim, self.dim)) for f in image_files]


class AllenCahnModel:
    def __init__(self, Nx, Ny, num_regions, dx, dy):
        self.Nx, self.Ny, self.num_regions = Nx, Ny, num_regions
        self.kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
        self.kx, self.ky = np.meshgrid(self.kx, self.ky, indexing='ij')

    def k2_k4(self):
        k2 = self.kx**2 + self.ky**2
        return k2**2, k2

    def get_images(self):
        reader = ReadInput()
        return reader.read()


def free_energy(etas, eta, igrain, ngrain):
    A, B = 1.0, 1.0
    summ = np.sum(etas[:, :, np.arange(ngrain) != igrain]**2, axis=2)
    return A * (2.0 * B * eta * summ + eta**3 - eta)

#%%
start_time = time.time()

model = AllenCahnModel(Nx, Ny, num_regions_in, dx, dy)

all_images = model.get_images()


k4, k2 = model.k2_k4()

full = np.zeros((len(all_images), ntimestep // nprint, Nx, Ny), dtype=np.float16)
boundary = np.zeros_like(full)


#%%
for ii, img in tqdm(enumerate(all_images), desc='Simulation Progress'):
    try:
        etas = GenerateEtas(img).create_etas()
        num_regions = etas.shape[2]
        glist = np.ones(num_regions)
        elapsed = 0

        for step in range(ntimestep):
            elapsed += dt
            for igrain in range(num_regions):
                if glist[igrain] == 1:
                    eta = etas[:, :, igrain]
                    dfdeta = free_energy(etas, eta, igrain, num_regions)
                    etak = np.fft.fft2(eta)
                    dfdetak = np.fft.fft2(dfdeta)
                    etak = (etak - dt * L * dfdetak) / (1 + dt * coefA * L * kappa * k2)
                    eta = np.fft.ifft2(etak).real
                    eta = np.clip(eta, 0.00000, 0.9999)
                    etas[:, :, igrain] = eta

                    if np.mean(eta) < 0.001:
                        glist[igrain] = 0
                        etas[:, :, igrain] = 0
                        continue

            if step % nprint == 0:
                if etas.shape[2]>0:

                    microstructure = np.argmax(etas, axis=2)
                    eta3 = np.sum(etas**2, axis=2)
                else:
                    microstructure = np.zeros((Nx, Ny), dtype=int)
                    eta3 = np.zeros((Nx, Ny))

                full[ii, step // nprint] = microstructure
                boundary[ii, step // nprint] = eta3
    except Exception as e:
        print(f"Error processing image {ii} at step {step}: {e}")
        continue
#%%


np.save("Dataset_b_128.npy", boundary)

print(f"\nCompute time: {time.time() - start_time:.2f} seconds")


# %%
