import os
import torch
import io
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np


num_patches = 25
num_tiles = 44
num_patches_test = 25
num_test_tiles = 14
size_patch_10 = 366
size_patch_20 = 183
size_patch_60 = 61
channels10 = 4
channels20 = 6
channels60 = 2


f_test = open('S2_tiles_testing_2.txt', 'r')

HR_test = torch.zeros((num_test_tiles, num_patches_test, channels10, size_patch_10, size_patch_10), dtype=torch.float32)
LR_test = torch.zeros((num_test_tiles, num_patches_test, channels20, size_patch_20, size_patch_20), dtype=torch.float32)
VLR_test = torch.zeros((num_test_tiles, num_patches_test, channels60, size_patch_60, size_patch_60), dtype=torch.float32)
target20_test = torch.zeros((num_test_tiles, num_patches_test, channels20, size_patch_10, size_patch_10), dtype=torch.float32)
target60_test = torch.zeros((num_test_tiles, num_patches_test, channels60, size_patch_10, size_patch_10), dtype=torch.float32)

root_test_path = "/mnt/gpid07/users/oriol.esquena/patches_sentinel/test_all/"

k = 0

for i in f_test:
    im_name = i
    date = im_name[11:26]

    test_HR = "test_resized60_"+date+".npy"
    test_LR = "test_resized120_"+date+".npy"
    test_VLR = "test_resized360_"+date+".npy"
    test_target20 = "real20_target_test_"+date+".npy"
    test_target60 = "real60_target_test_"+date+".npy"

    # read testing data from *.npy file
    HR_test_np = (np.load(root_test_path+test_HR))
    LR_test_np = (np.load(root_test_path+test_LR))
    VLR_test_np = (np.load(root_test_path+test_VLR))
    target_test20_np = (np.load(root_test_path+test_target20))
    target_test60_np = (np.load(root_test_path+test_target60))

    HR_test[k] = ((torch.from_numpy(HR_test_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    LR_test[k] = ((torch.from_numpy(LR_test_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    VLR_test[k] = ((torch.from_numpy(VLR_test_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    target20_test[k] = ((torch.from_numpy(target_test20_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    target60_test[k] = ((torch.from_numpy(target_test60_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)

    k += 1

HR_test = HR_test.reshape((num_test_tiles*num_patches_test, channels10, size_patch_10, size_patch_10)) / 2000
LR_test = LR_test.reshape((num_test_tiles*num_patches_test, channels20, size_patch_20, size_patch_20)) / 2000
VLR_test = VLR_test.reshape((num_test_tiles*num_patches_test, channels60, size_patch_60, size_patch_60)) / 2000
target20_test = target20_test.reshape((num_test_tiles*num_patches_test, channels20, size_patch_10, size_patch_10)) / 2000
target60_test = target60_test.reshape((num_test_tiles*num_patches_test, channels60, size_patch_10, size_patch_10)) / 2000

f_test.close()


f_train = open('S2_tiles_training_2.txt', 'r')

HR_data = torch.zeros((num_tiles, num_patches, channels10, size_patch_10, size_patch_10), dtype=torch.float32)
LR_data = torch.zeros((num_tiles, num_patches, channels20, size_patch_20, size_patch_20), dtype=torch.float32)
VLR_data = torch.zeros((num_tiles, num_patches, channels60, size_patch_60, size_patch_60), dtype=torch.float32)
target20_data = torch.zeros((num_tiles, num_patches, channels20, size_patch_10, size_patch_10), dtype=torch.float32)
target60_data = torch.zeros((num_tiles, num_patches, channels60, size_patch_10, size_patch_10), dtype=torch.float32)

root_train_path = "/mnt/gpid07/users/oriol.esquena/patches_sentinel/train_all/"

k = 0

for i in f_train:
    im_name = i
    date = im_name[11:26]

    file_name_HR = "input10_resized60_"+date+".npy"
    file_name_LR = "input20_resized120_"+date+".npy"
    file_name_VLR = "input60_resized360_"+date+".npy"
    file_name_target20 = "real20_target_"+date+".npy"
    file_name_target60 = "real60_target_"+date+".npy"

    # read training data from *.npy file
    HR_data_np = (np.load(root_train_path+file_name_HR))
    LR_data_np = (np.load(root_train_path+file_name_LR))
    VLR_data_np = (np.load(root_train_path+file_name_VLR))
    target20_data_np = (np.load(root_train_path+file_name_target20))
    target60_data_np = (np.load(root_train_path+file_name_target60))

    HR_data[k] = ((torch.from_numpy(HR_data_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    LR_data[k] = ((torch.from_numpy(LR_data_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    VLR_data[k] = ((torch.from_numpy(VLR_data_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    target20_data[k] = ((torch.from_numpy(target20_data_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
    target60_data[k] = ((torch.from_numpy(target60_data_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)

    k += 1

HR_data = HR_data.reshape((num_tiles*num_patches, channels10, size_patch_10, size_patch_10)) / 2000
LR_data = LR_data.reshape((num_tiles*num_patches, channels20, size_patch_20, size_patch_20)) / 2000
VLR_data = VLR_data.reshape((num_tiles*num_patches, channels60, size_patch_60, size_patch_60)) / 2000
target20_data = target20_data.reshape((num_tiles*num_patches, channels20, size_patch_10, size_patch_10)) / 2000
target60_data = target60_data.reshape((num_tiles*num_patches, channels60, size_patch_10, size_patch_10)) / 2000

f_train.close()


class PatchesDataset(Dataset):
    """Patches dataset."""

    def __init__(self, hr, lr, vlr, target20, target60, transform=None):
        """
        Args:
            HRdata: images of high resolution
            LRdata: images of low resolution
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hr = hr
        self.lr = lr
        self.vlr = vlr
        self.target20 = target20
        self.target60 = target60
        self.transform = None

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, idx):

        hr_data = self.hr[idx]
        lr_data = self.lr[idx]
        vlr_data = self.vlr[idx]
        t20_data = self.target20[idx]
        t60_data = self.target60[idx]

        if self.transform is not None:
            meanLR, std_devLR, meanTAR, std_devTAR, imageHR, imageLR, imageTAR = self.transform(hr_data, lr_data, t_data)

        return hr_data, lr_data, vlr_data, t20_data, t60_data


set_ds = PatchesDataset(HR_data, LR_data, VLR_data, target20_data, target60_data)
# train_ds, val_ds = torch.utils.data.random_split(set_ds, [990, 110])
test_ds = PatchesDataset(HR_test, LR_test, VLR_test, target20_test, target60_test)
