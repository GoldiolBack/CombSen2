from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from create_dataset_both import set_ds, test_ds
from torch.utils.data.sampler import SubsetRandomSampler
import skimage.metrics as skm
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

logs_base_dir = "runs"
os.makedirs(logs_base_dir, exist_ok=True)

tb = SummaryWriter()


def crop(imageHR, imageLR, imageVLR, target20, target60, sizeHR, sizeLR, sizeVLR, sizeTarget):
    imageHR_crop = torch.zeros([imageHR.shape[0] * 6, imageHR.shape[1], sizeHR, sizeHR], dtype=torch.float32)
    imageLR_crop = torch.zeros([imageLR.shape[0] * 6, imageLR.shape[1], sizeLR, sizeLR], dtype=torch.float32)
    imageVLR_crop = torch.zeros([imageVLR.shape[0] * 6, imageVLR.shape[1], sizeVLR, sizeVLR], dtype=torch.float32)
    target20_crop = torch.zeros([target20.shape[0] * 6, target20.shape[1], sizeTarget, sizeTarget], dtype=torch.float32)
    target60_crop = torch.zeros([target60.shape[0] * 6, target60.shape[1], sizeTarget, sizeTarget], dtype=torch.float32)
    m = 0
    x = rd.sample(range(imageHR.shape[0] * 6), imageHR.shape[0] * 6)
    hi = np.floor((imageHR.shape[2] - sizeHR)/6)

    for i in range(imageHR.shape[0]):
        for cr in range(6):
            j1 = np.random.randint(low=0, high=hi)*6
            j2 = np.round((j1 / 2)).astype(dtype=np.int)
            j6 = np.round((j1 / 6)).astype(dtype=np.int)
            k1 = np.random.randint(low=0, high=imageHR.shape[2] - sizeHR)
            k2 = np.round((k1 / 2)).astype(dtype=np.int)
            k6 = np.round((k1 / 6)).astype(dtype=np.int)
            imageHR_crop[x[m]] = imageHR[i, :, j1:(j1 + sizeHR), k1:(k1 + sizeHR)]
            imageLR_crop[x[m]] = imageLR[i, :, j2:(j2 + sizeLR), k2:(k2 + sizeLR)]
            imageVLR_crop[x[m]] = imageVLR[i, :, j6:(j6 + sizeVLR), k6:(k6 + sizeVLR)]
            target20_crop[x[m]] = target20[i, :, j1:(j1 + sizeTarget), k1:(k1 + sizeTarget)]
            target60_crop[x[m]] = target60[i, :, j1:(j1 + sizeTarget), k1:(k1 + sizeTarget)]
            m += 1

    return imageHR_crop, imageLR_crop, imageVLR_crop, target20_crop, target60_crop


# def des_std_im(meanLR, std_devLR, meanTAR, std_devTAR, lr, output, target):
#     lr = (lr * std_devLR) + meanLR
#     output = (output * std_devLR) + meanLR
#     target = (target * std_devTAR) + meanTAR
#
#     return lr, output, target


class Net(nn.Module):
    def __init__(self, input_size=12, feature_size=128, kernel_size=3):
        super(Net, self).__init__()
        self.ups2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.ups6 = nn.Upsample(scale_factor=6, mode='bicubic', align_corners=True)
        self.conv1 = nn.Conv2d(input_size, feature_size, kernel_size, stride=(1, 1), padding=(1, 1))
        self.conv2_2 = nn.Conv2d(feature_size, 6, kernel_size, 1, 1)
        self.conv2_6 = nn.Conv2d(feature_size, 2, kernel_size, 1, 1)
        self.rBlock = ResBlock(feature_size, kernel_size)

    def forward(self, input10, input20, input60, num_layers=6):
        upsamp20 = self.ups2(input20)
        upsamp60 = self.ups6(input60)
        sentinel = torch.cat((input10, upsamp20, upsamp60), 1)
        x = sentinel
        x = self.conv1(x)
        x = F.relu(x)
        for i in range(num_layers):
            x = self.rBlock(x)
        y = self.conv2_2(x)
        z = self.conv2_6(x)
        y += upsamp20
        z += upsamp60
        return y, z, upsamp20, upsamp60


class ResBlock(nn.Module):
    def __init__(self, feature_size=128, channels=3, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size, 1, 1)

    def forward(self, x, scale=0.1):
        tmp = self.conv3(x)
        tmp = F.relu(tmp)
        tmp = self.conv3(tmp)
        tmp = tmp * scale
        tmp += x
        return tmp


def train(args, train_loader, model, device, optimizer, epoch):
    model.train()
    for batch_idx, (hr, lr, vlr, target20, target60) in enumerate(train_loader):
        hr_crop, lr_crop, vlr_crop, target20_crop, target60_crop = crop(hr, lr, vlr, target20, target60,
                                                                        int(args.crop_size), int(args.crop_size / 2),
                                                                        int(args.crop_size / 6), int(args.crop_size))
        lr_crop, hr_crop, vlr_crop, target20_crop, target60_crop = lr_crop.to(device), hr_crop.to(device), vlr_crop.to(device), target20_crop.to(device), target60_crop.to(device)
        optimizer.zero_grad()
        output2, output6, ups2, ups6 = model(hr_crop, lr_crop, vlr_crop)
        loss_function = nn.L1Loss()
        loss_2 = loss_function(output2, target20_crop)
        loss_6 = loss_function(output6, target60_crop)
        loss = loss_2*0.8 + loss_6*0.2
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_2: {:.6f}\tLoss_6: {:.6f}'.format(
                epoch, batch_idx * len(hr_crop), 990*6,
                ((100. * batch_idx * len(hr_crop)) / (990*6)), loss.item(), loss_2.item(), loss_6.item()))
            tb.add_scalar("Loss_train", loss.item(), epoch)


def test(args, test_loader, model, device, epoch):
    model.eval()
    test_loss2 = 0
    test_loss6 = 0
    rmse2 = 0
    psnr2 = 0
    ssim2 = 0
    rmse_i2 = 0
    psnr_i2 = 0
    ssim_i2 = 0
    rmse6 = 0
    psnr6 = 0
    ssim6 = 0
    rmse_i6 = 0
    psnr_i6 = 0
    ssim_i6 = 0
    ka = 0
    kb = 0
    kc = 0
    kd = 0
    m = 0
    with torch.no_grad():
        for hr, lr, vlr, target20, target60 in test_loader:
            lr_crop, hr_crop, vlr_crop, target20_crop, target60_crop = lr.to(device), hr.to(device), vlr.to(device), target20.to(device), target60.to(device)
            output2, output6, ups2, ups6 = model(hr_crop, lr_crop, vlr_crop)
            test_loss_function = nn.L1Loss(reduction='mean')
            test_loss2 += test_loss_function(output2, target20_crop).item()
            test_loss6 += test_loss_function(output6, target60_crop).item()
            real20 = (np.moveaxis(target20_crop.cpu().numpy(), 1, 3)) * 2000
            predicted20 = (np.moveaxis(output2.cpu().numpy(), 1, 3)) * 2000
            real60 = (np.moveaxis(target60_crop.cpu().numpy(), 1, 3)) * 2000
            predicted60 = (np.moveaxis(output6.cpu().numpy(), 1, 3)) * 2000
            input2 = (np.moveaxis(ups2.cpu().numpy(), 1, 3)) * 2000
            input6 = (np.moveaxis(ups6.cpu().numpy(), 1, 3)) * 2000
            for i in range(real20.shape[0]):
                for j in range(real20.shape[3]):
                    if np.sqrt(skm.mean_squared_error(real20[i, :, :, :], input2[i, :, :, :])) < np.sqrt(
                            skm.mean_squared_error(real20[i, :, :, :], predicted20[i, :, :, :])):
                        ka += 1
                        kb += 1
                    elif skm.peak_signal_noise_ratio(real20[i, :, :, j], input2[i, :, :, j],
                                                   data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min()) > 300:
                        ka += 1
                    elif skm.peak_signal_noise_ratio(real20[i, :, :, j], input2[i, :, :, j],
                                                     data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min()) < 0:
                        ka += 1
                    elif real20[i, :, :, j].max() - real20[i, :, :, j].min() == 0:
                        ka += 1
                    else:
                        rmse_i2 += np.sqrt(skm.mean_squared_error(real20[i, :, :, j], input2[i, :, :, j]))
                        psnr_i2 += skm.peak_signal_noise_ratio(real20[i, :, :, j], input2[i, :, :, j],
                                                               data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min())
                        ssim_i2 += skm.structural_similarity(real20[i, :, :, j], input2[i, :, :, j], multichannel=False,
                                                             data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min())

                    if skm.peak_signal_noise_ratio(real20[i, :, :, j], predicted20[i, :, :, j],
                                                   data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min()) > 300:
                        kb += 1
                    elif skm.peak_signal_noise_ratio(real20[i, :, :, j], predicted20[i, :, :, j],
                                                     data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min()) < 0:
                        kb += 1
                    elif real20[i, :, :, j].max() - real20[i, :, :, j].min() == 0:
                        kb += 1
                    else:
                        rmse2 += np.sqrt(skm.mean_squared_error(real20[i, :, :, j], predicted20[i, :, :, j]))
                        psnr2 += skm.peak_signal_noise_ratio(real20[i, :, :, j], predicted20[i, :, :, j],
                                                             data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min())
                        ssim2 += skm.structural_similarity(real20[i, :, :, j], predicted20[i, :, :, j], multichannel=False,
                                                           data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min())

                    if j < real60.shape[3]:
                        if skm.peak_signal_noise_ratio(real60[i, :, :, j], input6[i, :, :, j],
                                                       data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min()) > 300:
                            kc += 1
                        elif skm.peak_signal_noise_ratio(real60[i, :, :, j], input6[i, :, :, j],
                                                         data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min()) < 0:
                            kc += 1
                        elif real60[i, :, :, j].max() - real60[i, :, :, j].min() == 0:
                            kc += 1
                        else:
                            rmse_i6 += np.sqrt(skm.mean_squared_error(real60[i, :, :, j], input6[i, :, :, j]))
                            psnr_i6 += skm.peak_signal_noise_ratio(real60[i, :, :, j], input6[i, :, :, j],
                                                                   data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min())
                            ssim_i6 += skm.structural_similarity(real60[i, :, :, j], input6[i, :, :, j], multichannel=False,
                                                                 data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min())

                        if skm.peak_signal_noise_ratio(real60[i, :, :, j], predicted60[i, :, :, j],
                                                       data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min()) > 300:
                            kd += 1
                        elif skm.peak_signal_noise_ratio(real60[i, :, :, j], predicted60[i, :, :, j],
                                                         data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min()) < 0:
                            kd += 1
                        elif real60[i, :, :, j].max() - real60[i, :, :, j].min() == 0:
                            kd += 1
                        else:
                            rmse6 += np.sqrt(skm.mean_squared_error(real60[i, :, :, j], predicted60[i, :, :, j]))
                            psnr6 += skm.peak_signal_noise_ratio(real60[i, :, :, j], predicted60[i, :, :, j],
                                                                 data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min())
                            ssim6 += skm.structural_similarity(real60[i, :, :, j], predicted60[i, :, :, j], multichannel=False,
                                                               data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min())
            m += 1

    print(ka)
    print(kb)
    print(kc)
    print(kd)
    print(len(test_loader.dataset))
    rmse_i2 /= ((real20.shape[3] * len(test_loader.dataset)) - ka)
    psnr_i2 /= ((real20.shape[3] * len(test_loader.dataset)) - ka)
    ssim_i2 /= ((real20.shape[3] * len(test_loader.dataset)) - ka)
    rmse2 /= ((real20.shape[3] * len(test_loader.dataset)) - kb)
    psnr2 /= ((real20.shape[3] * len(test_loader.dataset)) - kb)
    ssim2 /= ((real20.shape[3] * len(test_loader.dataset)) - kb)
    rmse_i6 /= ((real60.shape[3] * len(test_loader.dataset)) - kc)
    psnr_i6 /= ((real60.shape[3] * len(test_loader.dataset)) - kc)
    ssim_i6 /= ((real60.shape[3] * len(test_loader.dataset)) - kc)
    rmse6 /= ((real60.shape[3] * len(test_loader.dataset)) - kd)
    psnr6 /= ((real60.shape[3] * len(test_loader.dataset)) - kd)
    ssim6 /= ((real60.shape[3] * len(test_loader.dataset)) - kd)
    test_loss2 /= m
    test_loss6 /= m

    print('\nTest set (x2): Average values between Input and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f})\n'.format(rmse_i2, psnr_i2, ssim_i2))

    print('\nTest set (x2): Average values between Output and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f}), Loss: {:.6f}\n'.format(rmse2, psnr2, ssim2, test_loss2))

    print('\nTest set (x6): Average values between Input and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f})\n'.format(rmse_i6, psnr_i6, ssim_i6))

    print('\nTest set (x6): Average values between Output and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f}), Loss: {:.6f}\n'.format(rmse6, psnr6, ssim6, test_loss6))

    print('\nTest set total: Average values between Input and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f})\n'.format((rmse_i6+rmse_i2)/2, (psnr_i6+psnr_i2)/2, (ssim_i6+ssim_i2)/2))

    print('\nTest set total: Average values between Output and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f}), Loss: {:.6f}\n'.format((rmse6+rmse2)/2, (psnr6+psnr2)/2, (ssim6+ssim2)/2, (test_loss6+test_loss2)/2))

    np.save('test_input2.npy', (np.moveaxis(lr.numpy(), 1, 3) * 2000))
    np.save('test_input6.npy', (np.moveaxis(vlr.numpy(), 1, 3) * 2000))
    np.save('test_bicubic2.npy', input2)
    np.save('test_bicubic6.npy', input6)
    np.save('test_real20.npy', real20)
    np.save('test_real60.npy', real60)
    np.save('test_output20.npy', predicted20)
    np.save('test_output60.npy', predicted60)


def validation(args, val_loader, model, device, epoch):
    model.eval()
    val_loss2 = 0
    val_loss6 = 0
    rmse2 = 0
    psnr2 = 0
    ssim2 = 0
    rmse_i2 = 0
    psnr_i2 = 0
    ssim_i2 = 0
    rmse6 = 0
    psnr6 = 0
    ssim6 = 0
    rmse_i6 = 0
    psnr_i6 = 0
    ssim_i6 = 0
    ka = 0
    kb = 0
    kc = 0
    kd = 0
    m = 0
    with torch.no_grad():
        for hr, lr, vlr, target20, target60 in val_loader:
            lr_crop, hr_crop, vlr_crop, target20_crop, target60_crop = lr.to(device), hr.to(device), vlr.to(device), target20.to(device), target60.to(device)
            output2, output6, ups2, ups6 = model(hr_crop, lr_crop, vlr_crop)
            val_loss_function = nn.L1Loss(reduction='mean')
            val_loss2 += val_loss_function(output2, target20_crop).item()
            val_loss6 += val_loss_function(output6, target60_crop).item()
            real20 = (np.moveaxis(target20_crop.cpu().numpy(), 1, 3)) * 2000
            predicted20 = (np.moveaxis(output2.cpu().numpy(), 1, 3)) * 2000
            real60 = (np.moveaxis(target60_crop.cpu().numpy(), 1, 3)) * 2000
            predicted60 = (np.moveaxis(output6.cpu().numpy(), 1, 3)) * 2000
            input2 = (np.moveaxis(ups2.cpu().numpy(), 1, 3)) * 2000
            input6 = (np.moveaxis(ups6.cpu().numpy(), 1, 3)) * 2000
            for i in range(real20.shape[0]):
                for j in range(real20.shape[3]):
                    if skm.peak_signal_noise_ratio(real20[i, :, :, j], input2[i, :, :, j],
                                                   data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min()) > 300:
                        ka += 1
                    elif skm.peak_signal_noise_ratio(real20[i, :, :, j], input2[i, :, :, j],
                                                     data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min()) < 0:
                        ka += 1
                    elif real20[i, :, :, j].max() - real20[i, :, :, j].min() == 0:
                        ka += 1
                    else:
                        rmse_i2 += np.sqrt(skm.mean_squared_error(real20[i, :, :, j], input2[i, :, :, j]))
                        psnr_i2 += skm.peak_signal_noise_ratio(real20[i, :, :, j], input2[i, :, :, j],
                                                               data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min())
                        ssim_i2 += skm.structural_similarity(real20[i, :, :, j], input2[i, :, :, j], multichannel=False,
                                                             data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min())

                    if skm.peak_signal_noise_ratio(real20[i, :, :, j], predicted20[i, :, :, j],
                                                   data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min()) > 300:
                        kb += 1
                    elif skm.peak_signal_noise_ratio(real20[i, :, :, j], predicted20[i, :, :, j],
                                                     data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min()) < 0:
                        kb += 1
                    elif real20[i, :, :, j].max() - real20[i, :, :, j].min() == 0:
                        kb += 1
                    else:
                        rmse2 += np.sqrt(skm.mean_squared_error(real20[i, :, :, j], predicted20[i, :, :, j]))
                        psnr2 += skm.peak_signal_noise_ratio(real20[i, :, :, j], predicted20[i, :, :, j],
                                                             data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min())
                        ssim2 += skm.structural_similarity(real20[i, :, :, j], predicted20[i, :, :, j], multichannel=False,
                                                           data_range=real20[i, :, :, j].max() - real20[i, :, :, j].min())

                    if j < real60.shape[3]:
                        if skm.peak_signal_noise_ratio(real60[i, :, :, j], input6[i, :, :, j],
                                                       data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min()) > 300:
                            kc += 1
                        elif skm.peak_signal_noise_ratio(real60[i, :, :, j], input6[i, :, :, j],
                                                         data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min()) < 0:
                            kc += 1
                        elif real60[i, :, :, j].max() - real60[i, :, :, j].min() == 0:
                            kc += 1
                        else:
                            rmse_i6 += np.sqrt(skm.mean_squared_error(real60[i, :, :, j], input6[i, :, :, j]))
                            psnr_i6 += skm.peak_signal_noise_ratio(real60[i, :, :, j], input6[i, :, :, j],
                                                                   data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min())
                            ssim_i6 += skm.structural_similarity(real60[i, :, :, j], input6[i, :, :, j], multichannel=False,
                                                                 data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min())

                        if skm.peak_signal_noise_ratio(real60[i, :, :, j], predicted60[i, :, :, j],
                                                       data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min()) > 300:
                            kd += 1
                        elif skm.peak_signal_noise_ratio(real60[i, :, :, j], predicted60[i, :, :, j],
                                                         data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min()) < 0:
                            kd += 1
                        elif real60[i, :, :, j].max() - real60[i, :, :, j].min() == 0:
                            kd += 1
                        else:
                            rmse6 += np.sqrt(skm.mean_squared_error(real60[i, :, :, j], predicted60[i, :, :, j]))
                            psnr6 += skm.peak_signal_noise_ratio(real60[i, :, :, j], predicted60[i, :, :, j],
                                                                 data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min())
                            ssim6 += skm.structural_similarity(real60[i, :, :, j], predicted60[i, :, :, j], multichannel=False,
                                                               data_range=real60[i, :, :, j].max() - real60[i, :, :, j].min())
            m += 1

    print(ka)
    print(kb)
    print(kc)
    print(kd)
    print(len(val_loader.dataset))
    rmse_i2 /= ((real20.shape[3] * 110) - ka)
    psnr_i2 /= ((real20.shape[3] * 110) - ka)
    ssim_i2 /= ((real20.shape[3] * 110) - ka)
    rmse2 /= ((real20.shape[3] * 110) - kb)
    psnr2 /= ((real20.shape[3] * 110) - kb)
    ssim2 /= ((real20.shape[3] * 110) - kb)
    rmse_i6 /= ((real60.shape[3] * 110) - kc)
    psnr_i6 /= ((real60.shape[3] * 110) - kc)
    ssim_i6 /= ((real60.shape[3] * 110) - kc)
    rmse6 /= ((real60.shape[3] * 110) - kd)
    psnr6 /= ((real60.shape[3] * 110) - kd)
    ssim6 /= ((real60.shape[3] * 110) - kd)
    val_loss2 /= m
    val_loss6 /= m
    tb.add_scalar("Loss_val2", val_loss2, epoch)
    tb.add_scalar("RMSE_val2", rmse2, epoch)
    tb.add_scalar("PSNR_val2", psnr2, epoch)
    tb.add_scalar("SSIM_val2", ssim2, epoch)
    tb.add_scalar("Loss_val6", val_loss6, epoch)
    tb.add_scalar("RMSE_val6", rmse6, epoch)
    tb.add_scalar("PSNR_val6", psnr6, epoch)
    tb.add_scalar("SSIM_val6", ssim6, epoch)
    tb.add_scalar("Loss_val", (val_loss6*0.5 + val_loss2*0.5), epoch)
    tb.add_scalar("RMSE_val", (rmse6*0.5 + rmse2*0.5), epoch)
    tb.add_scalar("PSNR_val", (psnr6*0.5 + psnr2*0.5), epoch)
    tb.add_scalar("SSIM_val", (ssim6*0.5 + ssim2*0.5), epoch)

    print('\nValidation set (x2): Average values between Input and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f})\n'.format(rmse_i2, psnr_i2, ssim_i2))

    print('\nValidation set (x2): Average values between Output and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f}), Loss: {:.6f}\n'.format(rmse2, psnr2, ssim2, val_loss2))

    print('\nValidation set (x6): Average values between Input and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f})\n'.format(rmse_i6, psnr_i6, ssim_i6))

    print('\nValidation set (x6): Average values between Output and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f}), Loss: {:.6f}\n'.format(rmse6, psnr6, ssim6, val_loss6))

    print('\nValidation set total: Average values between Input and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f})\n'.format((rmse_i6 + rmse_i2) / 2, (psnr_i6 + psnr_i2) / 2, (ssim_i6 + ssim_i2) / 2))

    print('\nValidation set total: Average values between Output and Target -->\nRMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f}), Loss: {:.6f}\n'.format((rmse6 + rmse2) / 2, (psnr6 + psnr2) / 2, (ssim6 + ssim2) / 2,
                                                   (val_loss6 + val_loss2) / 2))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch TFG Net')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=40, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--crop-size', type=int, default=120, metavar='N',
                        help='crop size (default: 120')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    # Creating data indices for training and validation splits:
    dataset_size = len(set_ds)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    print(split)
    print(dataset_size - split)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(set_ds, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    val_loader = DataLoader(set_ds, batch_size=args.test_batch_size, sampler=valid_sampler, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.004)

    scheduler = StepLR(optimizer, step_size=50, gamma=args.gamma)
    model = model.type(dst_type=torch.float32)

    # get some random training images
    # dataiter = iter(train_loader)
    # hr, lr, target = dataiter.next()

    # # visualize the model
    # tb.add_graph(model, (hr, lr))

    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, model, device, optimizer, epoch)
        if epoch % 5 == 0:
            validation(args, val_loader, model, device, epoch)
        if epoch == args.epochs:
            test(args, test_loader, model, device, epoch)
            torch.save(model.state_dict(), "net_both.pt")
        if epoch % 30 == 0:
            torch.save(model.state_dict(), "net_both_x.pt")
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "net_both.pt")

    tb.close()


if __name__ == '__main__':
    main()
