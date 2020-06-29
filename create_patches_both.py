import skimage.io as skio
from skimage.transform import resize
import sklearn.feature_extraction.image as skfi
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


f_test = open('S2_tiles_testing.txt', 'r')
save_test_path = "/mnt/gpid07/users/oriol.esquena/patches_sentinel/test_all/"

for i in f_test:
    print(i)
    im_name = i
    date = im_name[11:26]

    root_path = "/mnt/gpid07/datasets/remote_sensing/imagenes_oriol/images_sentinel_tiff/test/"

    # read the image and normalize its data
    im10 = skio.imread(root_path+"10m/"+date+".tiff")
    im20 = skio.imread(root_path+"20m/"+date+".tiff")
    im60 = skio.imread(root_path+"60m/"+date+".tiff")

    # create patches out of the image
    j = 0
    k = 0
    # num_patches = 180
    num_patches = 25
    num_patches_per_line = 5
    size_im60 = 1830
    size_patch_10 = 366
    size_im120 = 915
    size_patch_20 = 183
    size_im360 = 305
    size_patch_60 = 61
    channels10 = 4
    channels20 = 6
    channels60 = 2
    # max_pixel = np.round((im10.shape[0] - size_im10)/6).astype(dtype=np.int)

    gauss10 = gaussian_filter(im10, sigma=1/6)
    gauss20 = gaussian_filter(im20, sigma=1/6)
    gauss60 = gaussian_filter(im60, sigma=1/6)
    gauss_t20 = gaussian_filter(im20, sigma=1/3)
    rs10 = resize(gauss10, (size_im60, size_im60), preserve_range=True)
    rs20 = resize(gauss20, (size_im120, size_im120), preserve_range=True)
    rs60 = resize(gauss60, (size_im360, size_im360), preserve_range=True)
    rs_t20 = resize(gauss_t20, (size_im60, size_im60), preserve_range=True)

    patches10 = np.ndarray((num_patches, size_patch_10, size_patch_10, channels10))
    patches20 = np.ndarray((num_patches, size_patch_20, size_patch_20, channels20))
    patches60 = np.ndarray((num_patches, size_patch_60, size_patch_60, channels60))
    patches20_target = np.ndarray((num_patches, size_patch_10, size_patch_10, channels20))
    patches60_target = np.ndarray((num_patches, size_patch_10, size_patch_10, channels60))

    x10 = 0
    x20 = 0
    x60 = 0
    y10 = 0
    y20 = 0
    y60 = 0
    im = 0

    for j in range(num_patches_per_line):
        for k in range(num_patches_per_line):
            patches10[im] = rs10[x10:(x10+size_patch_10), y10:(y10+size_patch_10), :]
            patches20[im] = rs20[x20:(x20+size_patch_20), y20:(y20+size_patch_20), :]
            patches60[im] = rs60[x60:(x60+size_patch_60), y60:(y60+size_patch_60), :]
            patches20_target[im] = rs_t20[x10:(x10+size_patch_10), y10:(y10+size_patch_10), :]
            patches60_target[im] = im60[x10:(x10+size_patch_10), y10:(y10+size_patch_10), :]

            x10 += size_patch_10
            x20 += size_patch_20
            x60 += size_patch_60
            im += 1

        x10 = 0
        x20 = 0
        x60 = 0
        y10 += size_patch_10
        y20 += size_patch_20
        y60 += size_patch_60

    np.save(save_test_path+"test_resized60_"+date+".npy", patches10)
    np.save(save_test_path+"test_resized120_"+date+".npy", patches20)
    np.save(save_test_path + "test_resized360_" + date + ".npy", patches60)
    np.save(save_test_path+"real20_target_test_"+date+".npy", patches20_target)
    np.save(save_test_path + "real60_target_test_" + date + ".npy", patches60_target)

f_test.close()
print("Test finished")

save_train_path = "/mnt/gpid07/users/oriol.esquena/patches_sentinel/train_60/"
f_train = open('S2_tiles_training_2.txt', 'r')

for i in f_train:
    im_name = i
    date = im_name[11:26]

    root_path = "/mnt/gpid07/datasets/remote_sensing/imagenes_oriol/images_sentinel_tiff/train/"

    # read the image and normalize its data
    im10 = skio.imread(root_path+"10m/"+date+".tiff")
    im20 = skio.imread(root_path+"20m/"+date+".tiff")
    im60 = skio.imread(root_path+"60m/"+date+".tiff")

    # create patches out of the image

    j = 0
    k = 0
    # num_patches = 180
    num_patches = 25
    num_patches_per_line = 5
    size_im60 = 1830
    size_patch_10 = 366
    size_im120 = 915
    size_patch_20 = 183
    size_im360 = 305
    size_patch_60 = 61
    channels10 = 4
    channels20 = 6
    channels60 = 2
    # max_pixel = np.round((im10.shape[0] - size_im10)/6).astype(dtype=np.int)

    gauss10 = gaussian_filter(im10, sigma=1/6)
    gauss20 = gaussian_filter(im20, sigma=1/6)
    gauss60 = gaussian_filter(im60, sigma=1/6)
    gauss_t20 = gaussian_filter(im20, sigma=1/3)
    rs10 = resize(gauss10, (size_im60, size_im60), preserve_range=True)
    rs20 = resize(gauss20, (size_im120, size_im120), preserve_range=True)
    rs60 = resize(gauss60, (size_im360, size_im360), preserve_range=True)
    rs_t20 = resize(gauss_t20, (size_im60, size_im60), preserve_range=True)

    patches10 = np.ndarray((num_patches, size_patch_10, size_patch_10, channels10))
    patches20 = np.ndarray((num_patches, size_patch_20, size_patch_20, channels20))
    patches60 = np.ndarray((num_patches, size_patch_60, size_patch_60, channels60))
    patches20_target = np.ndarray((num_patches, size_patch_10, size_patch_10, channels20))
    patches60_target = np.ndarray((num_patches, size_patch_10, size_patch_10, channels60))

    x10 = 0
    x20 = 0
    x60 = 0
    y10 = 0
    y20 = 0
    y60 = 0
    im = 0

    for j in range(num_patches_per_line):
        for k in range(num_patches_per_line):
            patches10[im] = rs10[x10:(x10+size_patch_10), y10:(y10+size_patch_10), :]
            patches20[im] = rs20[x20:(x20+size_patch_20), y20:(y20+size_patch_20), :]
            patches60[im] = rs60[x60:(x60+size_patch_60), y60:(y60+size_patch_60), :]
            patches20_target[im] = rs_t20[x10:(x10+size_patch_10), y10:(y10+size_patch_10), :]
            patches60_target[im] = im60[x10:(x10+size_patch_10), y10:(y10+size_patch_10), :]

            x10 += size_patch_10
            x20 += size_patch_20
            x60 += size_patch_60
            im += 1

        x10 = 0
        x20 = 0
        x60 = 0
        y10 += size_patch_10
        y20 += size_patch_20
        y60 += size_patch_60

    np.save(save_train_path+"input10_resized60_"+date+".npy", patches10)
    np.save(save_train_path+"input20_resized120_"+date+".npy", patches20)
    np.save(save_train_path+"input60_resized360_"+date+".npy", patches60)
    np.save(save_train_path+"real20_target_"+date+".npy", patches20_target)
    np.save(save_train_path+"real60_target_"+date+".npy", patches60_target)

print("Train finished")
