import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import earthpy.plot as ep

from PIL import Image
import skimage.io as skio

import rasterio
from rasterio import plot as rio_plot
import os

# fuente: https://www.hatarilabs.com/ih-en/sentinel2-images-explotarion-and-processing-with-python-and-rasterio

f_test = open('S2_tiles_testing.txt', 'r')
f_test_l1c =open('l1c_code.txt', 'r')

for i in f_test_l1c:
	im_name = f_test.readline()
	date = im_name[11:26]
	code = im_name[38:44]

	root_path = "/home/usuaris/imatge/lsalgueiro/projects/remote_sensing/SpaceNet/imagenes_sentinel/"+im_name[0:65]

	codigo_l1c = i[0:34]

	imagePath = root_path+"/GRANULE/"+codigo_l1c+"/IMG_DATA/"
	band_prefix = code+"_"+date

	band2 = rasterio.open(imagePath+band_prefix+'_B02.jp2', driver='JP2OpenJPEG')  #blue
	band3 = rasterio.open(imagePath+band_prefix+'_B03.jp2', driver='JP2OpenJPEG')  #green
	band4 = rasterio.open(imagePath+band_prefix+'_B04.jp2', driver='JP2OpenJPEG')  #red
	band8 = rasterio.open(imagePath+band_prefix+'_B08.jp2', driver='JP2OpenJPEG')  #nir
	# band5 = rasterio.open(imagePath+band_prefix+'_B05.jp2', driver='JP2OpenJPEG')  #nir
	# band6 = rasterio.open(imagePath+band_prefix+'_B06.jp2', driver='JP2OpenJPEG')  #nir
	# band7 = rasterio.open(imagePath+band_prefix+'_B07.jp2', driver='JP2OpenJPEG')  #nir
	# band8a = rasterio.open(imagePath+band_prefix+'_B8A.jp2', driver='JP2OpenJPEG')  #nir
	# band11 = rasterio.open(imagePath+band_prefix+'_B11.jp2', driver='JP2OpenJPEG')  #nir
	# band12 = rasterio.open(imagePath+band_prefix+'_B12.jp2', driver='JP2OpenJPEG')  #nir

	# rio_plot.show(band2)

	# EXPORT RASTER TIFF
	raster = rasterio.open("/mnt/gpid07/users/oriol.esquena/images_sentinel/test/"+"10m/"+date+".tiff", "w", driver="Gtiff",
	                       width=band2.width, height=band2.height,
	                       count=4, crs=band2.crs, transform=band2.transform,
	                       dtype=band2.dtypes[0])
	raster.write(band4.read(1), 1)  # write band4 -red- in position 1
	raster.write(band3.read(1), 2)  # write band3 -green- in position 2
	raster.write(band2.read(1), 3)  # write band2 -blue- in position 3
	raster.write(band8.read(1), 4)  # write band8 -nir- in position 4
	# raster.write(band5.read(1), 1)  # write band5 -- in position 5
	# raster.write(band6.read(1), 2)  # write band6 -- in position 6
	# raster.write(band7.read(1), 3)  # write band7 -- in position 7
	# raster.write(band8a.read(1), 4)  # write band8a -- in position 8
	# raster.write(band11.read(1), 5)  # write band11 -- in position 9
	# raster.write(band12.read(1), 6)  # write band12 -- in position 10

	raster.close()

	print(band3.width)

f_test.close()
f_test_l1c.close()


f_train = open('S2_tiles_training.txt', 'r')
f_train_l1c =open('l1c_code_train.txt', 'r')
# training tiles
for i in f_train_l1c:
	im_name = f_train.readline()
	date = im_name[11:26]
	code = im_name[38:44]

	root_path = "/home/usuaris/imatge/lsalgueiro/projects/remote_sensing/SpaceNet/imagenes_sentinel/"+im_name[0:65]

	codigo_l1c = i[0:34]

	imagePath = root_path+"/GRANULE/"+codigo_l1c+"/IMG_DATA/"
	band_prefix = code+"_"+date

	band2 = rasterio.open(imagePath+band_prefix+'_B02.jp2', driver='JP2OpenJPEG')  #blue
	band3 = rasterio.open(imagePath+band_prefix+'_B03.jp2', driver='JP2OpenJPEG')  #green
	band4 = rasterio.open(imagePath+band_prefix+'_B04.jp2', driver='JP2OpenJPEG')  #red
	band8 = rasterio.open(imagePath+band_prefix+'_B08.jp2', driver='JP2OpenJPEG')  #nir
	# band5 = rasterio.open(imagePath+band_prefix+'_B05.jp2', driver='JP2OpenJPEG')  #nir
	# band6 = rasterio.open(imagePath+band_prefix+'_B06.jp2', driver='JP2OpenJPEG')  #nir
	# band7 = rasterio.open(imagePath+band_prefix+'_B07.jp2', driver='JP2OpenJPEG')  #nir
	# band8a = rasterio.open(imagePath+band_prefix+'_B8A.jp2', driver='JP2OpenJPEG')  #nir
	# band11 = rasterio.open(imagePath+band_prefix+'_B11.jp2', driver='JP2OpenJPEG')  #nir
	# band12 = rasterio.open(imagePath+band_prefix+'_B12.jp2', driver='JP2OpenJPEG')  #nir

	# rio_plot.show(band2)

	# EXPORT RASTER TIFF
	raster = rasterio.open("/mnt/gpid07/users/oriol.esquena/images_sentinel/train/"+"10m/"+date+".tiff", "w", driver="Gtiff",
	                       width=band2.width, height=band2.height,
	                       count=4, crs=band2.crs, transform=band2.transform,
	                       dtype=band2.dtypes[0])
	raster.write(band4.read(1), 1)  # write band4 -red- in position 1
	raster.write(band3.read(1), 2)  # write band3 -green- in position 2
	raster.write(band2.read(1), 3)  # write band2 -blue- in position 3
	raster.write(band8.read(1), 4)  # write band8 -nir- in position 4
	# raster.write(band5.read(1), 1)  # write band5 -- in position 5
	# raster.write(band6.read(1), 2)  # write band6 -- in position 6
	# raster.write(band7.read(1), 3)  # write band7 -- in position 7
	# raster.write(band8a.read(1), 4)  # write band8a -- in position 8
	# raster.write(band11.read(1), 5)  # write band11 -- in position 9
	# raster.write(band12.read(1), 6)  # write band12 -- in position 10

	raster.close()

	print(band4.width)

f_train.close()
f_train_l1c.close()
