# image filters

import cv2
import magic
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import re
import os

def get_image_size(path):
    t = magic.from_file(path)
    results = re.findall ('(\d+)\s?x\s?(\d+)', t)
    size = sorted(results, reverse=True)[0]
    return int(size[0]), int(size[1])

mpl.rcParams['text.color'] = 'w'
mpl.rcParams['xtick.color'] = 'w'
mpl.rcParams['ytick.color'] = 'w'
mpl.rcParams['axes.labelcolor'] = 'w'

def _detect_blur_fft(image_path, size=60, plot_output=False):#, thresh=10):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# grab the dimensions of the image and use the dimensions to
	# derive the center (x, y)-coordinates
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))

	# compute the FFT to find the frequency transform, then shift
	# the zero frequency component (i.e., DC component located at
	# the top-left corner) to the center where it will be more
	# easy to analyze
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)

	# check to see if we are visualizing our output
	if plot_output:
		magnitude_plot = 20 * np.log(np.abs(fftShift))

	# zero-out the center of the FFT shift (i.e., remove low
	# frequencies), apply the inverse shift such that the DC
	# component once again becomes the top-left, and then apply
	# the inverse FFT
	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)

	# compute the magnitude spectrum of the reconstructed image,
	# then compute the mean of the magnitude values
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)


	if plot_output:
		# display the original input image
		(fig, ax) = plt.subplots(1, 2, )
		ax[0].imshow(image, cmap="gray")
		ax[0].set_title(os.path.basename(image_path))
		ax[0].set_xticks([])
		ax[0].set_yticks([])
		# display the magnitude image
		ax[1].imshow(magnitude_plot, cmap="gray")
		ax[1].set_title(f"Mag Spectrum: blur={mean:.2f}")
		ax[1].set_xticks([])
		ax[1].set_yticks([])
		plt.show()


	return mean
	# return (mean, mean <= thresh)


def detect_blur_fft(image_path, size=60, plot_output=False):#, thresh=10):
	try:
		return _detect_blur_fft(image_path, size, plot_output)
	except:
		return 0.


def filter_out(path):
    w,h = get_image_size(path)
    if w < 120 or h<120: # Too small
        return True
    if w > h: # landscape
        return True
    if detect_blur_fft(path) < -3: # skip blur image
        return True
    return False




def rename_blur_images(rootdir):
    # determin threshold with precalculated result(xxx_blur_fft.txt)
    #
    import pandas as pd
    from mlutils.image_show import show_images, openImage


    rootdir =r'C:\MLDatas\Facereg\VGGFace2\train'
    csv_path = os.path.join(rootdir + '_blur_fft.txt')
    df = pd.read_csv(csv_path, names=['path', 'blur'], index_col=None)
    display(df)

    maxV = 0
    condition = f'{maxV-1} <= blur and blur < {maxV}'
    condition = f'blur < {maxV}'
    # filtered = df.query('blur < -2.5').sample(n=10, random_state=1004)
    filtered = df.query(condition) #, random_state=1004)
    print(condition, f': {len(filtered)}/{len(df)} , {len(filtered)*100/len(df):.2f}%')

    if 0 and 'verify by human':
        samples = filtered.sample(n=20)
        display(samples)
        image_paths = [os.path.join(rootdir, 'train', row[1].values[0]) for row in samples.iterrows()]
        openImage(image_paths)

    # rename filtered files
    filtered.reset_index()
    for i, row in filtered.iterrows():
        print(row['path'], row['blur'])

        path = os.path.join(rootdir, row['path'])
        try:
            os.rename(path, path+'.blur')
        except Exception as ex:
            print(ex)

# if 0:
# 	rootdir =r'C:\MLDatas\Facereg\VGGFace2\train'
# 	# rootdir =r'C:\MLDatas\Facereg\VGGFace2\test'
# 	print('get blurlevel of image files in', rootdir)

# 	folders = find_folders(rootdir)

# 	fp = open(rootdir+'_blur_fft.txt', 'w')
# 	rootdir_len = len(rootdir)+1
# 	def log_blurlevel(path):
# 		blurlevel = detect_blur_fft(path)
# 		fp.write(f'{path[rootdir_len:]},{blurlevel:.4f}\n')
# 		return False

# 	images = get_image_files(rootdir, folders, log_blurlevel)
# 	fp.close()



# histogram equalized image
def histogram_equalization(img):
	assert img is not None
	if isinstance(img, str):
		with open(img, 'rb') as fp:
			img = Image.open(fp).convert('RGB')
			img = np.array(img)
	elif not hasattr(img, 'shape'):
		img = np.array(img)

	yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
	yuv[:,:,0] = clahe.apply(yuv[:,:,0])
	return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

	hist,bins = np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	return cdf[img]
	plot.imshow(cdf[img])
	plot.set_title('color equ')
	plot.hist(equalized_img.flatten(),256,[0,256], color = 'g')
	plt.xlim([0,256])