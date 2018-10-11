import skvideo.io as skv
import skvideo.datasets
import skimage.io as ski
import numpy as np
import numba
from numba import jit, vectorize


print('Opening video file')
# videodata = skv.vread('spxl.mov')
videodata = skv.vread(skvideo.datasets.bigbuckbunny())
shape = videodata.shape
num_of_frames = shape[0]
img_shape = shape[1:]

print(f"Image size: {img_shape}")
print(f"Number of Frames: {num_of_frames}")

# @vectorize(target='parallel', nopython=True)
def rms(x):
	return np.sqrt(np.mean(np.square(x)))
vrms = np.vectorize(rms)

@jit(nopython=True)
def max_from_rms(a, axis):
	return a[np.argmax(rms(a))]	

stack = np.apply_over_axes(max_from_rms, videodata, [1,2])

stack = stack / 255

ski.imsave('test.png', stack)

