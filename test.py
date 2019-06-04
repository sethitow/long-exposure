import skvideo.io as skv
import skvideo.datasets
import skimage.io as ski
import numpy as np
import copy
import numba
from numba import jit

print('Opening video file')
videodata = skv.vread('spxl.mov')
# videodata = skv.vread(skvideo.datasets.bigbuckbunny())
shape = videodata.shape
num_of_frames = shape[0]
img_shape = shape[1:]

print(f"Image size: {img_shape}")
print(f"Number of Frames: {num_of_frames}")


def rms(x):
	return np.sqrt(np.mean(np.square(x)))

@jit(nopython=True)
def get_value(tuples):
	means = []
	for item in tuples:
		means.append(np.mean(item))
	return tuples[np.argmax(means)]

def run(img_shape, videodata):
	stack = np.zeros(img_shape)
	iterations = img_shape[0] * img_shape[1]
	counter = 0
	for x in range(0, img_shape[0]):
		for y in range(0, img_shape[1]):
			counter = counter + 1
			print(counter/iterations)
			print(numba.typeof(videodata[:,x,y,:]))
			print("gothere")
			stack[x,y] = get_value(videodata[:,x,y,:])
stack = run(img_shape, videodata)	
stack = stack / 255

ski.imsave('test.png', stack)

