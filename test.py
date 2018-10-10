import skvideo.io as skv
import skvideo.datasets
import skimage.io as ski
import numpy as np
import copy

print('Opening video file')
videodata = skv.FFmpegReader('spxl.mov')
# videodata = skv.vread(skvideo.datasets.bigbuckbunny())
shape = videodata.shape
num_of_frames = shape[0]
img_shape = shape[1:]

print(f"Image size: {img_shape}")
print(f"Number of Frames: {num_of_frames}")

stack = np.zeros(img_shape)

def rms(x):
	return np.sqrt(np.mean(np.square(x)))
vrms = np.vectorize(rms)

iterations = img_shape[0] * img_shape[1]
counter = 0
for x in range(0, img_shape[0]):
	for y in range(0, img_shape[1]):
		counter = counter + 1
		print(counter/iterations)
		tuples = videodata[:,x,y,:]
		t_rms = np.apply_along_axis(rms, 1, tuples) 
		value = tuples[np.argmax(t_rms)]	
		stack[x,y] = value
						
stack = stack / 255

ski.imsave('test.png', stack)

