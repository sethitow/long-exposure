import skvideo.io as skv
import skvideo.datasets
import skimage.io as ski
import numpy as np
import copy

print('Opening video file')
# videodata = skv.FFmpegReader('spxl.mov')
videodata = skv.FFmpegReader(skvideo.datasets.bigbuckbunny())
shape = videodata.getShape()
num_of_frames = shape[0]
img_shape = shape[1:]

print(f"Image size: {img_shape}")
print(f"Number of Frames: {num_of_frames}")

stack = np.zeros(img_shape)

for frame in videodata:
	stack = stack + frame
				
stack = stack / num_of_frames / 255

ski.imsave('test.png', stack)

