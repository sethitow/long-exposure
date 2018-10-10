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

for x in range(0, img_shape[0]):
	for y in range(0, img_shape[1]):
		print(f"{x}, {y}")
		data = copy.deepcopy(videodata)
		tuples = []
		rms = []
		for frame in data.nextFrame():
			tuples.append(frame[x,y])
			rms.append(np.sqrt(np.mean(np.square(frame[x,y]))))
		value = tuples[rms.index(max(rms))]
		stack[x,y] = value
		print(value)
				

stack = stack / num_of_frames / 255

ski.imsave('test.png', stack)

