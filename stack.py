"""Stack video frames to simulate a long exposure
"""
import time

import skvideo.io as skv
import skimage.io as ski
import numpy as np
from numba import jit


print("Opening video file")
videodata = skv.vread("spxl.mov")
videodata = videodata[::30]
print("Video file opened")

shape = videodata.shape
num_of_frames = shape[0]
img_shape = shape[1:]

print(f"Image size: {img_shape}")
print(f"Number of Frames: {num_of_frames}")


@jit(nopython=True)
def max_from_rms(arr):
    """Returns pixel with the highest RMS value"""
    rmsv = np.zeros((num_of_frames))
    for idx, pixel in enumerate(arr):
        rmsv[idx] = np.sqrt(np.mean(np.square(pixel)))
    return arr[np.argmax(rmsv)]


@jit(nopython=True)
def array_from_coordinates(x_coord: int, y_coord: int, vid: np.ndarray):
    """Returns an ndarray of pixels along the time axis for a give x,y coordinate"""
    arr = np.zeros((num_of_frames, 3))
    for idx, frame in enumerate(vid):
        arr[idx] = frame[x_coord][y_coord]
    return arr


@jit(nopython=True)
def make_img(vid):
    """Generate image stack"""
    x_dim = vid.shape[1]
    y_dim = vid.shape[2]
    stack = np.zeros((x_dim, y_dim, 3))
    for x_coord in range(x_dim):
        for y_coord in range(y_dim):
            pixel_arr = array_from_coordinates(x_coord, y_coord, vid)
            stack[x_coord][y_coord] = max_from_rms(pixel_arr)
    return stack


start = time.perf_counter()
img = make_img(videodata)
end = time.perf_counter()

print(f"Processing time: {end-start} seconds.")

start = time.perf_counter()
ski.imsave("test.png", img)
end = time.perf_counter()
print(f"Image saved in {end-start} seconds.")
