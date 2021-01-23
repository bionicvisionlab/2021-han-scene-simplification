import math
import numpy as np

# remap x to a certain range
def restrict_x_value(d, max_d, min_d, target_max_d, target_min_d):
    k = (target_min_d - target_max_d) / (min_d - max_d)
    b = target_min_d - k * min_d
    return k*d+b

# naive linear map; remap x to [2, 10] and then y = kx + b fit to (2, max_y) and (10, min_y)
def depth_linear_map(d, max_depth, min_depth, min_y, max_y):
    d = restrict_x_value(d, max_depth, min_depth, 10, 2)
    k = (min_y - max_y) / (10 - 2)
    b = min_y - k * 10
    return k*d+b

# quadratic map; remap x to [2, 10] and then y = a(x-h)^2 + k fit to (2, max_y) and (10, min_y) as the vertex
def depth_quadratic_map(depth_map, max_depth, min_depth, min_y, max_y):
    depth_map = restrict_x_value(depth_map, max_depth, min_depth, 10, 2)
    a = (max_y - min_y) / math.pow(2 - 10, 2)
    return a * np.power(depth_map - 10, 2) + min_y

# flipped quadratic map; remap x to [2, 10] and then y = a(x-h)^2 + k fit to (10, min_y) and (2, max_y) as the vertex
def depth_flipped_quadratic_map(depth_map, max_depth, min_depth, min_y, max_y):
    depth_map = restrict_x_value(depth_map, max_depth, min_depth, 10, 2)
    a = (min_y - max_y) / math.pow(10 - 2, 2)
    return a * np.power(depth_map - 2, 2) + max_y

# exponential map; remap x to [2, 10] and y = a*e^(-x) + b fit to (2, max_y) and (10, min_y)
def depth_exponential_map(depth_map, max_depth, min_depth, min_y, max_y):
    depth_map = restrict_x_value(depth_map, max_depth, min_depth, 10, 2)
    a = (max_y - min_y) / (math.exp(-2) - math.exp(-10))
    b = max_y - a * math.exp(-2)
    return a * np.exp(-depth_map) + b

# flipped exponential map; remap x to [2, 10] and y = a*e^x + b fit to (2, max_y) and (10, min_y)
def depth_flipped_exponential_map(depth_map, max_depth, min_depth, min_y, max_y):
    depth_map = restrict_x_value(depth_map, max_depth, min_depth, 10, 2)
    a = (max_y - min_y) / (math.exp(2) - math.exp(10))
    b = max_y - a * math.exp(2)
    return a * np.exp(depth_map) + b

# generate images with depths encoded by brightness
# sidewalk_mask is used to indicated the region to retain; all the pixles outside the mask will be set to 0. 
# In the current usage, sidewalk_mask is set to be all 1's which means everything is retained. 
def gen_image_brightness(depth_map, sidewalk_mask, mode="linear", min_brightness=50, max_brightness=255):
    if np.count_nonzero(sidewalk_mask) == 0:
        return np.zeros((sidewalk_mask.shape[0], sidewalk_mask.shape[1], 3)).astype(np.uint8)
    max_depth = depth_map[sidewalk_mask].max()
    min_depth = depth_map[sidewalk_mask].min()
    if mode == "quadratic":
        one_channel = depth_quadratic_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    elif mode == "exponential":
        one_channel = depth_exponential_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    elif mode == "flipped_quadratic":
        one_channel = depth_flipped_quadratic_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    elif mode == "flipped_exponential":
        one_channel = depth_flipped_exponential_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    else:
        one_channel = depth_linear_map(depth_map, max_depth, min_depth, min_brightness, max_brightness)
    complement_sidewalk_mask = sidewalk_mask == False
    one_channel[complement_sidewalk_mask] = 0
    im = np.array([one_channel] * 3).astype(np.uint8)
    return np.transpose(im, (1, 2, 0))