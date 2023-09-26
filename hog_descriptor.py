from skimage.color import rgb2gray
from scipy.ndimage import sobel
import numpy as np

# resize the input image to the nearest multiple of tile_size.
def resize_image(image, tile_size):
    height, width = image.shape[:2]
    height -= height % tile_size
    width  -= width % tile_size
    return image[:height,:width]

# compute the gradients of an image.
def get_gradients(image):
    if len(image.shape) == 3 or image.shape[-1] == 3:
        image = rgb2gray(image)
    dx = scipy.ndimage.sobel(image, axis=1)
    dy = scipy.ndimage.sobel(image, axis=0)
    return dy, dx

# compute the magnitude of gradients.
def get_magnitude(gradient_h, gradient_w):
    return np.hypot(gradient_w,gradient_h)

# compute the gradient orientations
def get_orientations(gradient_h, gradient_w):
    return np.degrees(np.arctan2(gradient_h, gradient_w))

# hog descriptor
def hog_descriptor(image, tile_size=8, num_bins=9, epsilon=1e-4):
    """
    Compute Histogram of Oriented Gradients (HOG) features for an input image.

    Parameters:
    - image (numpy.ndarray): The input image for which HOG features are computed.
    - tile_size (int, optional): The size of each HOG cell/tile in pixels. Default is 8.
    - num_bins (int, optional): The number of bins for the histogram of gradient orientations. Default is 9.
    - epsilon (float, optional): A small constant to prevent numerical instability. Default is 1e-4.
    
    Returns:
    - hog_features: numpy.ndarray
        The computed HOG features for the image.
    """
    # resize image to conform to tile size
    image = resize_image(image, tile_size) 
    # get gradients in both directions
    gradient_y, gradient_x = get_gradients(image+epsilon)  
    # compute magnitude of gradients
    magnitudes = get_magnitude(gradient_y, gradient_x) 
    # compute orientation of gradients in degrees
    orientations = get_orientations(gradient_y, gradient_x)

    # Create dimensions for the HOG histogram
    _dims = (np.array(image.shape[:2])//tile_size)*tile_size
    dims = *_dims, num_bins

    _dims_hog = (np.array(image.shape[:2])//tile_size)
    dims_hog = *_dims_hog, num_bins

    # initialize an array to store HOG histograms
    hogs = np.zeros(dims_hog, dtype=np.float32)  

    # scale the orientations and calculate indices for the two bins used for interpolation
    scaled = (orientations/180*num_bins)%num_bins
    vals_bin_2, indx_1 = np.modf(scaled)
    indx_1 = indx_1.astype(int)
    vals_bin_1 = 1-vals_bin_2
    indx_2 = ((indx_1 + 1) % num_bins).astype(int)


    # populate the HOG histogram with weighted gradient magnitudes
    for y, x in np.ndindex(*dims[:2]):
        hogs[y // tile_size, x // tile_size, indx_1[y, x]] += vals_bin_1[y, x] * magnitudes[y, x]
        hogs[y // tile_size, x // tile_size, indx_2[y, x]] += vals_bin_2[y, x] * magnitudes[y, x]

    # initialize an array for HOG features

    hog_features = np.empty(shape=(*_dims_hog-1,num_bins*2*2))  

    for i in range(hogs.shape[0]-1):
        for j in range(hogs.shape[1]-1):
            block = hogs[i:i+2, j:j+2, ]
            block_raveled = block.ravel()
            bin_norm = np.linalg.norm(block_raveled)
            hog_features[i,j] = (block_raveled/bin_norm)

    return hog_features
