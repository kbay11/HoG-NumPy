# HoG Feature Descriptor NumPy

A fairly efficient, (almost) from scratch implementation of Histogram of Gradients feature descriptor in NumPy. Inspired by https://learnopencv.com/histogram-of-oriented-gradients/
Inspired by the following [post](https://learnopencv.com/histogram-of-oriented-gradients/). Block normalisation is done in blocks of 2x2. Can be added as hyper parameter later on. 


## Installation

The entire list of dependencies can be found in requirements.txt
The packages used can be installed with:

```bash
pip install numpy scipy scikit-image
```

## Usage

```python
from skimage.io import imread
from hog_descriptor import hog_descriptor

# Load an image
image = imread('example.jpg')

# Compute HoG features
hog_features = hog_descriptor(image, tile_size=8, num_bins=9, epsilon=1e-4)
```

## License

[MIT](https://choosealicense.com/licenses/mit/)