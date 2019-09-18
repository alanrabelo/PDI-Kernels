from convolution import Convolution
import numpy as np
from matplotlib import pyplot as plt

kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
          ])

sobel_H = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
          ])

sobel_V = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
          ])

laplacian_1 = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
          ])

laplacian_2 = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
          ])

prewitt_1 = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
          ])

prewitt_2 = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
          ])

# images = ['lena_original']
images = ['Lena', 'Zebra']

for image_name in images:
    convolution = Convolution(kernel_size=len(kernel))
    image = convolution.load_image('%s' % image_name)
    # convolution.equalize_histogram(image, image_name)
    convolution.multi_limiarize(image, image_name, thresholds=[80, 160])

    # convolution.prewitt(image, 'PREWITT')
    # convolution.sobel(image, 'SOBEL')
    # convolution.laplacian(image, 'LAPLACE')
    #
    # image = convolution.load_image('%s' % image_name)
    #
    # image = convolution.salt_and_pepper(image)
    # convolution.filter_average(image, kernel)
    # convolution.filter_median(image, kernel)

