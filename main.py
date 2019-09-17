from convolution import Convolution
import numpy as np

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
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
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

images = ['lena_original']
# images = ['lena_original', 'zebra', 'manhattan_medium']

for image in images:
    convolution = Convolution(kernel_size=len(kernel))
    image = convolution.load_image('%s' % image)
    image = convolution.filter_average(image, kernel)
    image = convolution.filter_average(image, kernel)

    convolution.convolve(image, sobel_H, 'sobel_H')
    convolution.convolve(image, sobel_V, 'sobel_V')
    convolution.convolve(image, laplacian_1, 'laplacian1')
    convolution.convolve(image, laplacian_2, 'laplacian2')
    convolution.convolve(image, prewitt_1, 'prewitt_1')
    convolution.convolve(image, prewitt_2, 'prewitt_2')
    image = convolution.salt_and_pepper(image)
    convolution.filter_average(image, kernel)
    convolution.filter_median(image, kernel)

