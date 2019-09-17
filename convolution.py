from PIL import Image
import numpy as np
import random
import math
import os

class Convolution:

    IMAGE_MODE = 'L'

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def load_image(self, filename):
        self.filename = filename
        img = Image.open('Sample Images/' + filename + '.jpg').convert(self.IMAGE_MODE)
        img.load()
        data = np.asarray(img)
        return data

    def save_image(self, filename, image):

        path = 'Outputs/' + self.filename
        if not os.path.exists(path):
            os.mkdir(path)

        image_cropped = image[self.kernel_size:-self.kernel_size, self.kernel_size:-self.kernel_size]

        new_image = Image.fromarray(image_cropped, mode=self.IMAGE_MODE)
        new_image.save('Outputs/%s/%s.jpg' % (self.filename, filename))
        return image

    def scale(self, X, x_min=0, x_max=255):
        nom = (X - X.min()) * (x_max - x_min)
        denom = X.max() - X.min()
        denom = denom + (denom is 0)
        return x_min + np.array(nom / denom)

    def salt_and_pepper(self, image, probability_threshold=0.95):
        new_image = np.pad(image, pad_width=self.kernel_size, mode='edge')
        height = len(image)
        width = len(image[0])

        for row_index in range(height):
            for col_index in range(width):
                    rand = random.uniform(0, 1)
                    if rand > probability_threshold:
                        black_or_white = 0 if bool(random.getrandbits(1)) == 0 else 255
                        new_image[row_index, col_index] = black_or_white

        return self.save_image('salt_and_pepper', new_image)

    def filter_average(self, image, kernel):
        kernel_size = len(kernel)
        new_image = np.array(image)
        height = len(image)
        width = len(image[0])
        kernel_center = math.floor(len(kernel) / 2)

        for row_index in range(height-kernel_size+1):
            for col_index in range(width-kernel_size+1):
                    image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size]
                    result = kernel * image_part
                    result_average = np.average(result)
                    new_image[row_index + kernel_center, col_index + kernel_center] = result_average

        return self.save_image('average', new_image)

    def filter_median(self, image, kernel):

        kernel_size = len(kernel)

        new_image = np.array(image, dtype='uint8')
        height = len(image)
        width = len(image[0])
        kernel_center = math.floor(len(kernel) / 2)

        for row_index in range(height-kernel_size+1):
            for col_index in range(width-kernel_size+1):
                    image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size]
                    result = kernel * image_part
                    result_average = np.median(result)
                    new_image[row_index + kernel_center, col_index + kernel_center] = result_average

        return self.save_image('median', new_image)

    def convolve(self, image, kernel, filename):

        kernel_size = len(kernel)

        new_image = np.array(image, dtype='uint8')
        height = len(image)
        width = len(image[0])
        kernel_center = math.floor(len(kernel) / 2)

        for row_index in range(height-kernel_size+1):
            for col_index in range(width-kernel_size+1):
                    image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size]
                    result = kernel * image_part
                    result_average = np.average(result)
                    new_image[row_index + kernel_center, col_index + kernel_center] = result_average

        return self.save_image(filename, new_image)