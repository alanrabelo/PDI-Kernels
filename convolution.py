from PIL import Image
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt

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

    def filter_gaussian(self, image):

        kernel = [
                   [ 1.0278445 ,  4.10018648,  6.49510362,  4.10018648,  1.0278445 ],
                   [ 4.10018648, 16.35610171, 25.90969361, 16.35610171,  4.10018648],
                   [ 6.49510362, 25.90969361, 41.0435344 , 25.90969361,  6.49510362],
                   [ 4.10018648, 16.35610171, 25.90969361, 16.35610171,  4.10018648],
                   [ 1.0278445 ,  4.10018648,  6.49510362,  4.10018648,  1.0278445]
                ]

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

        return self.save_image('GAUSSIAN', new_image)

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
                    result_average = np.sum(result)
                    new_image[row_index + kernel_center, col_index + kernel_center] = result_average

        if np.min(new_image) < 0 or np.max(new_image) > 255:
            print('OOOOpppsss')
        return self.save_image(filename, new_image)

    def prewitt(self, image, filename):

        horizontal = [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
                      ]
        vertical = [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
                      ]
        kernel_size = len(horizontal)
        new_image = np.array(image, dtype='uint8')
        height = len(image)
        width = len(image[0])
        kernel_center = math.floor(len(horizontal) / 2)

        for row_index in range(height-kernel_size+1):
            for col_index in range(width-kernel_size+1):
                    image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size]
                    result_H = horizontal * image_part
                    result_V = vertical * image_part
                    result = (np.sum(result_H)**2 + np.sum(result_V)**2)**0.5
                    new_image[row_index + kernel_center, col_index + kernel_center] = result

        if np.min(new_image) < 0 or np.max(new_image) > 255:
            print('OOOOpppsss')
        return self.save_image(filename, new_image)

    def sobel(self, image, filename):

        horizontal = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
                      ]
        vertical = [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
                      ]
        kernel_size = len(horizontal)
        new_image = np.array(image, dtype='uint8')
        height = len(image)
        width = len(image[0])
        kernel_center = math.floor(len(horizontal) / 2)

        for row_index in range(height-kernel_size+1):
            for col_index in range(width-kernel_size+1):
                    image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size]
                    result_H = horizontal * image_part
                    result_V = vertical * image_part
                    result = (np.sum(result_H)**2 + np.sum(result_V)**2)**0.5
                    new_image[row_index + kernel_center, col_index + kernel_center] = result

        if np.min(new_image) < 0 or np.max(new_image) > 255:
            print('OOOOpppsss')
        return self.save_image(filename, new_image)

    def laplacian(self, image, filename):

        laplacian = [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
          ]

        kernel_size = len(laplacian)
        new_image = np.array(image, dtype='uint8')
        height = len(image)
        width = len(image[0])
        kernel_center = math.floor(len(laplacian) / 2)

        for row_index in range(height-kernel_size+1):
            for col_index in range(width-kernel_size+1):
                    image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size]
                    result = laplacian * image_part
                    result = np.sum(result)
                    new_image[row_index + kernel_center, col_index + kernel_center] = result

        if np.min(new_image) < 0 or np.max(new_image) > 255:
            print('OOOOpppsss')
        return self.save_image(filename, new_image)

    def equalize_histogram(self, image, filename):

        hist, bins = np.histogram(image.ravel(), 256, [0, 256])

        size = len(image) * len(image[0])
        hist_acc = np.array(hist, dtype=float)

        for i in range(1, len(hist_acc)):
            hist_acc[i] += hist_acc[i - 1]

        normalized = hist
        for i in range(len(hist_acc)):
            normalized[i] = 255 * hist_acc[i] / size

        normalized_image = np.array(image)

        for row_index in range(len(image)):
            for col_index in range(len(image[0])):
                value = image[row_index, col_index]
                normalized_image[row_index, col_index] = normalized[value]

        self.save_image('%s_normalized' % filename, normalized_image)
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.title('Histograma da %s' % filename)
        # plt.show()

    def limiarize(self, image, filename, threshold=128):

        new_image = np.array(image)

        for row_index in range(len(image)):
            for col_index in range(len(image[0])):
                value = image[row_index, col_index]
                new_image[row_index, col_index] = 255 if value > threshold else 0

        self.save_image(('%s_limiarized' % filename), new_image)

    def multi_limiarize(self, image, filename, thresholds=None):

        if thresholds is None:
            thresholds = [50]

        thresholds.insert(0, 0)
        thresholds = np.append(thresholds, 255)
        new_image = np.array(image)

        for row_index in range(len(image)):
            for col_index in range(len(image[0])):
                value = image[row_index, col_index]
                next_value = thresholds[0]

                for threshold in thresholds:
                    if threshold < value:
                        next_value = threshold
                    else:
                        break

                new_image[row_index, col_index] = next_value

        print(new_image)
        self.save_image(('%s_multi_limiarized' % filename), new_image)