from PIL import Image
import numpy as np
import random
import math

class Convolution:

    IMAGE_MODE = 'L'

    def load_image(self, filename):
        img = Image.open(filename).convert(self.IMAGE_MODE)
        img.load()
        data = np.asarray(img)

        data = np.pad(data, pad_width=1, mode='reflect')
        return data

    def salt_and_pepper(self, image, probability_threshold=0.95):
        new_image = np.array(image)
        height = len(image)
        width = len(image[0])

        for row_index in range(height):
            for col_index in range(width):
                # for dimension_index in range(3):

                    rand = random.uniform(0, 1)

                    if rand > probability_threshold:
                        black_or_white = 0 if bool(random.getrandbits(1)) == 0 else 255
                        # new_image[row_index, col_index, dimension_index] = black_or_white
                        new_image[row_index, col_index] = black_or_white

        s_n_p = Image.fromarray(new_image, mode=self.IMAGE_MODE)
        s_n_p.save('Outputs/salt_and_pepper.jpg')
        return new_image

    def filter_average(self, image):

        matrix = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        kernel_size = len(matrix)

        new_image = np.array(image)
        height = len(image)
        width = len(image[0])
        kernel_center = math.floor(len(matrix) / 2)

        # new_image = np.insert(new_image, 0, new_image[:, :kernel_center, :], axis=1)
        # new_image = np.insert(new_image, 0, new_image[:, -1:, :], axis=1)

        for row_index in range(height-kernel_size):
            for col_index in range(width-kernel_size):
                # for dimension_index in range(3):
                    image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size]
                    # image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size, dimension_index]
                    result = matrix * image_part
                    result_average = np.average(result)
                    new_image[row_index + kernel_center, col_index + kernel_center] = result_average
                    # new_image[row_index + kernel_center, col_index + kernel_center, dimension_index] = result_average

        s_n_p = Image.fromarray(new_image, mode=self.IMAGE_MODE)
        s_n_p.show()
        s_n_p.save('Outputs/average.jpg')

    def filter_median(self, image):

        matrix = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        kernel_size = len(matrix)

        new_image = np.array(image)
        height = len(image)
        width = len(image[0])
        kernel_center = math.floor(len(matrix) / 2)

        for row_index in range(height-kernel_size):
            for col_index in range(width-kernel_size):
                # for dimension_index in range(3):

                    image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size]
                    # image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size, dimension_index]
                    result = matrix * image_part
                    result_average = np.median(result)
                    # new_image[row_index + kernel_center, col_index + kernel_center, dimension_index] = result_average
                    new_image[row_index + kernel_center, col_index + kernel_center] = result_average

        s_n_p = Image.fromarray(new_image, mode=self.IMAGE_MODE)
        s_n_p.show()
        s_n_p.save('Outputs/median.jpg')
