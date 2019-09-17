from PIL import Image
import numpy as np
import random
from matplotlib import pyplot as plt

class Convolution:

    def load_image(self, filename):
        img = Image.open(filename).convert('L')
        img.load()
        data = np.asarray(img)
        return data

    def salt_and_pepper(self, image, probability_threshold=0.95):
        new_image = np.array(image)
        height = len(image)
        width = len(image[0])

        for row_index in range(height):
            for col_index in range(width):
                rand = random.uniform(0, 1)

                if rand > probability_threshold:
                    black_or_white = 0 if bool(random.getrandbits(1)) == 0 else 255
                    new_image[row_index][col_index] = black_or_white

        s_n_p = Image.fromarray(new_image, mode='L')
        s_n_p.save('salt_and_pepper.jpg')
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

        for row_index in range(height-kernel_size):
            for col_index in range(width-kernel_size):

                image_part = image[row_index:row_index+kernel_size, col_index:col_index+kernel_size]
                result = np.dot(matrix, image_part)
                result = np.average(result)
                new_image[row_index+1, col_index+1] = result
                print(result)


        s_n_p = Image.fromarray(new_image, mode='L')
        s_n_p.show()
        s_n_p.save('average.jpg')
