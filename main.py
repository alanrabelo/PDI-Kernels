from convolution import Convolution

convolution = Convolution()
image = convolution.load_image('lena_low.jpg')
image = convolution.salt_and_pepper(image)
convolution.filter_average(image)