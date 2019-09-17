from convolution import Convolution

convolution = Convolution()
image = convolution.load_image('Sample Images/lena_low.jpg')
image = convolution.salt_and_pepper(image)
convolution.filter_average(image)
# convolution.filter_median(image)
