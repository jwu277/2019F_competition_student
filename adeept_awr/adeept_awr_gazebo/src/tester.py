#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from license_processing import LicenseProcessor
from CNN_model import CNNModel

lp = LicenseProcessor()

CNN = CNNModel()
CNN.train()
#CNN.test_nn()

# for filename in os.listdir('images_pre'):
#     print(filename + " is testing plate detection")
#     lp.license_finder(cv2.imread('images_pre/'+filename))

# # Create a test set with the cropped images in the folder below
# for filename in os.listdir('cropped_plates'):
#     lp.parse_plate_test_set(filename)
    

# for filename in os.listdir('cropped_plates'):
#     img = cv2.imread('cropped_plates/'+filename)
#     plt.imshow(img)
#     plt.show()