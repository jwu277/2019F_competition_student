import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
from license_processing import license_finder, parse_plate


# for filename in os.listdir('images_pre'):
#     print(filename + " is testing plate detection")
#     license_finder(filename)

for filename in os.listdir('cropped_plates'):
    print(filename +" is testing character parsing")
    parse_plate(filename)