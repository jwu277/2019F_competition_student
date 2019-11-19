#!/usr/bin/env python

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
from license_processing import LicenseProcessor

lp = LicenseProcessor()
print(os.listdir('images_pre'))
for filename in os.listdir('images_pre'):
    print(filename + " is testing plate detection")
    lp.license_finder(cv2.imread('images_pre/'+filename))

for filename in os.listdir('cropped_plates'):
    print(filename +" is testing character parsing")
    chars = lp.parse_plate(cv2.imread('cropped_plates/'+filename))
    