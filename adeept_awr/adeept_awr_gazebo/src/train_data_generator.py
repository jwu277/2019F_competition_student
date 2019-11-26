import numpy as np
import cv2
import os
from random import randint

directory = os.getcwd() 
liscence_plates = directory + "/pictures"
cropped_directory = directory + "/cropped_chars"
width_pixels = 100 #width of the pictures containing a single letter
img_height = 298
img_width = 600
x1 = 50
x2 = 150
x3 = 350
x4 = 450

img_new_height = 100
img_new_width = 100
img_full_height = 220

#test, iterate through all pictures file "pictures"
for filename in os.listdir(liscence_plates):
    file_path = liscence_plates + "/" + filename
    img = cv2.imread(file_path)
    
    

    char1 = filename[6]
    char2 = filename[7]
    char3 = filename[8]
    char4 = filename[9]

    #now with the image, crop it into 4 sections and place them in
    #cropped pictures directory
    crop_img1 = img_gray[0:img_height, x1:x1+width_pixels]
    crop_img2 = img_gray[0:img_height, x2:x2+width_pixels]
    crop_img3 = img_gray[0:img_height, x3:x3+width_pixels]
    crop_img4 = img_gray[0:img_height, x4:x4+width_pixels]

    imgs = [crop_img1, crop_img2, crop_img3, crop_img4 ]
    chars = [char1, char2, char3, char4]

    for i in range(4):
        img_new = cv2.resize(imgs[i],(img_new_height,img_new_width)) #keep the license plate square to prevent stretching
        imgs[i] = np.resize(img_new,(220,100))
        imgs[i][100:220,0:100] = 200

        #blur the character randomly with values between 5-12
        blur = randint(5,20)
        imgs[i] = cv2.blur(imgs[i],(blur,blur))

    #start at i = 50 since this wont overwrite the simulated plates
    for i in range(4):
            j = 50
            path = cropped_directory + "/" + chars[i] + "_" + str(j) + ".png"

            while os.path.exists(path):
                j += 1
                path = cropped_directory + "/" + chars[i] + "_" + str(j) + ".png"

            cv2.imwrite(path, imgs[i])


    #now store the images into the folder "cropped pictures" with the name of the letter (manual)
    # new_filename = filename.replace('.png', '')
    # i = 0
    # while os.path.exists(cropped_directory + "/" + char1 + str(i) + ".png"):
    #     i = i+1
    # cv2.imwrite(cropped_directory + "/" + char1 + str(i) + ".png", crop_img1)

    # i = 0
    # while os.path.exists(cropped_directory + "/" + char2 + str(i) + ".png"):
    #     i = i+1
    # cv2.imwrite(cropped_directory + "/" + char2 + str(i) + ".png", crop_img2)

    # i = 0
    # while os.path.exists(cropped_directory + "/" + char3 + str(i) + ".png"):
    #     i = i+1
    # cv2.imwrite(cropped_directory + "/" + char3 + str(i) + ".png", crop_img3)

    # i = 0
    # while os.path.exists(cropped_directory + "/" + char4 + str(i) + ".png"):
    #     i = i+1
    # cv2.imwrite(cropped_directory + "/" + char4 + str(i) + ".png", crop_img4)


