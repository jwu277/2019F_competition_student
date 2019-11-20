import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

class LicenseProcessor:

    def __init__(self):

        self.__im_counter = 0 # for testing, increments every time license_finder is called
        self.__im_counter_parse = 0 #used for saving unique names to files of parsed characters
        self.__path = os.path.dirname(os.path.abspath(__file__))
        self.__img_mem = None # should be named lp_mem... TODO

        self.__LP_RECOG_GRAY_THRESH = 0.7 # proportion of adjacent grayscale pixels needed
        self.__GRAYSCALE_DELTA_THRESH = 0x00 # max difference in BGR values for grayscale pixels
    
    def mem(self):
        return self.__img_mem

    #license_finder takes an image and determines if there is a parking spot with a license plate present in it
    #@Param
    #   img: the image to detect a license plate in
    #@Returns
    #   True iff a plate was found, and stores the new image in "cropped_plates"
    #   False iff there are no plates in the image
    def license_finder(self, img):
        #img = cv2.imread('images_pre/'+filename)
        rows,cols,ch = img.shape

        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            
        #mask all colours except blue
        mask1 = cv2.inRange(hsv, (110, 50, 0), (130, 255,255))
        mask = cv2.bitwise_and(img,img, mask=mask1)

        #use Hough transform to find verticle lines
        gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        minLineLength=90
        lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=80,lines=np.array([]), minLineLength=minLineLength,maxLineGap=30)

        #if there are no lines return 0
        if lines is None:
            #print(filename + " has no lines")
            return False

        a,b,c = lines.shape
        #lines = lines[lines[:,:,0].argsort()] # sort by x1
        # todo: remove double edges

        #create an empty list of points
        pts = []

        #drawing the lines for debugging
        for i in range(a):
            if np.abs(lines[i][0][0] - lines[i][0][2]) == 0:
                #cv2.line(mask, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 3, cv2.LINE_AA)
                x1 = lines[i][0][0]
                y1 = lines[i][0][1]
                x2 = lines[i][0][2]
                y2 = lines[i][0][3]

                pt1 = [y1,x1]
                pt2 = [y2,x2]
                #verify that the line is valid, done by checking for the correct colours on either side
                yav = (int)((y1 + y2) / 2)
                y_min = np.min([y1, y2])
                y_max = np.max([y1, y2])
                y_range = y_max - y_min

                #test lines
                # TODO what iff tl1/tl2 are off screen?
                tl1 = img[y_min:y_max,x1 - 5]
                tl2 = img[y_min:y_max,x1 + 5]
                
                #verify that one side of the test point is a license plate
                if np.sum((np.max(tl1, axis=1) - np.min(tl1, axis=1)) <= self.__GRAYSCALE_DELTA_THRESH) >= self.__LP_RECOG_GRAY_THRESH * y_range or \
                    np.sum((np.max(tl2, axis=1) - np.min(tl2, axis=1)) <= self.__GRAYSCALE_DELTA_THRESH) >= self.__LP_RECOG_GRAY_THRESH * y_range:

                    if len(pts) == 0:
                        #append pt2 for a smaller y-value first at the same x
                        pts.append(pt2)
                        pts.append(pt1)

                        mask[y1,x1] = [255, 0, 255]
                        mask[y2,x2] = [255, 0, 255]

                    #get rid of points in close proximity
                    for point in pts:
                        if abs(point[0] - y2) < 20 and abs(point[1] - x2) < 20:
                            break

                        if point == pts[len(pts)-1]:   
                            #append pt2 for a smaller y-value first at the same x
                            pts.append(pt2)
                            pts.append(pt1)

                            mask[y1,x1] = [255, 0, 255]
                            mask[y2,x2] = [255, 0, 255]

        #check to see if a license plate was found in the image, if not, return 0
        if len(pts) != 4:
            #print(filename + " has no plate")
            return False
        
        # filter out buggy lines by looking at height difference
        height1 = abs(pts[0][0]-pts[1][0])
        height2 = abs(pts[2][0]-pts[3][0])
        if abs(height1 - height2) >= min(height1, height2) * 0.3:
            #print("found a buggy line")
            return False

        #filter out lines that aree in too-close proximity with their x-values
        if abs(pts[0][1] - pts[2][1]) < 50:
            #print("lines too close together")
            return False

        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # # Show result
        # cv2.imshow("Result Image", img)
        # cv2.waitKey(0)

        #testing perspective transform
        xp1, yp1, yp2, xp3 = 0, 0, 0, 0
        xp2, yp3, xp4, yp4 = rows, rows+100, rows, rows+100

        #correctly label the points 1TL 2TR 3BL 4BR
        xmin = 9999
        xmax = 0
        ymin = 9999
        ymax = 0

        #find smallest and largest x and y values for the pixels
        for pt in pts:
            if pt[0] < ymin:
                ymin = pt[0]
            if pt[0] > ymax:
                ymax = pt[0]
            if pt[1] < xmin:
                xmin = pt[1]
            if pt[1] > xmax:
                xmax = pt[1]

        #these indexes are used to determine the order that images go into the perspective transform
        index1 = 0
        index2 = 0

        for i in range(len(pts)):
            if pts[i][1] == xmin:
                index1 = i
            if pts[i][1] == xmax:
                index2 = i 

        x1 = pts[index1-1][1]
        y1 = pts[index1-1][0]
        x3 = pts[index1][1]
        y3 = pts[index1][0]

        x2 = pts[index2-1][1]
        y2 = pts[index2-1][0]
        x4 = pts[index2][1]
        y4 = pts[index2][0]
        
        un_transformed_points = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
        transformed_points = np.float32([[xp1,yp1],[xp2,yp2],[xp3,yp3],[xp4,yp4]])

        M = cv2.getPerspectiveTransform(un_transformed_points, transformed_points)
        dst = cv2.warpPerspective(img,M,(xp4,yp4))
        
        #save the image to the images_post folder, change the picture iterator until it does not overwrite another image
        path = self.__path + "/cropped_plates" + "/plate" + str(self.__im_counter) + ".png"

        while os.path.exists(path):
            self.__im_counter += 1
            path = self.__path + "/cropped_plates" + "/plate" + str(self.__im_counter) + ".png"

        cv2.imwrite(path, dst)

        #cv2.imwrite(path + "/" + str(self.__im_counter) + "_og" + ".png", img)
        #print("License plate found")
        self.__img_mem = dst

        # #for testing show the images
        #plt.imshow(dst)
        #plt.show()
        return True

    #Takes a license plate image and crops it with individual plate letters and the parking stall number
    #@Param:
    #   img: the image of a license plate to be parsed
    #@Returns:
    #   A list with len(list) = 5 in the order [char1, char2, char3, char4, parking stall number] where the elements are B&W images of the plate numbers
    def parse_plate(self, img):
        #img = cv2.imread('cropped_plates/' + filename)

        #turn the image to B&W
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #| cv2.THRESH_OTSU after cv2.thresh_binary

        height, width, channels = img.shape
        letter_top, letter_bot, letter_height = 600, 700, 100
        letter_width = 100

        #crop all of the characters in the license plate
        char1 = img_bw[letter_top:letter_bot, 60:60 + letter_width]   
        char2 = img_bw[letter_top:letter_bot, 185:185 + letter_width]   
        char3 = img_bw[letter_top:letter_bot, 415:415 + letter_width]   
        char4 = img_bw[letter_top:letter_bot, 540:540 + letter_width]

        # #store the files in the correct location with the correct name
        # path = self.__path + "/cropped_chars"
        # cv2.imwrite(path + "/" + str(self.__im_counter_parse) + "_char1.png", char1)
        # cv2.imwrite(path + "/" + str(self.__im_counter_parse) + "_char2.png", char2)
        # cv2.imwrite(path + "/" + str(self.__im_counter_parse) + "_char3.png", char3)
        # cv2.imwrite(path + "/" + str(self.__im_counter_parse) + "_char4.png", char4)

        # self.__im_counter_parse += 1
        #crop the parking stall number
        img_stall = img_bw[300:500, 350:700]  
        img_stall = cv2.resize(img_stall,(100,100))
        #cv2.imwrite(path + "/" + str(self.__im_counter_parse) + "_stallnum.png", img_stall)

        return [char1, char2, char3, char4, img_stall]
        #print("Plate parsed!")
    
    
    #used to generate the testing dataset. This is required to keep the naming of the 
    def parse_plate_test_set(self, filename):
        img = cv2.imread('cropped_plates/' + filename)

        #turn the image to B&W
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #| cv2.THRESH_OTSU after cv2.thresh_binary

        height, width, channels = img.shape
        letter_top, letter_bot, letter_height = 600, 700, 100
        letter_width = 100

        #crop all of the characters in the license plate
        char1 = img_bw[letter_top:letter_bot, 60:60 + letter_width]   
        char2 = img_bw[letter_top:letter_bot, 185:185 + letter_width]   
        char3 = img_bw[letter_top:letter_bot, 415:415 + letter_width]   
        char4 = img_bw[letter_top:letter_bot, 540:540 + letter_width]
        #crop the parking stall number
        img_stall = img_bw[300:500, 350:700]  
        img_stall = cv2.resize(img_stall,(100,100))

        imgs = [char1, char2, char3, char4, img_stall]

        #allow the user to change the names of the images. This is done by showing the original, and then taking keyboard inputs a-z (lc) and 0-9
        cv2.imshow('Rename: ' + filename,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        #A list of the characters in the image
        chars = raw_input("Enter the plate name: ")

        #store the files in the correct location with the correct name without overwriting previous files
        for i in range(5):
            j = 0
            path = self.__path + "/cropped_chars" + "/" + chars[i] + "_" + str(j) + ".png"

            while os.path.exists(path):
                j += 1
                path = self.__path + "/cropped_chars" + "/" + chars[i] + "_" + str(j) + ".png"

            cv2.imwrite(path, imgs[i])
        