import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

#license_finder takes an image and determines if there is a parking spot with a license plate present in it
#@Param
#   filename: the name of the file to locate license plates in
#@Returns
#   True iff a plate was found, and stores the new image in "cropped_plates"
#   False iff there are no plates in the image
def license_finder(filename):
    img = cv2.imread('images_pre/'+filename)
    rows,cols,ch = img.shape

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
    #mask all colours except blue
    mask1 = cv2.inRange(hsv, (110, 50, 0), (130, 255,255))
    mask = cv2.bitwise_and(img,img, mask=mask1)

    #use Hough transform to find verticle lines
    gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength=1
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=80,lines=np.array([]), minLineLength=minLineLength,maxLineGap=30)

    #if there are no lines return 0
    if lines is None:
        #print(filename + " has no lines")
        return False

    a,b,c = lines.shape

    #create an empty list of points
    pts = []

    #drawing the lines for debugging
    for i in range(a):
        if lines[i][0][0] == lines[i][0][2]:
            #cv2.line(mask, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 3, cv2.LINE_AA)
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[i][0][2]
            y2 = lines[i][0][3]

            pt1 = [y1,x1]
            pt2 = [y2,x2]
            #verify that the line is valid, done by checking for the correct colours on either side
            yav = (int)((y1 + y2) / 2)
            #ydiff = abs(y1-y2)

            #test points   
            tpt1 = img[yav,x1 - 5]
            tpt2 = img[yav,x1 + 5]
            
            #verify that one side of the test point is a license plate
            if (tpt1[0] == tpt1[1] and tpt1[1] == tpt1[2]) or \
            (tpt2[0] == tpt2[1] and tpt2[1] == tpt2[2]):

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
    if len(pts) != 2:
        #print(filename + " has no plate")
        return False

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

    #save the image to the images_post folder
    path = os.getcwd() + "/cropped_plates"
    cv2.imwrite(path + "/" + filename, dst)

    #for testing show the images
    plt.imshow(dst)
    plt.show()
    return True

#Takes a license plate image and crops it with individual plate letters and the parking stall number
#Places the new images in "cropped_chars"
#@Param:
#   filename: the name of the license plate image
#@Returns:
#   True iff the image was succesfully parsed
#   False iff the image parsing failed
def parse_plate(filename):
    img = cv2.imread('cropped_plates/' + filename)
    height, width, channels = img.shape
    letter_top, letter_bot, letter_height = 600, 700, 100
    letter_width = 100

    #crop all of the characters in the license plate
    char1 = img[letter_top:letter_bot, 60:60 + letter_width]   
    char2 = img[letter_top:letter_bot, 185:185 + letter_width]   
    char3 = img[letter_top:letter_bot, 415:415 + letter_width]   
    char4 = img[letter_top:letter_bot, 540:540 + letter_width]

    #store the files in the correct location with the correct name
    path = os.getcwd() + "/cropped_chars"
    cv2.imwrite(path + "/char1.png", char1)
    cv2.imwrite(path + "/char2.png", char2)
    cv2.imwrite(path + "/char3.png", char3)
    cv2.imwrite(path + "/char4.png", char4)

    #crop the parking stall number
    stall_number = img[300:500, 350:700]  
    stall_number = cv2.resize(stall_number,(100,100))
    cv2.imwrite(path + "/stallnum.png", stall_number)

    cv2.imshow("number",stall_number)
    cv2.waitKey(0)
    