from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
import socket
import sys
import base64
import os

HOST = '192.168.1.9'  # this is your localhost
PORT = 8888

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# socket.socket: must use to create a socket.
# socket.AF_INET: Address Format, Internet = IP Addresses.
# socket.SOCK_STREAM: two-way, connection-based byte streams.
print('socket created')

# Bind socket to Host and Port
try:
    s.bind((HOST, PORT))
except socket.error as err:
    print('Bind Failed, Error Code: ' + str(err[0]) + ', Message: ' + err[1])
    sys.exit()

print('Socket Bind Success!')
print(socket.gethostname())
# listen(): This method sets up and start TCP listener.
s.listen(10)
print('Socket is now listening')

flag = 1
while 1:
    client, addr = s.accept()
    print('Connect with ' + addr[0] + ':' + str(addr[1]))


## recive the image and save it to disk
    if flag ==1:
        data = b''
        with open("imageToSave.jpg", "wb") as fh:
            while True:
                b=client.recv(1024)
                data += b
                if not b:
                    break

            fh.write(base64.decodebytes(data))


        print("done  recive ! ")
        flag =0
        
        image = cv2.imread("imageToSave.jpg", 1)
        orig = image.copy()
        image=cv2.resize(image,(422,555))
        print(image.shape)

        # convert image to gray scale. This will remove any color noise
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur the image to remove high frequency noise 
        # it helps in finding/detecting contour in gray image
        grayImageBlur = cv2.blur(grayImage,(3,3))
        # then we performed canny edge detection
        edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)
        # find the contours in the edged image, sort area wise 
        # keeping only the largest ones 
        allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        allContours = imutils.grab_contours(allContours)
        # descending sort contours area and keep top 1
        allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]
        # approximate the contour
        perimeter = cv2.arcLength(allContours[0], True) 
        dimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)
        # show the contour on image
        cv2.drawContours(image, [dimensions], -1, (0,255,0), 2)
        cv2.imshow("Contour Outline", image)
        
        # reshape coordinates array
        dimensions = dimensions.reshape(4,2)
        # list to hold coordinates
        rect = np.zeros((4,2), dtype="float32")
        # top left corner will have the smallest sum, 
        # bottom right corner will have the largest sum
        s = np.sum(dimensions, axis=1)
        rect[0] = dimensions[np.argmin(s)]
        rect[2] = dimensions[np.argmax(s)]
        # top-right will have smallest difference
        # botton left will have largest difference
        diff = np.diff(dimensions, axis=1)
        rect[1] = dimensions[np.argmin(diff)]
        rect[3] = dimensions[np.argmax(diff)]
        # top-left, top-right, bottom-right, bottom-left
        (tl, tr, br, bl) = rect
        # compute width of ROI
        widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
        widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
        maxWidth = max(int(widthA), int(widthB))
        # compute height of ROI
        heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
        heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
        maxHeight = max(int(heightA), int(heightB))
        # Set of destinations points for "birds eye view"
        # dimension of the new image
        dst = np.array([
            [0,0],
            [maxWidth-1, 0],
            [maxWidth-1, maxHeight-1],
            [0, maxHeight-1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        transformMatrix = cv2.getPerspectiveTransform(rect, dst)
        # transform ROI
        scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))
        # convert to gray
        scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
        # increase contrast incase its document
        T = threshold_local(scanGray, 9, offset=8, method="gaussian")
        scanBW = (scanGray > T).astype("uint8") * 255
        # display final high-contrast image
        cv2.imwrite("scanned.jpg",scanBW)
        cv2.imshow("scanBW", scanBW)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
## read the image from the file and send it back to the mobile
    else:
        with open("scanned.jpg", "rb") as image_file:
            encode_bytes = base64.encodebytes(image_file.read())

        client.send(encode_bytes)
#        client.send(translation.encode('utf-8'))
        print(" done send !")
        flag = 1

    client.close()
s.close()






