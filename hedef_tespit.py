import cv2 
import numpy as np 
import imutils
import os 
from matplotlib import pyplot as plt

img_liste = ["1.png" ,"2.png","3.png","5.png","6.png"]

img = cv2.imread("2.png")

hsv = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2HSV)

lower = np.array([0, 0, 127])      
upper = np.array([179, 255, 127]) 
mask = cv2.inRange(img , lower , upper)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (3,3))
closed_mask = cv2.morphologyEx(mask , cv2.MORPH_CLOSE , kernel )
closed_mask = cv2.erode(closed_mask , None , iterations=1)
closed_mask  = cv2.dilate(closed_mask  ,None , iterations=2)

cnts = cv2.findContours(closed_mask , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
cnts = [c for c in cnts if cv2.contourArea(c) > 80]

hedef_kordinatları = []
for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    hedef_kordinatları.append((x,y,w,h))

img = cv2.imread("3.png")

img = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 0])      
upper = np.array([123, 201,120 ]) 

img = cv2.inRange(img , lower , upper)
img  = cv2.dilate(img  ,None , iterations=2)

cnts_daire  = cv2.findContours(img , cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts_daire = imutils.grab_contours(cnts_daire)

cnts_daire = [c for c in cnts_daire if cv2.contourArea(c) > 6400 ]
(a,b)= img.shape[:2]

matris = np.ones((a,b) , dtype= np.uint8)*255
cnts_daire  = sorted (cnts_daire , key = cv2.contourArea , reverse = True)

cnts_daire = [c for i,c in enumerate(cnts_daire) if i %2 == 0 ]


