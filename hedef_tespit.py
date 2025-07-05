import cv2 
import numpy as np 
import imutils
import os 
from matplotlib import pyplot as plt

def daire_ic_alan(contour1 , contour2 , img):
    
    mask_inner = np.zeros_like(img) #img görüntüsünün boytulatı kadar sıfırlarla dolu matris cıkar
    mask_outer = np.zeros_like(img)

    # sıfır matrislerine contour cizilir
    cv2.drawContours(mask_inner, [contour1], -1, 255, -1)
    cv2.drawContours(mask_outer, [contour2], -1, 255, -1)

# İki maske farkı → sadece iki daire arası alan
    ring_mask = cv2.subtract(mask_outer, mask_inner) # üzerine contourları cizilen matrisler piksel piksel cıkartılır
                                                
# Koordinatları al
    ys, xs = np.where(ring_mask == 255) #255 piksellere sahip kordinatlar alınır
    coordinate = list(zip(xs, ys)) #tuplelar listelenir 
    return coordinate


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


mask_inner = np.zeros_like(img)

# 2. İç ve dış çemberleri doldurarak çiz (içi dolu olacak şekilde)
cv2.drawContours(mask_inner, [cnts_daire[4]], -1, 255, -1)

# 3. Maske farkı (halka kısmı)
ring_mask = cv2.subtract(mask_inner,0)

# 4. Halka bölgesindeki koordinatları al
ys, xs = np.where(ring_mask == 255)
puan_alan_1 = list(zip(xs, ys))

puan_alan_5 = daire_ic_alan(cnts_daire[1] , cnts_daire[0] , img)
puan_alan_4 = daire_ic_alan(cnts_daire[2] , cnts_daire[1] , img)
puan_alan_3 = daire_ic_alan(cnts_daire[3] , cnts_daire[2] , img)
puan_alan_2 = daire_ic_alan(cnts_daire[4] , cnts_daire[3] , img)

alanlar = [puan_alan_1 , puan_alan_2,puan_alan_3 , puan_alan_4 , puan_alan_5]





