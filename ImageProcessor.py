
import cv2 as cv
from PIL import Image
import numpy as np
import pytesseract
import re
import spacy
from scipy import ndimage

TESSDATA_PREFIX = r'C:\Program Files\Tesseract-OCR\tessdata'
tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract

img = cv.imread("images/visiting2.jpg")


gray = cv.multiply(cv.cvtColor(img, cv.COLOR_BGR2GRAY),1.5)

# edges = cv.Canny(gray,5,5,apertureSize = 3)
thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5,5)
orig_thresh = thresh.copy()


################# DEFINING KERNEL #################

h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 1))
v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 50))

detected_h = cv.morphologyEx(thresh, cv.MORPH_OPEN, h_kernel, iterations=1)
detected_v = cv.morphologyEx(thresh, cv.MORPH_OPEN, v_kernel, iterations=1)


for i in zip(np.where(detected_h>0)[0],np.where(detected_h>0)[1]):
    thresh[i[0],i[1]] = 0
for i in zip(np.where(detected_v>0)[0],np.where(detected_v>0)[1]):
    thresh[i[0],i[1]] = 0

minLineLength=20
lines = cv.HoughLinesP(image=thresh,rho=1,theta=np.pi/180, threshold=200,lines=np.array([]), minLineLength=minLineLength,maxLineGap=5)
if np.array(lines).any():
    a,b,c = lines.shape
    for i in range(a):
        cv.line(thresh, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 0), 5, cv.LINE_AA)



kernel = np.ones((1,1), np.uint8)  
img_erosion = cv.erode(255-thresh, kernel, iterations=1)


#----------------------------PHASE 2--------------------------------

def isRectangleOverlap(R1, R2):
    if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
        return False
    else:
        return True

dilate = cv.dilate(255-img_erosion, cv.getStructuringElement(cv.MORPH_RECT, (15,10) ),iterations=1)
cnts = cv.findContours(dilate,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
rows=[]
for c in cnts:
    x,y,w,h = cv.boundingRect(c)
    if (w>1.5*h) and (img_erosion[y:y+h,x:x+w].size/img_erosion.size<.5) and (img_erosion[y:y+h,x:x+w].size/img_erosion.size>.001):
        rows.append([x,y,x+w,y+h])


ori = np.zeros(thresh.shape, np.uint8)
ori.fill(255)
for x,y,w,h in rows:
    ori[y:h,x:w] = img_erosion[y:h,x:w]
ori = cv.dilate(ori, cv.getStructuringElement(cv.MORPH_RECT, (1,1) ),iterations=3)



rows = [list((x,y,min(ori.shape[1],w+int(.5*(h-y))),h)) for x,y,w,h in rows]


ov_rect = []
for n1,i in enumerate(rows):
    for n2,j in enumerate(rows):
        if (isRectangleOverlap(i,j)):
            ov_rect.append(sorted([n1,n2]))

k = [sorted(i) for i in ov_rect if not i[0]==i[1]]
# k = list(pd.Series(k).drop_duplicates())
[k.remove(i) for i in k if i in k]

for i,j in k[:3]:
    rows.append([min(rows[i][0],rows[j][0]),min(rows[i][1],rows[j][1]),max(rows[i][2],rows[j][2]),max(rows[i][3],rows[j][3])])
    rows.pop(j)
    rows.pop(i)  

copied = img.copy()
for x,y,w,h in rows:
    cv.rectangle(copied,(x,y),(w,h),(234,132,76),2)

#-----------------------phase 3

data =[pytesseract.image_to_string(img[y-3:h+3,x-3:w+3], lang='eng', config='--psm 6') for x,y,w,h in rows][::-1]
data = [i.strip() for i in data]
data = [re.sub('[\s-]',' ',i[:-1])+re.sub('[\s\.,_\?]','',i[-1]) for i in data if i]


def get_entity_label(x):
    text1= NER(x)
    name = []
    org = []
    addr = []
    for word in text1.ents:
        word,label = [word.text,word.label_]
        if label == 'PERSON':
            name.append(word)
        if label == 'ORG':
            org.append(word)
        if label == 'GPE':
            addr.append(word)
    return name,org, addr


NER = spacy.load("en_core_web_sm")
ent = [get_entity_label(i) for i in data]

#name
try:
    name = [''.join(i[0]) for i in ent if i[0]]
    name = name[0]
except:
    name ='    '



#organisation
try:
    org = [''.join(i[1]) for i in ent[:int(''.join([str(n) for n,i in enumerate(ent) if name in i[0]]))+1 if name else '']]
    org = org[0]
except:
    org = '    '


#Address
try:
    address = ','.join([data[i] for i in [n for n,i in enumerate(ent) if i[2]]])
except:
    address = '    '


#Contact and Website
tlds = ['com','net','in','gov.in']
website = ''.join([''.join(re.findall('|'.join(['\w+\.' + i for i in tlds]),j)) for j in data])
res = [re.findall(r"((?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}))", i) for i in data]
res = [','.join(i) for i in res if i]


#Email
email= ''.join([''.join(re.findall('|'.join(['.+@[a-zA-Z0-9]+\.' + i + '$' for i in tlds]),j)) for j in data])
email = ','.join([i for i in email.split() if not re.findall('^w{3}',i)])

#import pandas as pd
#pd.DataFrame([[name,address,org,res,email,website]], columns=['NAME', 'ADDRESS', 'ORGANIZATION','CONTACT DETAILS','EMAIL','WEBSITE'])
















