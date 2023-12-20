import cv2

# trained Dataset
trainedDataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#image Read
img=cv2.imread("Images/brooke-cagle--uHVRvDr7pg-unsplash.jpg")
# gray Scale
gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
faces =trainedDataset.detectMultiScale(gray)
print(faces)
for x,y,w,h,in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("mark",img)

#cv2.imshow("gray",gray)
cv2.waitKey()



