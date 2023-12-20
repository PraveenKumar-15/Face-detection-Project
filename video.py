import cv2
import face

trainedDataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


video=cv2.VideoCapture("Videos/Untitled.mp4")
while True:
    success,frame=video.read()
    if success==True:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        face =trainedDataset.detectMultiScale(gray_image)
        for x, y, w, h, in face:
         cv2.rectangle(frame,(x, y), (x + w, y + h), (0, 0, 255),2)
        cv2.imshow("video",frame)
        cv2.waitKey(1)

    else:
        print("video Completed or Frame Nil")
        break


