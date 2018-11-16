#%% Video Analysis

import numpy as np
import matplotlib.pyplot as plt
import cv2


cap = cv2.VideoCapture(0)
# 0 ; First webcam - 1 ; Second webcom ...
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

#%% Image Analysis
img = cv2.imread('name_of_the_image.jpg', cv2.IMREAD_GRAYSCALE)

#Correspondings are
#IMREAD_GRAYSCALE = 0
#IMREAD_COLOR = 1
#IMREAD_UNCHANGED = -1

cv2.imshow('image', img)    #Necessery for showing image
                            #'image' represents name of the window

cv2.imwrite('watchgray.png', img)

cv2.waitKey(0)              #Basically this just waits for any key to be pressed
cv2.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.plot([50, 100], [80, 100], 'c', linewidth=5)
plt.show()
#OpenCV is BG R, Matplotlib is RGB

#%% Drawing and Writing on Image

cv2.line(img, (0,0), (150,150), (255,255,255), 15)
cv2.rectangle(img, (15,25), (200,150), (0,255,0), 5)
# img; where do you want to draw
# Starting point, Ending point, BGR values, Width
cv2.circle(img, (100,63), 55, (0,0,255),+ -1)   #55 is radius


pts = np.array([[10,5], [20, 30], [70, 20], [50,10]], np.int32)
#pts. = pts.reshape((-1,-1,2))
cv2.polylines(img, [pts], True, (0,255,255), 5)


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "OpenCV Tuts!", (0,130), font, 1, (200,255,255), 2, cv2.LINE_AA)
