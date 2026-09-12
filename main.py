
import cv2
import numpy as np
from matplotlib import pyplot as plt

# image = cv2.imread("foto/images.jpeg")

# [h,w,c] = image.shape
# for i in range(h):
#     for j in range(w):
#         image[i,j,0] = 0
#         image[i,j,2] = 0

# # plt.imshow(image)
# # plt.title("plt")
# # plt.show()

# cv2.imshow("images",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("gagal membuka kamera")
    exit

while True:
    ret, frame = video.read()

    if not ret :
        print("gagal mendapatkan frame dari kamera")
        break
    [h,w,c] = frame.shape
    for i in range (h):
        for j in range(w):
            frame[i,j,0] = 0
            frame[i,j,1] = 0

    cv2.imshow("Kamera",frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
