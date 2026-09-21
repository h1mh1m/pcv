import numpy as np 
import matplotlib.pyplot as plt
import math
import cv2

image = cv2.imread("foto/images.jpeg",cv2.IMREAD_GRAYSCALE)

def CitraNegatif(im):
    tinggi, lebar = im.shape
    im_negatif = np.zeros((tinggi, lebar),dtype=np.uint8)
    for i in range(tinggi):
        for j in range(lebar):
            im_negatif[i,j] = 255 - im[i,j]

    return im_negatif

def TransformasiLogarimtik(im):
    tinggi, lebar = im.shape
    imhasil = np.zeros((tinggi, lebar), dtype=np.uint8)

    max_nilai = 0
    for i in range(tinggi):
        for j in range(lebar):
            if im[i,j] > max_nilai:
                max_nilai = im[i,j]
    
    c =  255 / math.log(1 + max_nilai) if max_nilai > 0 else 0
    for i in range(tinggi):
        for j in range(lebar):
            nilai_baru = c * math.log(1 + im[i,j])
            imhasil[i,j] = int(nilai_baru)

    return imhasil

def TransformasiPangkat(im, gamma):
    tinggi, lebar = im.shape
    imhasil = np.zeros((tinggi, lebar), dtype=np.uint8)
    for i in range(tinggi):
        for j in range(lebar):
            normalize = im[i,j] / 255.0
            nilai_baru = 255 * (normalize ** gamma)
            if nilai_baru > 255 : nilai_baru = 255
            if nilai_baru < 0 : nilai_baru = 0

            imhasil[i,j] = int(nilai_baru)
    return imhasil


cv2.imshow("transformasi negatif",CitraNegatif(image))
cv2.imshow("transformasi logaritmik", TransformasiLogarimtik(image))
cv2.imshow("transformasi pangkat", TransformasiPangkat(image, gamma=1.0))
# TransformasiLogarimtik(image) 

cv2.waitKey(0)
cv2.destroyAllWindows()
    

