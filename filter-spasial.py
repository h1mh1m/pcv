import cv2
import numpy as np
def konvolusi(f, w, tepi="replicate"):
    m, n = w.shape
    a, b = m // 2, n // 2
    # Langkah 1: putar kernel 180 derajat -> inilah yang membedakan
    # konvolusi dari korelasi
    w_putar = np.flipud(np.fliplr(w))
    # Langkah 2: lebarkan citra supaya tepi punya tetangga
    mode = {"zero": "constant", "replicate": "edge", "reflect": "reflect"}[tepi]
    f_pad = np.pad(f.astype(np.float64), ((a, a), (b, b)), mode=mode)
    # Langkah 3: geser jendela ke seluruh posisi
    g = np.zeros(f.shape, dtype=np.float64)
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            jendela = f_pad[x:x + m, y:y + n] # ambil ketetanggaan
            g[x, y] = np.sum(jendela * w_putar) # jumlah hasil kali
    return g
def ke_uint8(g):
    return np.clip(np.floor(g + 0.5), 0, 255).astype(np.uint8)
f = np.array([[10, 10, 10, 10, 10],
[10, 50, 50, 50, 10],
[10, 50, 150, 50, 10],
[20, 40, 40, 40, 20],
[20, 20, 20, 20, 20]], dtype=np.float64)

w = np.ones((3, 3), np.float64) / 9.0
print(np.round(konvolusi(f, w, tepi="zero"), 2))
# Versi OpenCV. Perhatikan: filter2D menghitung KORELASI,
# jadi untuk kernel tak simetris kernelnya perlu dibalik lebih dulu.
hasil_cv = cv2.filter2D(f, -1, cv2.flip(w, -1), borderType=cv2.BORDER_REPLICATE)

im = cv2.imread("citra.jpg", cv2.IMREAD_GRAYSCALE)
# ---------- Box filter ----------
kernel_box = np.ones((3, 3), np.float32) / 9.0
hasil_box = cv2.filter2D(im, -1, kernel_box, borderType=cv2.BORDER_REPLICATE)
hasil_box2 = cv2.blur(im, (3, 3)) # jalan pintas, hasilnya sama
# ---------- Gaussian ----------
kernel_gauss = np.array([[1, 2, 1],
[2, 4, 2],
[1, 2, 1]], np.float32) / 16.0

hasil_gauss = cv2.filter2D(im, -1, kernel_gauss, borderType=cv2.BORDER_REPLICATE)
hasil_gauss2 = cv2.GaussianBlur(im, (5, 5), sigmaX=1.0) # sigma ditentukan sendiri
# Melihat kernel Gaussian yang dipakai OpenCV (bentuk satu dimensi, karena terpisahkan

k1d = cv2.getGaussianKernel(ksize=5, sigma=1.0)
print(np.round((k1d @ k1d.T), 4)) # kernel 2D hasil perkalian luar
# ---------- Median, untuk derau salt and pepper ----------
hasil_median = cv2.medianBlur(im, 3)
# Membandingkan kekuatan penghalusan lewat simpangan baku
for nama, hasil in [("asli", im), ("box", hasil_box),("gauss", hasil_gauss), ("median", hasil_median)]:
    print("%-8s simpangan baku = %.2f" % (nama, hasil.std()))

im = cv2.imread("citra.jpg", cv2.IMREAD_GRAYSCALE)
imf = im.astype(np.float64) # kerjakan dalam float, jangan uint8
# ---------- Laplacian dan penajamannya ----------
kernel_lap = np.array([[0, 1, 0],
[1, -4, 1],
[0, 1, 0]], np.float64)

lap = cv2.filter2D(imf, -1, kernel_lap, borderType=cv2.BORDER_REPLICATE)
# pusat kernel bernilai -4, maka citra dikurangi Laplacian-nya
tajam = imf - lap
tajam = np.clip(tajam, 0, 255).astype(np.uint8) # WAJIB dipotong
# Laplacian bawaan OpenCV, hasilnya setara
lap_cv = cv2.Laplacian(im, cv2.CV_64F, ksize=1)
# ---------- Unsharp masking dan highboost ----------
def unsharp(im, k=1.0, ukuran=3):
    imf = im.astype(np.float64)
    halus = cv2.blur(imf, (ukuran, ukuran)) # versi halus
    mask = imf - halus # bagian yang hilang
    hasil = imf + k * mask # dikembalikan, dikuatkan k
    return np.clip(hasil, 0, 255).astype(np.uint8)
hasil_unsharp = unsharp(im, k=1.0) # unsharp masking
hasil_highboost = unsharp(im, k=2.5)

gx = cv2.Sobel(imf, cv2.CV_64F, 1, 0, ksize=3) # turunan arah x
gy = cv2.Sobel(imf, cv2.CV_64F, 0, 1, ksize=3) # turunan arah y
besar = np.abs(gx) + np.abs(gy) # hampiran murah
besar_tepat = np.sqrt(gx**2 + gy**2) # bentuk sebenarnya
arah = np.arctan2(gy, gx) # arah gradien, radian
tepi = np.clip(besar, 0, 255).astype(np.uint8)