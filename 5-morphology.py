import cv2
import numpy as np

def run_morphology_demo(image_path=None):
    # 1. Load Citra atau Buat Citra Biner Dummy
    if image_path:
        # Load dalam bentuk grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    else:
        # Buat gambar dummy biner ukuran 300x300 dengan beberapa noise/objek
        binary_img = np.zeros((300, 300), dtype=np.uint8)
        
        # Tambah persegi di tengah (objek utama)
        cv2.rectangle(binary_img, (75, 75), (225, 225), 255, -1)
        
        # Tambah noise putih kecil di background (untuk demo Opening)
        binary_img[30, 30] = 255
        binary_img[40, 250] = 255
        binary_img[250, 40] = 255
        
        # Tambah lubang/noise hitam kecil di dalam objek (untuk demo Closing)
        binary_img[140:160, 140:160] = 0

    # 2. Tentukan Structuring Element (Kernel)
    # Menggunakan kernel berbentuk persegi 5x5
    kernel = np.ones((5, 5), np.uint8)

    # 3. Operasi Morfologi
    # a. Erosi (Erosion) -> Mengikis tepi objek
    erosion = cv2.erode(binary_img, kernel, iterations=1)

    # b. Dilasi (Dilation) -> Memperbesar/menebalkan tepi objek
    dilation = cv2.dilate(binary_img, kernel, iterations=1)

    # c. Opening -> Erosi diikuti Dilasi (Menghilangkan noise kecil di background)
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # d. Closing -> Dilasi diikuti Erosi (Menutup lubang kecil di dalam objek)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # 4. Tampilkan Hasil (Gabungkan secara horizontal untuk perbandingan)
    top_row = np.hstack((binary_img, erosion, dilation))
    bottom_row = np.hstack((binary_img, opening, closing))
    
    # Cetak info di konsol
    print("Operasi morfologi berhasil dijalankan.")
    print("Gunakan tombol keyboard apapun pada jendela gambar untuk menutup.")

    # Tampilkan jendela OpenCV
    cv2.imshow("Atas: Asli | Erosi | Dilasi", top_row)
    cv2.imshow("Bawah: Asli | Opening | Closing", bottom_row)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Jalankan demo (bisa masukkan path gambar asli jika ada, contoh: run_morphology_demo('daun.jpg'))
    run_morphology_demo(image_path="foto/images.jpeg")