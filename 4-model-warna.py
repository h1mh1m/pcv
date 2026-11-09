import numpy as np
import cv2

# ==========================================
# 1. RGB ke CMYK
# ==========================================
def rgb_to_cmyk(rgb_img):
    """
    Mengonversi citra RGB (0-255) ke CMYK (0-1).
    """
    # Normalisasi skala [0, 255] -> [0, 1]
    rgb_normalized = rgb_img.astype(float) / 255.0
    R = rgb_normalized[:, :, 2]
    G = rgb_normalized[:, :, 1]
    B = rgb_normalized[:, :, 0] # Catatan: jika menggunakan OpenCV, urutan channel adalah BGR

    # Hitung Key (Black / K)
    K = 1.0 - np.maximum(np.maximum(R, G), B)
    
    # Hindari pembagian dengan nol untuk piksel hitam murni
    C = np.zeros_like(R)
    M = np.zeros_like(G)
    Y = np.zeros_like(B)
    
    mask = (K < 1.0)
    C[mask] = (1.0 - R[mask] - K[mask]) / (1.0 - K[mask])
    M[mask] = (1.0 - G[mask] - K[mask]) / (1.0 - K[mask])
    Y[mask] = (1.0 - B[mask] - K[mask]) / (1.0 - K[mask])

    return C, M, Y, K

# ==========================================
# 2. RGB ke HSV
# ==========================================
def rgb_to_hsv_opencv(bgr_img):
    """
    Mengonversi citra BGR ke HSV menggunakan OpenCV.
    H: 0-179, S: 0-255, V: 0-255
    """
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)
    return H, S, V

# ==========================================
# 3. RGB ke HSI (Manual Calculation)
# ==========================================
def rgb_to_hsi(bgr_img):
    """
    Mengonversi citra BGR ke HSI secara matematis.
    H: 0 - 360 derajat, S: 0 - 1, I: 0 - 1
    """
    img = bgr_img.astype(float) / 255.0
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Hitung Intensity (I)
    I = (R + G + B) / 3.0

    # Hitung Saturation (S)
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.zeros_like(I)
    
    # Hindari pembagian dengan 0 pada piksel hitam
    mask = I > 0
    S[mask] = 1.0 - (min_rgb[mask] / I[mask])

    # Hitung Hue (H)
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B))
    
    eps = 1e-6 # Penstabil nilai nol
    theta = np.arccos(np.clip(num / (den + eps), -1.0, 1.0))
    
    H = np.degrees(theta)
    H[B > G] = 360.0 - H[B > G] # Koreksi jika B > G

    return H, S, I

# ==========================================
# Contoh Penggunaan & Pengujian
# ==========================================
if __name__ == "__main__":
    # 1. Buat citra dummy RGB (ukuran 200x200 piksel)
    # Atau ganti baris ini dengan membaca gambar asli: bgr_img = cv2.imread('gambar.jpg')
    bgr_img = np.zeros((200, 200, 3), dtype=np.uint8)
    bgr_img[:, :100] = [0, 0, 255]   # Merah (Red) di sebelah kiri
    bgr_img[:, 100:] = [255, 255, 0] # Cyan di sebelah kanan

    print("--- 1. Konversi RGB -> CMYK ---")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    C, M, Y, K = rgb_to_cmyk(rgb_img)
    print(f"Piksel Merah (RGB) -> Cyan: {C[100, 50]:.2f}, Magenta: {M[100, 50]:.2f}, Yellow: {Y[100, 50]:.2f}, Black: {K[100, 50]:.2f}")

    print("\n--- 2. Konversi RGB -> HSV ---")
    H_hsv, S_hsv, V_hsv = rgb_to_hsv_opencv(bgr_img)
    print(f"Piksel Merah (HSV OpenCV) -> H: {H_hsv[100, 50]}, S: {S_hsv[100, 50]}, V: {V_hsv[100, 50]}")

    print("\n--- 3. Konversi RGB -> HSI ---")
    H_hsi, S_hsi, I_hsi = rgb_to_hsi(bgr_img)
    print(f"Piksel Merah (HSI) -> H: {H_hsi[100, 50]:.1f}°, S: {S_hsi[100, 50]:.2f}, I: {I_hsi[100, 50]:.2f}")