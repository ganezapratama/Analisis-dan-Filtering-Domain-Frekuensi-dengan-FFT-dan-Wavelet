import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft2, fftshift, ifft2, ifftshift

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread('cameramen.jpeg', 0)

img = cv2.resize(img, (256, 256))

# =========================
# BUAT CITRA NOISE PERIODIK
# =========================
x = np.arange(256)
y = np.arange(256)
X, Y = np.meshgrid(x, y)

noise = 30 * np.sin(2 * np.pi * X / 20)
img_noise = np.clip(img + noise, 0, 255).astype(np.uint8)

cv2.imwrite('noisy_image.jpeg', img_noise)

# =========================
# FFT FUNCTION
# =========================
def fft_analysis(image):
    f = fft2(image)
    fshift = fftshift(f)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    log_mag = np.log(1 + magnitude)
    return magnitude, phase, log_mag

# FFT kedua citra
mag1, phase1, log_mag1 = fft_analysis(img)
mag2, phase2, log_mag2 = fft_analysis(img_noise)

# =========================
# RECONSTRUCTION
# =========================
def reconstruct(magnitude, phase):
    complex_img = magnitude * np.exp(1j * phase)
    img_back = np.abs(ifft2(ifftshift(complex_img)))
    return np.uint8(np.clip(img_back, 0, 255))

# =========================
# FILTERING
# =========================
def ideal_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-crow)**2 + (j-ccol)**2) <= cutoff:
                mask[i, j] = 1
    return mask

def gaussian_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i-crow)**2 + (j-ccol)**2)
            mask[i, j] = np.exp(-(d**2)/(2*(cutoff**2)))
    return mask

# Lowpass (pakai citra noise)
lp_mask = ideal_lowpass(img.shape, 30)
img_lp = reconstruct(mag2 * lp_mask, phase2)

# Gaussian
g_mask = gaussian_lowpass(img.shape, 30)
img_g = reconstruct(mag2 * g_mask, phase2)

# =========================
# NOTCH FILTER
# =========================
def notch_filter(shape, centers, radius=5):
    rows, cols = shape
    mask = np.ones((rows, cols))

    for (x, y) in centers:
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i-x)**2 + (j-y)**2) < radius:
                    mask[i, j] = 0
    return mask

centers = [(128+13,128), (128-13,128)]
notch = notch_filter(img.shape, centers)

img_notch = reconstruct(mag2 * notch, phase2)

# =========================
# WAVELET
# =========================
coeffs = pywt.wavedec2(img, 'haar', level=2)
cA, (cH, cV, cD), *_ = coeffs

# =========================
# PSNR
# =========================
def psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

print("PSNR Lowpass:", psnr(img, img_lp))
print("PSNR Gaussian:", psnr(img, img_g))
print("PSNR Notch:", psnr(img, img_notch))

# =========================
# VISUALISASI
# =========================
plt.figure(figsize=(14,10))

# Citra asli
plt.subplot(3,4,1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(3,4,2)
plt.imshow(log_mag1, cmap='gray')
plt.title('FFT Original')

plt.subplot(3,4,3)
plt.imshow(phase1, cmap='hsv')
plt.title('Phase Original')

# Citra noise
plt.subplot(3,4,5)
plt.imshow(img_noise, cmap='gray')
plt.title('Noisy')

plt.subplot(3,4,6)
plt.imshow(log_mag2, cmap='gray')
plt.title('FFT Noisy')

plt.subplot(3,4,7)
plt.imshow(phase2, cmap='hsv')
plt.title('Phase Noisy')

# Filtering
plt.subplot(3,4,9)
plt.imshow(img_lp, cmap='gray')
plt.title('Lowpass')

plt.subplot(3,4,10)
plt.imshow(img_g, cmap='gray')
plt.title('Gaussian')

plt.subplot(3,4,11)
plt.imshow(img_notch, cmap='gray')
plt.title('Notch')

plt.tight_layout()
plt.show()

# =========================
# VISUALISASI WAVELET
# =========================
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(cA, cmap='gray')
plt.title('cA (Approximation)')

plt.subplot(2,2,2)
plt.imshow(cH, cmap='gray')
plt.title('cH (Horizontal Detail)')

plt.subplot(2,2,3)
plt.imshow(cV, cmap='gray')
plt.title('cV (Vertical Detail)')

plt.subplot(2,2,4)
plt.imshow(cD, cmap='gray')
plt.title('cD (Diagonal Detail)')

plt.tight_layout()
plt.show()