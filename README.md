# Fourier Cross Correlation using FFT

This project demonstrates the use of **Fast Fourier Transform (FFT)** to perform cross-correlation between two images. FFT-based correlation offers a computationally efficient way to measure alignment and similarity between signals or images, which is especially useful in computer vision tasks like object tracking and image registration.
## 📁 Notebook Overview: `Fourier_cross_corelation.ipynb`
### 🔧 1. **Library Imports and Helper Functions**

- `cv2` (OpenCV): Used to read and resize grayscale images.
- `matplotlib`: For plotting images and correlation results.
- `numpy`: For efficient numerical computation.

#### `load_image(image_path)`
- Reads a grayscale image from disk.
- Resizes it to 256x256.
- (Commented out) Optional application of a 2D Hann window to reduce edge artifacts in FFT.

#### `preprocess_image(image)`
- Normalizes the image to range `[0, 1]` by converting to `float32` and dividing by 255.

---

### 📁 2. **Uploading Images**
```python
from google.colab import files
upload = files.upload()
```
- Allows image upload directly from local filesystem to Colab for analysis.

---

### 🖼️ 3. **Image Selection**
- Two images are selected for cross-correlation comparison.


### 📊 5. **Cross-Correlation via FFT**
```python
def fft_cross_correlation(img1, img2):
    ...
```

- **FFT2**: Computes 2D FFTs of both images.
- **Multiplication**: Multiplies the FFT of `img1` with the complex conjugate of `img2`.
- **Inverse FFT**: Computes inverse FFT to get correlation surface.
- **Normalization**: Returns the normalized real part of the correlation map.

---

### 🔍 6. **Finding Maximum Correlation Point**
```python
def find_max_location(correlation):
    max_loc = np.unravel_index(np.argmax(correlation), correlation.shape)
    return max_loc
```
- Finds the pixel location where the correlation score is highest.
- Indicates the best alignment between the two images.

---

### 🧪 7. **Main Flow**
- Images are loaded, preprocessed, and passed into the `fft_cross_correlation` function.
- The correlation result is visualized using `plt.imshow`.
- Peak correlation point is printed for analysis.

---

### 📸 8. **Visualization**
```python
plt.imshow(correlation, cmap='hot')
plt.title('Cross-Correlation Heatmap')
```
- A heatmap showing where the two images are most similar.
- The brighter the region, the higher the similarity.


---

## 🚀 Running the Notebook

### Requirements:
```bash
pip install numpy matplotlib opencv-python
```

### Usage:
1. Open in Google Colab.
2. Upload two grayscale images.
3. Run all cells to compute and visualize cross-correlation.
## 💡 Potential Extensions

- Use in video tracking to align frames.
- Compare object templates in a larger scene.
- Apply window functions to reduce spectral leakage.

---


