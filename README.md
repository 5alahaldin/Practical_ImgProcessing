### 1. **Equalization**

**What it does**: Histogram equalization is used to improve the contrast of an image by adjusting its intensity distribution. It works by redistributing the pixel values across the entire available range, stretching the histogram of pixel intensities.

**How it works**:
- The algorithm first converts the image to grayscale because histogram equalization is generally applied to single-channel images.
- It then calculates the cumulative distribution function (CDF) of the pixel intensities, which maps the input pixel values to a more evenly distributed set of output values.
- The resulting image has improved contrast, making features more distinct, especially in low-contrast images.

**Applications**:
- Useful for improving visibility in images with poor lighting or low contrast, such as medical imaging and satellite photos.

---

### 2. **Contrast Stretching**

**What it does**: Contrast stretching enhances the contrast of an image by scaling the pixel values in a specified range, typically between 0 and 255 (the full grayscale range). The result is an image with increased contrast where the differences between pixel intensities are more noticeable.

**How it works**:
- The algorithm first identifies the minimum and maximum pixel values in the image.
- It then maps these values to the desired `min_val` and `max_val` range (usually 0 to 255) using a linear transformation formula:

$$I_{\text{new}} = (I - \text{min}) \cdot \left( \frac{N_{\text{max}} - N_{\text{min}}}{\text{max} - \text{min}} \right) + N_{\text{mix}}$$

- This transformation increases the image’s contrast by spreading out the pixel values across the range.

**Applications**:
- Used for enhancing details in low-contrast images where features are hard to distinguish, like in low-light conditions.

---

### 3. **Filtering (Spatial and Frequency Domains)**

**What it does**: Filtering is used to process an image by modifying pixel values based on a filter (kernel). Filters can be used for various purposes, including blurring, sharpening, edge detection, and noise reduction. This can be done in both the spatial domain (direct manipulation of pixel values) and the frequency domain (transforming the image to the frequency domain, applying filters, and then transforming it back).

#### Types of Filters:
- **Mean Filter**: Averages pixel values within a kernel (typically a 3x3 or 5x5 matrix), resulting in a blurred image that reduces noise.
  
- **Gaussian Filter**: Applies a Gaussian function to smooth the image. It reduces noise and detail but keeps the edges relatively sharp.
  
- **Sobel Filter**: Used for edge detection by calculating gradients in the x and y directions and combining them to highlight edges.

**How it works**:
- **Mean Filter**: The algorithm slides a kernel (matrix) over the image and computes the average of the surrounding pixels. This is a basic form of smoothing.
  
- **Gaussian Filter**: Similar to the mean filter, but the kernel values follow a Gaussian distribution, giving more weight to the center pixels and less to the surrounding ones. This results in smoother edges and reduces high-frequency noise.

- **Sobel Filter**: This filter calculates the gradient of pixel intensities in the horizontal and vertical directions using Sobel operators. The magnitude of the gradient at each pixel highlights the edges in the image.

**Applications**:
- **Mean Filter**: Removes salt-and-pepper noise in an image.
- **Gaussian Filter**: Commonly used for blurring and reducing detail.
- **Sobel Filter**: Edge detection in object recognition or feature extraction.

---

### 4. **Segmentation**

**What it does**: Image segmentation divides an image into multiple segments or regions based on pixel intensity, color, or texture. The goal is to simplify the representation of an image or make it easier to analyze.

**How it works**:
- In this case, segmentation is achieved by using a simple thresholding technique.
- The image is first converted to grayscale (if not already).
- A pixel value greater than a given threshold is classified as the foreground (usually white), and anything below the threshold is classified as the background (usually black).
- The threshold can be adjusted to fine-tune the segmentation results.

**Applications**:
- Used for separating objects from the background, such as in medical imaging (detecting tumors), object detection, or image analysis.

---

### 5. **Interpolation**

**What it does**: Interpolation is used for resizing images by estimating pixel values in new locations. It can increase or decrease the image size, depending on the scaling factor.

**How it works**:
- The algorithm resizes the image by calculating new pixel values using the original pixels and the desired scaling factor.
- **Linear interpolation** estimates values between two existing pixels. For each new pixel location, the algorithm calculates a weighted average of nearby pixel values.
  
**Applications**:
- Used in image resizing, such as scaling an image to fit a display or reduce the image size for storage.

---

### 6. **Log Transformation**

**What it does**: Logarithmic transformation is used to adjust the image's brightness, particularly to enhance the darker regions while compressing the lighter ones.

**How it works**:
- The algorithm applies a logarithmic function to the pixel values, which amplifies the dark regions of the image while reducing the bright areas.

$$s = c \cdot \log(1 + r)$$
$$c = \frac{255}{\log(1 + \text{max}(r))}$$


where `c` is a constant that scales the transformation.
  
- The result is an image where the low-intensity pixels (dark areas) become brighter, making it easier to analyze low-contrast areas.

**Applications**:
- Enhances low-light images, such as in night-time photography or low-contrast scientific images.

---

### 7. **Gaussian Blurring**

**What it does**: Gaussian blurring is a smoothing technique that reduces image noise and detail by averaging pixel values within a kernel that follows a Gaussian distribution.

**How it works**:
- A Gaussian filter is applied by convolving the image with a kernel that has values weighted based on a Gaussian function.
- This blurring effect helps to reduce high-frequency noise while maintaining relatively smooth edges.

**Applications**:
- Used in pre-processing steps for image analysis, such as edge detection or noise removal.

---

### 8. **Edge Detection (Canny)**

**What it does**: Edge detection is used to identify boundaries of objects within an image. The Canny edge detection algorithm is one of the most popular for detecting sharp changes in intensity (edges).

**How it works**:
- The algorithm involves several steps:
  1. **Smoothing**: The image is blurred to reduce noise using a Gaussian filter.
  2. **Gradient Calculation**: The intensity gradient of the image is calculated to identify where the pixel intensity changes rapidly (edges).
  3. **Non-Maximum Suppression**: The algorithm removes non-edges, keeping only the strongest edges.
  4. **Edge Tracing by Hysteresis**: The final edges are traced by connecting strong edges and suppressing weak ones.
  
- The result is a binary image where edges are marked as white pixels.

**Applications**:
- Used for detecting boundaries in images, such as in object detection, face detection, or feature extraction.

---

### 9. **Morphological Transformations (Dilation and Erosion)**

**What it does**: Morphological transformations manipulate the image structure based on its shape. Two common operations are **dilation** and **erosion**.

- **Dilation** increases the white region in an image, expanding boundaries.
- **Erosion** decreases the white region, shrinking boundaries.

**How it works**:
- Both operations use a **structuring element** (kernel) that slides over the image.
  - **Dilation** replaces the pixel with the maximum value in the kernel's region.
  - **Erosion** replaces the pixel with the minimum value in the kernel's region.

**Applications**:
- **Dilation**: Used to fill holes in an object or to connect disjointed parts of an object.
- **Erosion**: Used to remove small noise or separate objects that are too close.

---

### 10. **Median Filtering**

**What it does**: Median filtering is used to reduce noise in an image, particularly salt-and-pepper noise, by replacing each pixel value with the median of the surrounding pixels.

**How it works**:
- The algorithm slides a kernel (usually 3x3) over the image and for each pixel, it replaces the original pixel value with the median of the neighboring pixels in the kernel.
- This method preserves edges better than mean filtering because it does not average out high-intensity edges.

**Applications**:
- Often used in image denoising, particularly in medical imaging or photographs with grainy noise.
