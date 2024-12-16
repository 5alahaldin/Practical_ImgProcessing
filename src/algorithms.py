import cv2
import numpy as np


class ImageProcessingAlgorithms:
  @staticmethod
  def invert_image(image):
    return cv2.bitwise_not(image)
  
  @staticmethod
  def contrast_stretching(image, min_value, max_value):
        if len(image.shape) == 2:  # Grayscale image
            # Compute min and max of the grayscale image
            min_val = np.min(image)
            max_val = np.max(image)
            # Apply contrast stretching
            stretched = ((image - min_val) / (max_val - min_val)) * (max_value - min_value) + min_value
            return np.uint8(np.clip(stretched, 0, 255))

        elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
            # Initialize an empty array for the stretched image
            stretched = np.zeros_like(image, dtype=np.float32)
            # Apply contrast stretching to each channel individually
            for channel in range(3):
                min_val = np.min(image[:, :, channel])
                max_val = np.max(image[:, :, channel])
                stretched[:, :, channel] = ((image[:, :, channel] - min_val) /
                                            (max_val - min_val)) * (max_value - min_value) + min_value
            # Clip values to 0-255 and convert back to uint8
            return np.uint8(np.clip(stretched, 0, 255))
        else:
            raise ValueError("Input image must be either grayscale or RGB.")
  def log_transform(image, c=1):
    if c is None:
        c = 255 / np.log(1 + np.max(image))
    return np.uint8(c * np.log1p(image))
  
  def equalization(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
      #  loop over the channels and apply histogram equalization to each channel
        equalized = np.zeros_like(image)
        for channel in range(3):
            equalized[:, :, channel] = cv2.equalizeHist(image[:, :, channel])
        return equalized
    else:
        raise ValueError("Input image must be either grayscale or RGB.")
 
  
  def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

  def edge_detection(image, threshold=100, method="canny"):
    if method == "canny":
        return cv2.Canny(image, threshold, threshold * 2)
    elif method == "sobel":
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        return np.uint8(np.sqrt(sobelx ** 2 + sobely ** 2))
    elif method == "robert":
        robertsx = np.array([[1, 0], [0, -1]])
        robertsy = np.array([[0, 1], [-1, 0]])
        roberts_x = cv2.filter2D(image, -1, robertsx)
        roberts_y = cv2.filter2D(image, -1, robertsy)
        return np.uint8(np.sqrt(roberts_x ** 2 + roberts_y ** 2))
    
  def segmentation(image, k=2):
            # use k-means method
      Z = image.reshape((-1, 3))
      Z = np.float32(Z)
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
      _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
      center = np.uint8(center)
      res = center[label.flatten()]
      return res.reshape((image.shape))
    
