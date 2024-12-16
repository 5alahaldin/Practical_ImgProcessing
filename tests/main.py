import sys
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from algorithms import ImageProcessingAlgorithms
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtWidgets import (
  QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
  QPushButton, QSlider, QWidget, QScrollArea, QFrame, QSpacerItem, QSizePolicy, QFileDialog
)
import uuid

LEFT_PANEL_WIDTH = 420
ICON_SIZE = 25
CACHE_DIR = "cache"
CACHE_IMAGE_PATH = os.path.join(CACHE_DIR, "original_image.png")

ICON_PATHS = {
  "Undo": "assets/icons/undo.svg",
  "Redo": "assets/icons/redo.svg",
  "Reset": "assets/icons/reset.svg",
  "Remove": "assets/icons/remove.svg",
  "Upload": "assets/icons/upload.svg",
  "Download": "assets/icons/download.svg",
  "Histogram": "assets/icons/histogram.svg",
}

class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("PIXELS")
    self.setGeometry(150, 150, 1200, 700)
    self.setFixedSize(1200, 700)

    # Central Widget
    central_widget = QWidget()
    self.setCentralWidget(central_widget)

    # Main Layout
    main_layout = QHBoxLayout(central_widget)
    main_layout.setContentsMargins(10, 10, 10, 10)
    main_layout.setSpacing(10)

    # Left Panel
    self.left_panel = self.create_left_panel()
    main_layout.addWidget(self.left_panel)

    # Right Panel (Image Display Area)
    self.right_panel = self.create_right_panel()
    main_layout.addWidget(self.right_panel, stretch=1)

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Initialize undo and redo stacks
    self.undo_stack = []
    self.redo_stack = []

  def create_left_panel(self):
    """Create the left panel with icons and algorithms."""
    left_panel = QWidget()
    left_panel.setFixedWidth(LEFT_PANEL_WIDTH)  # Set fixed width for left panel
    left_panel.setStyleSheet("background-color: white; border-radius: 10px;")
    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(10, 10, 10, 10)
    left_layout.setSpacing(10)

    # Icon Layout
    icon_layout = QHBoxLayout()
    icon_layout.setSpacing(10)
    for name, path in ICON_PATHS.items():
      icon_button = QPushButton()
      icon_button.setIcon(QIcon(path))
      icon_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
      icon_button.setToolTip(name)
      icon_button.setFlat(True)
      icon_button.setFixedSize(50, 50)
      icon_button.setStyleSheet(
        """
        QPushButton {
          border: none;
          background-color: ;
          border-radius: 5px;
          padding: 5px;
        }
        QPushButton:hover {
          background-color: #e0e0e0;
        }
        QPushButton:pressed {
          background-color: #d0d0d0;
        }
        """
      )
      if name == "Undo":
        icon_button.clicked.connect(self.undo_action)
      elif name == "Redo":
        icon_button.clicked.connect(self.redo_action)
      elif name == "Reset":
        icon_button.clicked.connect(self.reset_action)
      elif name == "Remove":
        icon_button.clicked.connect(self.remove_action)
      elif name == "Upload":
        icon_button.clicked.connect(self.upload_action)
      elif name == "Histogram":
        icon_button.clicked.connect(self.show_histogram)
      elif name == "Download":
        icon_button.clicked.connect(self.download_action)

      icon_layout.addWidget(icon_button)

    left_layout.addLayout(icon_layout)

    # Scrollable Algorithm Section
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setStyleSheet("border: none;")

    algorithm_widget = QWidget()
    scroll_area.setWidget(algorithm_widget)

    algorithm_layout = QVBoxLayout(algorithm_widget)
    algorithm_layout.setContentsMargins(10, 10, 10, 10)
    algorithm_layout.setSpacing(10)

    algorithms = {
      "To Gray": {},
      "Add Noise": {
          "Intensity": (0, 100)
      },
      "Invert": {},
      "Contrast Stretching": {
        "Min Value": (0, 255),
        "Max Value": (0, 255),
      },
      "Equalization": {},
      "Log Transform": {
        "Constant (c)": (1, 40),
      },
      "Gaussian Blur": {
        "Kernel Size": (3, 9),
        "Sigma": (0, 5),
      },
      "Edge Detection (Canny)": {
          "Threshold" : (0, 255)
      },
      "Edge Detection (Sobel)": {
          "Threshold" : (0, 255)
      }, 
      "Edge Detection (Roberts)": {},
      "Median Filter": {
        "Kernel Size": (3, 9),
      },
      "Average Filter": {
        "Kernel Size": (3, 9),
      },
      "Segmentation (K-Means)": {
          "K": (2, 10)
      },
      # "Morphological Transformations": {
      #     "Method": ["Erosion", "Dilation", "Opening", "Closing"],
      #   "Kernel Size": (3, 9),
      # },
      # "Interpolation": {
      #     "Method": ["Nearest Neighbor", "Bilinear", "Bicubic"]
      # },

    }

    for algo, settings in algorithms.items():
      algo_label = QLabel(algo)
      algo_label.setFont(QFont("Arial", 12, QFont.Bold))
      algo_label.setStyleSheet("margin: 5px 0; color: #333;")
      algorithm_layout.addWidget(algo_label)

      for setting_name, value in settings.items():
        setting_layout = QHBoxLayout()
        setting_label = QLabel(setting_name)
        setting_label.setStyleSheet("color: #555;")
        setting_layout.addWidget(setting_label)
        
        if isinstance(value, tuple):
          # If it's a range (min, max), create a slider
          slider = QSlider(Qt.Horizontal)
          slider.setMinimum(value[0])
          slider.setMaximum(value[1])
          slider.setValue((value[0] + value[1]) // 2)
          slider.setTickInterval(2)  # Set tick interval to 2 to skip even numbers
          slider.setSingleStep(2)  # Set step size to 2 to skip even numbers
          slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
              background: #ccc;
              height: 4px;
            }
            QSlider::handle:horizontal {
              background: #555;
              border: 1px solid #333;
              width: 10px;
              margin: -5px 0;
              border-radius: 5px;
            }
            """
          )
          value_label = QLabel(str(slider.value()))
          value_label.setStyleSheet("color: #555;")
          slider.valueChanged.connect(lambda value, label=value_label: label.setText(str(value)))
          setting_layout.addWidget(slider)
          setting_layout.addWidget(value_label)
          settings[setting_name] = (slider, value_label)  # Store the slider and label in settings

        elif isinstance(value, list):
          # If it's a list of options, create a QPushButton for each option
          for option in value:
            button_text = str(option)
            button = QPushButton(button_text)
            button.setStyleSheet(
              """
              QPushButton {
                color: #555;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
              }
              QPushButton:hover {
                background-color: #e0e0e0;
              }
              QPushButton:pressed {
                background-color: #d0d0d0;
              }
              """
            )
            setting_layout.addWidget(button)
            # Optionally, connect the button to a function if needed
            # button.clicked.connect(lambda checked, opt=option: self.some_function(opt))
        
        algorithm_layout.addLayout(setting_layout)

      apply_button_layout = QHBoxLayout()
      apply_button_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding))
      apply_button = QPushButton("Apply")
      apply_button.setStyleSheet(
        """
        QPushButton {
          background-color: #0078D4;
          color: white;
          font-weight: bold;
          padding: 5px 10px;
          border-radius: 5px;
        }
        QPushButton:hover {
          background-color: #005BB5;
        }
        QPushButton:pressed {
          background-color: #003A75;
        }
        """
      )
      apply_button.clicked.connect(lambda checked, algo=algo, settings=settings: self.apply_algorithm(algo, settings))
      apply_button_layout.addWidget(apply_button)
      algorithm_layout.addLayout(apply_button_layout)

      # Add separator between algorithms
      separator = QFrame()
      separator.setFrameShape(QFrame.HLine)
      separator.setFrameShadow(QFrame.Sunken)
      algorithm_layout.addWidget(separator)
      

    algorithm_layout.addStretch()
    left_layout.addWidget(scroll_area)
    return left_panel

  def create_right_panel(self):
    """Create the right panel for image display."""
    self.image_label = QLabel("Image Display Area")
    self.image_label.setFixedSize(750, 680)
    self.image_label.setStyleSheet(
      """
      background-color: #ffffff;
      border: 2px solid #ccc;
      border-radius: 10px;
      color: #555;
      font-size: 16px;
      """
    )
    self.image_label.setAlignment(Qt.AlignCenter)
    return self.image_label

  def undo_action(self):
    if len(self.undo_stack) > 1:
      self.redo_stack.append(self.undo_stack.pop())
      self.display_image(self.undo_stack[-1])
    else:
      print("Cannot undo beyond the uploaded image.")

  def redo_action(self):
    if self.redo_stack:
      self.undo_stack.append(self.redo_stack.pop())
      self.display_image(self.undo_stack[-1])
      

  def reset_action(self):
    print("Reset action triggered.")
    self.clear_cache(keep_original=True)
    self.undo_stack.clear()
    self.redo_stack.clear()
    if os.path.exists(CACHE_IMAGE_PATH):
      self.undo_stack.append(CACHE_IMAGE_PATH)
      self.display_image(CACHE_IMAGE_PATH)
  
  def remove_action(self):
    print("Remove action triggered.")
    self.image_label.clear()
    self.clear_cache()
    self.undo_stack.clear()
    self.redo_stack.clear()

  def upload_action(self):
    """Handle the upload button click."""
    file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
    if file_name:
      # Clear the cache directory before uploading a new image
      self.clear_cache()
      
      # Ensure the cache directory exists
      os.makedirs(CACHE_DIR, exist_ok=True)
      
      # Copy the new image to the cache directory
      shutil.copy(file_name, CACHE_IMAGE_PATH)
      
      # Display the new image
      self.display_image(CACHE_IMAGE_PATH)
      
      # Update the undo stack
      self.undo_stack.append(CACHE_IMAGE_PATH)
      self.redo_stack.clear()
      
  def display_image(self, file_path):
    """Display the uploaded image in the right panel."""
    pixmap = QPixmap(file_path)
    if pixmap.isNull():
      print("Error: Unable to load image.")
    else:
      self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

  def download_action(self):
    """Handle the download button click."""
    if not self.undo_stack:
      print("No image to download.")
      return

    # Get the last edited image
    image_path = self.undo_stack[-1]

    # Open a file dialog to save the image
    save_path, _ = QFileDialog.getSaveFileName(self, "Save Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
    if save_path:
      shutil.copy(image_path, save_path)
      print(f"Image saved to {save_path}")

  def apply_algorithm(self, algorithm, settings):
    if not self.undo_stack:
      print("No image to process.")
      return

    if algorithm == "To Gray":
      self.apply_to_gray()

    elif algorithm == "Add Noise":
      intensity_slider = settings["Intensity"][0]
      intensity_value = intensity_slider.value()
      self.apply_add_noise(intensity_value)

    elif algorithm == "Invert":
      self.apply_invert_filter()

    elif algorithm == "Contrast Stretching":
      min_slider = settings["Min Value"][0]
      max_slider = settings["Max Value"][0]
      min_value = min_slider.value()
      max_value = max_slider.value()
      self.apply_contrast_stretching(min_value, max_value)

    elif algorithm == "Equalization":
      self.apply_equalization()

    elif algorithm == "Log Transform":
      constant_slider = settings["Constant (c)"][0]
      constant_value = constant_slider.value()
      self.apply_log_transform(constant_value)

    elif algorithm == "Gaussian Blur":
      kernel_slider = settings["Kernel Size"][0]
      sigma_slider = settings["Sigma"][0]
      kernel_size = kernel_slider.value()
      sigma = sigma_slider.value()
      self.apply_gaussian_blur(kernel_size, sigma)

    elif algorithm == "Edge Detection (Canny)":
      threshold_slider = settings["Threshold"][0]
      threshold = threshold_slider.value()
      self.apply_edge_detection_canny(threshold)

    elif algorithm == "Edge Detection (Sobel)":
      threshold_slider = settings["Threshold"][0]
      threshold = threshold_slider.value()
      self.apply_edge_detection_sobel(threshold)

    elif algorithm == "Edge Detection (Roberts)":
      self.apply_edge_detection_roberts()

    elif algorithm == "Median Filter":
      kernel_slider = settings["Kernel Size"][0]
      kernel_size = kernel_slider.value()
      self.apply_median_filter(kernel_size)

    elif algorithm == "Average Filter":
      kernel_slider = settings["Kernel Size"][0]
      kernel_size = kernel_slider.value()
      self.apply_average_filter(kernel_size)

    elif algorithm == "Segmentation (K-Means)":
      k_slider = settings["K"][0]
      k = k_slider.value()
      self.apply_segmentation_kmeans(k)

# --------------------------invert-------------------------------
  def apply_invert_filter(self):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path)

    # Apply invert filter
    inverted_image = ImageProcessingAlgorithms.invert_image(image)

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, inverted_image)

    # Update undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)
    
# --------------------------contrast_stretching-------------------------------
  def apply_contrast_stretching(self, min_value, max_value):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path)

    # Apply contrast stretching
    processed_image = ImageProcessingAlgorithms.contrast_stretching(image, min_value, max_value)

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, processed_image)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

# --------------------------log_transform-------------------------------
  def apply_log_transform(self, c):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path)

    # Apply log transform
    log_transformed_image = ImageProcessingAlgorithms.log_transform(image, c)

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, log_transformed_image)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

# -------------------------equalization--------------------------------
  def apply_equalization(self):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path)

    # Apply histogram equalization
    equalized_image = ImageProcessingAlgorithms.equalization(image)

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, equalized_image)

    # Update undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

# -----------------------gaussian_blur----------------------------------
  def apply_gaussian_blur(self, kernel_size, sigma):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
      kernel_size += 1

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path)

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, blurred_image)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

# -----------------------edge_detection----------------------------------

  def apply_edge_detection_canny(self, threshold):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold, threshold * 2)

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, edges)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

  def apply_edge_detection_sobel(self, threshold):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Sobel edge detection
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edges = np.uint8(np.sqrt(sobelx ** 2 + sobely ** 2))

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, edges)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

  def apply_edge_detection_roberts(self):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Roberts edge detection
    robertsx = np.array([[1, 0], [0, -1]])
    robertsy = np.array([[0, 1], [-1, 0]])
    roberts_x = cv2.filter2D(image, -1, robertsx)
    roberts_y = cv2.filter2D(image, -1, robertsy)
    edges = np.uint8(np.sqrt(roberts_x ** 2 + roberts_y ** 2))

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, edges)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

# -----------------------to_gray----------------------------------
  def apply_to_gray(self):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, gray_image)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

# -----------------------add_noise----------------------------------
  def apply_add_noise(self, intensity):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path)

    # Generate noise
    noise = np.random.normal(0, intensity, image.shape).astype(np.uint8)

    # Add noise to the image
    noisy_image = cv2.add(image, noise)

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, noisy_image)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

# -----------------------median_filter----------------------------------
  def apply_median_filter(self, kernel_size):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
      kernel_size += 1

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path)

    # Apply median filter
    filtered_image = cv2.medianBlur(image, kernel_size)

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, filtered_image)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

# -----------------------average_filter----------------------------------

  def apply_average_filter(self, kernel_size):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
      kernel_size += 1

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path)

    # Apply average filter
    filtered_image = cv2.blur(image, (kernel_size, kernel_size))

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, filtered_image)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)
    
# -----------------------segmentation_kmeans----------------------------------
  def apply_segmentation_kmeans(self, k):
    if not self.undo_stack:
      print("No image to process.")
      return

    # Read the last edited image
    image_path = self.undo_stack[-1]
    image = cv2.imread(image_path)

    # Apply K-Means segmentation
    segmented_image = ImageProcessingAlgorithms.segmentation(image, k)

    # Generate a unique filename for the processed image
    processed_image_path = os.path.join(CACHE_DIR, f"processed_image_{uuid.uuid4().hex}.png")

    # Save the processed image
    cv2.imwrite(processed_image_path, segmented_image)

    # Update the undo stack
    self.undo_stack.append(processed_image_path)
    self.redo_stack.clear()

    # Display the processed image
    self.display_image(processed_image_path)

  def show_histogram(self):
    """Display the histogram and frequency domain of the currently displayed image."""
    if not self.undo_stack:
        print("No image to display histogram.")
        return

    # Read the current image
    current_image_path = self.undo_stack[-1]
    image = cv2.imread(current_image_path)

    # Create a figure for the histograms and frequency domain
    plt.figure(figsize=(8, 8))

    # Check if the image is grayscale or color
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Grayscale image
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.subplot(2, 2, 3)
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(histogram, color='gray')
        plt.xlim([0, 256])

        # Display the image in the spatial domain
        plt.subplot(2, 2, 1)
        plt.title("Spatial Domain")
        plt.axis('off')  # Hide the axes
        plt.imshow(image, cmap='gray')

        # Compute the frequency domain
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        # Display the frequency domain
        plt.subplot(2, 2, 2)
        plt.title("Frequency Domain")
        plt.axis('off')  # Hide the axes
        plt.imshow(magnitude_spectrum, cmap='gray')
    else:
        # Color image
        colors = ('b', 'g', 'r')
        plt.subplot(2, 2, 3)
        plt.title("Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        for i, color in enumerate(colors):
            histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(histogram, color=color)
            plt.xlim([0, 256])

        # Display the image in the spatial domain
        plt.subplot(2, 2, 1)
        plt.title("Spatial Domain")
        plt.axis('off')  # Hide the axes
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Convert to grayscale for frequency domain
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute the frequency domain
        f = np.fft.fft2(gray_image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        # Display the frequency domain
        plt.subplot(2, 2, 2)
        plt.title("Frequency Domain")
        plt.axis('off')  # Hide the axes
        plt.imshow(magnitude_spectrum, cmap='gray')

        # Display the grayscale histogram for the color image
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        plt.subplot(2, 2, 4)
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(histogram, color='gray')
        plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

  def clear_cache(self, keep_original=False):
    """Remove all files in the cache directory."""
    for file in os.listdir(CACHE_DIR):
      file_path = os.path.join(CACHE_DIR, file)
      if os.path.isfile(file_path):
        if keep_original and file_path == CACHE_IMAGE_PATH:
          continue
        os.remove(file_path)

def cleanup_cache():
  if os.path.exists(CACHE_IMAGE_PATH):
    os.remove(CACHE_IMAGE_PATH)
  for file in os.listdir(CACHE_DIR):
    file_path = os.path.join(CACHE_DIR, file)
    if os.path.isfile(file_path):
      os.remove(file_path)
  if os.path.exists(CACHE_DIR) and not os.listdir(CACHE_DIR):
    os.rmdir(CACHE_DIR)

if __name__ == "__main__":
  app = QApplication(sys.argv)
  app.setStyle("Fusion")
  window = MainWindow()
  window.show()
  app.aboutToQuit.connect(cleanup_cache)
  sys.exit(app.exec_())