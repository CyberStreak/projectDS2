# Einlesen eines Bildes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

image = 'images/archive/bloodcells_dataset/eosinophil/EO_27.jpg'

#plot Bild
img = mpimg.imread(image)
plt.imshow(img)
plt.show()


# OpenCV Bild lesen
image_bgr = cv2.imread(image, cv2.IMREAD_COLOR)

# Converieren in RGB
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis("off")


# 2D-Convolution Filter, vertikale Richtung, ohne Bildveränderung # Create kernel
kernel = np.array([[1, 1, 1],
[0, 0, 0], [-1, -1, -1]])

# Sharpen image
image_conv1 = cv2.filter2D(image, -1, kernel)
plt.imshow(image_conv1)
plt.axis("off")
plt.title("Convolutional Filter vertikal")
plt.show()
# Bild schärfen
kernel = np.array([[0, -1.2, 0], [-1.2, 6,-1.2],
[0, -1.2, 0]])

# Sharpen image
image_sharp = cv2.filter2D(image, -1, kernel)
plt.imshow(image_sharp)
plt.axis("off")

# 2D-Convolution Filter auf geschärftes Bild
kernel = np.array([[1, 1, 1], [0, 0, 0],
[-1, -1, -1]])


# Sharpen image
image_conv2 = cv2.filter2D(image_sharp, -1, kernel)
plt.imshow(image_conv2)
plt.axis("off")
plt.title("Convolutional Filter vertikal, auf geschärftes Bild")
plt.show()
# Image blur
image_blur = cv2.blur(image, (10,10))
plt.imshow(image_blur)

# 2D-Convolution Filter auf geblurtes Bild
kernel = np.array([[1, 1, 1], [0, 0, 0],
[-1, -1, -1]])

# Sharpen image
image_conv2 = cv2.filter2D(image_blur, -1, kernel)
plt.imshow(image_conv2)
plt.axis("off")
plt.title("Convolutional Filter vertikal, Bild blur")
plt.show()
# Bild mit verbessertem Kontrast
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
image_contrast = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
plt.imshow(image_contrast)
plt.axis("off")
# 2D-Convolution Filter auf Contrast-Ausgeglichenes Bild
kernel = np.array([[1, 1, 1], [0, 0, 0],
[-1, -1, -1]])
# Sharpen image
image_conv2 = cv2.filter2D(image_contrast, -1, kernel)
plt.imshow(image_conv2)
plt.axis("off")
plt.title("Convolutional Filter vertikal, Bild Contrast Verbessert")
plt.show()

# Bild mit nur Rot-Anteil
img_red = image[:, :, 0]
plt.imshow(img_red, cmap="Reds")
plt.title("Convolutional Filter vertikal, Bild Contrast Verbessert, Rotanteil")
plt.show()