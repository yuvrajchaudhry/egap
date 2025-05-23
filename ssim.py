from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB, title):
    imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0])) # For DLG

    s = ssim(imageA, imageB)
    m = mse(imageA, imageB)

    fig = plt.figure(title)
    plt.suptitle("SSIM: %.2f, MSE: %.2f" % (s,m))

    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    plt.show()


# Loading the images - Original, R-Gap Version, Evol-RGap version, New Third Image
original = cv2.imread("images_compare/origin26.png") # Replace with your original image path
rgapimg = cv2.imread("images_compare/rehuber.png") # Replace with your first image path
egapimg = cv2.imread("images_compare/requantile.png") # Replace with your second image path
dlgimg = cv2.imread("images_compare/remae.png")  # Replace with your third image path

# Converting to Grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
rgapimg = cv2.cvtColor(rgapimg, cv2.COLOR_BGR2GRAY)
egapimg = cv2.cvtColor(egapimg, cv2.COLOR_BGR2GRAY)
dlgimg = cv2.cvtColor(dlgimg, cv2.COLOR_BGR2GRAY)

# Initializing the figures
fig = plt.figure("Images")
images = [("Original", original), ("R-GAP", rgapimg), ("E-GAP", egapimg), ("DLG Image", dlgimg)]

# Looping over the figures
for (i, (name, image)) in enumerate(images):
    ax = fig.add_subplot(1, len(images), i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")

# Displaying the figures
plt.show()

# Comparing the original with each other image
compare_images(original, rgapimg, "Original vs. R-GAP")
compare_images(original, egapimg, "Original vs. E-GAP")
compare_images(original, dlgimg, "Original vs. DLG")
