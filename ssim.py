from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compare_images(imageA, imageB, title):
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


# Loading the images - Original, R-Gap Version, Evol-RGap version
original = cv2.imread("images_compare/evol5.png")
#rgapimg = cv2.imread("images_compare/5logre.png")
rgapimg = cv2.imread("images_compare/5hingergap.png")
#evolimg = cv2.imread("images_compare/rescale_reconstructed.png")
evolimg = cv2.imread("images_compare/5hingerergap.png")

# Converting to Grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
rgapimg = cv2.cvtColor(rgapimg, cv2.COLOR_BGR2GRAY)
evolimg = cv2.cvtColor(evolimg, cv2.COLOR_BGR2GRAY)

# Initializing the figures
fig = plt.figure("Images")
# images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
images = ("Original", original), ("R-Gap", rgapimg), ("Evol-RGAP", evolimg)

# Looping over the figures
for (i, (name, image)) in enumerate(images):
    # show the image
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")

# Displaying the figures
plt.show()
compare_images(original, original, "Original vs. Original")
compare_images(original, rgapimg, "Original vs. R-Gap")
compare_images(original, evolimg, "Original vs. Evol-RGap")
