from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    # s = ssim(imageA, imageB)
    s = ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()


# load the images -- the Original, the R-Gap Version, the Evol-RGap version
# original = cv2.imread("images/origin5.png")
# contrast = cv2.imread("images/rgap5re.png")
# shopped = cv2.imread("images/evol5re.png")
original = cv2.imread("images_compare/origin6.png")
rgapimg = cv2.imread("images_compare/rgap6re.png")
evolimg = cv2.imread("images_compare/evol6re.png")

# convert the images to grayscale
# original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
# shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
rgapimg = cv2.cvtColor(rgapimg, cv2.COLOR_BGR2GRAY)
evolimg = cv2.cvtColor(evolimg, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
# images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
images = ("Original", original), ("R-Gap", rgapimg), ("Evol-RGAP", evolimg)

# loop over the images
for (i, (name, image)) in enumerate(images):
    # show the image
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")

# show the figure
plt.show()

# compare the images
compare_images(original, original, "Original vs. Original")
compare_images(original, rgapimg, "Original vs. R-Gap")
compare_images(original, evolimg, "Original vs. Evol-RGap")
