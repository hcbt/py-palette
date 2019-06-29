# imports
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2

# arguments parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to image")
ap.add_argument("-c", "--clusters", required = True, type = int, help = "Amount of clusters")
ap.add_argument("-o", "--output", required = True, help = "Path to output")
args = vars(ap.parse_args())

# load image and convert it to rgb for matplotlib
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image into a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)

# output image
bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)
output = cv2.imwrite(args["output"], bar)