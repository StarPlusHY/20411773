import numpy as np
from sklearn.cluster import KMeans
import cv2 as cv
from skimage import morphology
import math
import os
# Downsampling
def downsample1(imageafter3channels):
    h, w, _ = imageafter3channels.shape
    steplength = 1  # Sampling interval; effectively no downsampling. This can be removed if unused
    data_new = imageafter3channels[0:h+1:steplength, ::steplength, :]  # Same as [:,:,:] with step length
    Ihsv = data_new
    img_downsample = np.double(Ihsv)  # Convert the result to double-precision floating-point
    # Sample the image after homomorphic filtering
    return img_downsample


def homomorphic_filter(src, d0=1, r1=0.1, rh=0.5, c=0.3):
    height, width, _ = src.shape
    imageafter3channels = np.zeros((height, width, 3))

    for tongdao in range(3):  # For each color channel
        image = np.float64(src[:, :, tongdao])
        rows, cols = image.shape  # Get image dimensions: rows (M) and columns (N)

        # Perform Fourier Transform
        gray_fft = np.fft.fft2(image)

        # Shift zero frequency component to the center
        gray_fftshift = np.fft.fftshift(gray_fft)

        # Create a zero matrix with the same shape as gray_fftshift
        dst_fftshift = np.zeros_like(gray_fftshift)

        # Create coordinate grid
        M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))

        # Apply frequency-domain enhancement function
        D = np.sqrt(M ** 2 + N ** 2)
        Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
        dst_fftshift = Z * gray_fftshift
        dst_fftshift = (rh - r1) * dst_fftshift + r1

        # Perform inverse Fourier Transform
        dst_ifftshift = np.fft.ifftshift(dst_fftshift)
        dst_ifft = np.fft.ifft2(dst_ifftshift)

        # Extract the real part
        dst = np.real(dst_ifft)

        # Normalize the result to the range [0, 255]
        max_num = dst.max()
        min_num = dst.min()
        xrange = max_num - min_num
        img_after = np.zeros((rows, cols), 'uint8')
        for i in range(rows):
            for j in range(cols):
                img_after[i, j] = np.uint8(255 * (dst[i, j] - min_num) / xrange)

        # cv.imshow('image', np.uint8(image))
        # cv.imshow('img_after', np.uint8(img_after))
        # cv.waitKey(0)

        imageafter3channels[:, :, tongdao] = img_after

        # # Clip negative values to 0 and values over 255 to 255 (optional)
        # dst = np.uint8(np.clip(dst, 0, 255))

    return imageafter3channels

















# def homomorphic_filter(part_of_original_Img):
#     height, width, _ = part_of_original_Img.shape
#     imageafter3channels = np.zeros((height, width, 3))
#     for tongdao in range(3):  # Iterate over 3 color channels
#         rH = 0.5  # Configure homomorphic filter parameters
#         rL = 0.1
#         c = 0.3  # c lies between rH and rL, usually around 0.2
#         D0 = 0.1
#         image = np.float64(part_of_original_Img[:, :, tongdao])
#         M, N = image.shape  # Get image dimensions: M = rows, N = columns
#         img_log = np.log(image + 1)  # Apply logarithmic transformation (convert double to float)

#         # Shift image to center in frequency domain, implemented without exponentials
#         img_py = np.zeros((M, N))  # Create a zero matrix of size M x N
#         for i in range(M):  # Replace exponential operation with alternating sign shift
#             for j in range(N):
#                 if ((i + j) % 2 == 0):
#                     img_py[i, j] = img_log[i, j]
#                 else:
#                     img_py[i, j] = -1 * img_log[i, j]

#         # Apply Fourier transform to the centered image
#         img_py_fft = np.fft.fft2(img_py)  # Equivalent to MATLAB's fft2(img_py)

#         # Create homomorphic filter function
#         img_tt = np.zeros((M, N))  # Initialize the filter matrix
#         deta_r = rH - rL  # Difference between high and low frequency gain
#         D = D0 ** 2
#         m_mid = np.floor(M / 2)  # Get center row index
#         n_mid = np.floor(N / 2)  # Get center column index

#         for i in range(M):
#             for j in range(N):
#                 dis = ((i - m_mid) ** 2 + (j - n_mid) ** 2)  # Distance squared to center
#                 img_tt[i, j] = deta_r * (1 - np.exp((-c) * (dis / D))) + rL  # Filter response at (i, j)

#         # Apply the filter in frequency domain
#         img_temp = img_py_fft * img_tt

#         # Apply inverse Fourier transform, take the real part, then absolute value
#         img_temp = abs(np.real(np.fft.ifft2(img_temp)))

#         # Exponentiate the result to revert log transform
#         img_temp = np.exp(img_temp) - 1

#         # cv.imshow('img_temp', np.uint8(img_temp))
#         # cv.waitKey(0)

#         # Normalize the result to [0, 255]
#         max_num = img_temp.max()
#         min_num = img_temp.min()
#         xrange = max_num - min_num
#         img_after = np.zeros((M, N), 'uint8')
#         for i in range(M):
#             for j in range(N):
#                 img_after[i, j] = np.uint8(255 * (img_temp[i, j] - min_num) / xrange)  # Normalize filtered result

#         cv.imshow('image', np.uint8(image))
#         cv.imshow('img_after', np.uint8(img_after))
#         cv.waitKey(0)

#         # Assign the result to the corresponding channel
#         imageafter3channels[:, :, tongdao] = img_after

#     return imageafter3channels



def turntovector(img_downsample):
    # Convert the image into an N×3 vector for clustering
    height, width, _ = img_downsample.shape
    vector_of_data = np.zeros((height * width, 3))
    for i in range(height):
        for j in range(width):
            k = width * i
            vector_of_data[k + j, 0] = img_downsample[i, j, 0]
            vector_of_data[k + j, 1] = img_downsample[i, j, 1]
            vector_of_data[k + j, 2] = img_downsample[i, j, 2]
    return vector_of_data


# np.set_printoptions(threshold=np.nan)
def bwaeraopen(image, size):
    '''
    @image: single-channel binary image of type uint8  
    @size: remove connected components smaller than this area (white regions on a black background)
    '''
    output = image.copy()
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(image)
    for i in range(1, nlabels - 1):
        regions_size = stats[i, 4]
        if regions_size < size:
            x0 = stats[i, 0]
            y0 = stats[i, 1]
            x1 = stats[i, 0] + stats[i, 2]
            y1 = stats[i, 1] + stats[i, 3]
            for row in range(y0, y1):
                for col in range(x0, x1):
                    if labels[row, col] == i:
                        output[row, col] = 0
    return output


def dectetion_based_on_clustering(vector_of_data, number_of_categories, threshold, height, weight):
    # Crack detection based on clustering

    # KMeans parameters reference:
    # n_clusters: number of clusters to form
    # init: method for initialization (‘k-means++’ or ‘random’ or user-specified)
    # n_init: number of time the k-means algorithm will be run with different centroid seeds
    # max_iter: maximum number of iterations of the k-means algorithm for a single run
    # tol: relative tolerance with regards to inertia to declare convergence
    # precompute_distances: whether to precompute distances (deprecated)
    # verbose: verbosity mode
    # random_state: seed for the random number generator
    # copy_x: whether to copy X before clustering
    # n_jobs: number of jobs to run in parallel
    # algorithm: algorithm to use (‘auto’, ‘full’, or ‘elkan’)

    estimator = KMeans(n_clusters=number_of_categories, max_iter=350, n_init=4).fit(vector_of_data)  # Build KMeans model

    '''Label each sample by cluster'''
    serial_number = estimator.labels_
    img = serial_number.reshape((height, weight))
    # cv.imshow("img", np.uint8(img * 60))

    '''Display cluster centroids'''
    huidu = estimator.cluster_centers_

    '''This can also be seen as the loss: sum of squared distances of samples to their closest cluster center'''
    # inertia = estimator.inertia_
    # serial_number, _ = KMeans(vector_of_data, number_of_categories)

    unique_serial_number = np.unique(serial_number)  # Unique cluster labels
    unique_serial_number_add = np.append(unique_serial_number, unique_serial_number.max() + 1)  # np.histogram requires n+1 bins
    M, _ = np.histogram(serial_number, unique_serial_number_add)  # M: frequency count of each label

    huiduxu = np.zeros(number_of_categories)
    for i in range(number_of_categories):
        r = huidu[i, 0]
        g = huidu[i, 1]
        b = huidu[i, 2]
        huiduxu[i] = r * 0.299 + g * 0.587 + b * 0.114  # Convert RGB centroid to grayscale

    # Determine the darkest cluster and the smallest one
    dic3 = dict(zip(unique_serial_number, huiduxu))
    minimum3 = huiduxu.min()
    number_of_minimum3 = list(dic3.keys())[list(dic3.values()).index(minimum3)]

    dic2 = dict(zip(unique_serial_number, M))
    minimum2 = M.min()
    number_of_minimum2 = list(dic2.keys())[list(dic2.values()).index(minimum2)]

    if number_of_minimum2 == number_of_minimum3:
        number_of_minimum = number_of_minimum2
    else:
        b = M.copy()
        ii = number_of_minimum2
        b[ii] = max(b)
        dic4 = dict(zip(unique_serial_number, b))
        minimum4 = b.min()
        ii = list(dic4.keys())[list(dic4.values()).index(minimum4)]
        number_of_minimum = ii

    # If the darkest cluster is very dark, prioritize it
    if minimum3 < 50:
        c = huiduxu.copy()
        ii = number_of_minimum3
        c[ii] = max(c)
        dic5 = dict(zip(unique_serial_number, c))
        minimum5 = c.min()
        ii = list(dic5.keys())[list(dic5.values()).index(minimum5)]
        number_of_minimum = ii

    # Create binary label image: crack = 100, others = 0
    new_serial_number = serial_number.copy()
    same_serial_1 = np.where(serial_number == number_of_minimum)  # Indices of crack-labeled points
    same_serial_2 = np.where(serial_number != number_of_minimum)
    new_serial_number[same_serial_1] = 100
    new_serial_number[same_serial_2] = 0

    cluster_result_img = (new_serial_number.reshape(height, weight)) / 100  # Normalized result for viewing
    cluster_result_img_for_observation = np.uint8(cluster_result_img * 100)  # For visualization

    # Remove small regions using a 4-connected component filter
    detection_img = bwaeraopen(cluster_result_img_for_observation, threshold)
    detection_img = np.double(detection_img)

    # Get coordinates of detected cracks
    [x_crack, y_crack] = np.where(detection_img == 1)
    return detection_img, x_crack, y_crack


def FillHole(mask):
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # Create a black image
        img_contour = cv.drawContours(drawing, contours, i, (255, 255, 255), -1)  # Fill the contour
        contour_list.append(img_contour)
    out = sum(contour_list)
    return out


# Edge detection
def edgedetection(detection_img_new):
    detection_img_new = np.uint8(detection_img_new)
    # cv.imshow('detection_img_new',detection_img_new)

    # Optional dilation and morphological operations (currently disabled)
    # kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))  # Dilation
    # detection_img_new = cv.dilate(detection_img_new, kernel_dilate, iterations=1)

    # Kernel example 1
    # list_kernel = [[1, 1, 1],
    #                [1, 1, 1],
    #                [1, 1, 1]]
    # kernellist = np.array(list_kernel, np.uint8)

    # Kernel example 2
    # list_kernel = [[1, 1, 1, 1, 1],
    #                [1, 2, 2, 2, 1],
    #                [1, 2, 3, 2, 1],
    #                [1, 2, 2, 2, 1],
    #                [1, 1, 1, 1, 1]]
    # kernellist = np.array(list_kernel, np.uint8)

    # kernelclose = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # Define structural element
    # detection_img_new = cv.morphologyEx(detection_img_new, cv.MORPH_CLOSE, kernellist)  # Closing operation

    # detection_img_new = cv.morphologyEx(detection_img_new, cv.MORPH_OPEN, kernelclose)  # Opening operation
    # kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))  # Erosion
    # detection_img_new = cv.erode(detection_img_new, kernel_erode, iterations=1)

    detection_img_new = FillHole(detection_img_new)  # Fill holes in the binary image

    # cv.imshow('detection_img_new', detection_img_new)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(detection_img_new, connectivity=8)
    # num_labels: Number of labels
    # labels: each pixel's label (1, 2, 3...), indicating different connected components
    # stats: statistics of each label, a matrix with 5 columns: x, y, width, height, area
    # centroids: center points of connected components

    set_of_MinorAxislength = [100]  # Temporary list to store minor axis lengths

    for size_of_structure in range(1, num_labels):
        majorAxisLength = stats[size_of_structure][3]
        minorAxisLength = stats[size_of_structure][2]
        S = stats[size_of_structure][4]
        if (minorAxisLength ** 2 + majorAxisLength ** 2) / S > 6:
            set_of_MinorAxislength.append(
                S / math.sqrt(minorAxisLength ** 2 + majorAxisLength ** 2))  # Calculate feature length
        else:
            detection_img_new[np.where(labels == size_of_structure)] = 0  # Remove small or irregular regions

    # cv.imshow('tianchong', detection_img_new)
    minimum_of_MinorAxisLength = min(set_of_MinorAxislength) * 2
    window_size = round(minimum_of_MinorAxisLength)  # Round and assign as window size

    # Edge detection using Sobel operator
    x = cv.Sobel(detection_img_new, cv.CV_16S, 1, 0)
    y = cv.Sobel(detection_img_new, cv.CV_16S, 0, 1)
    Scale_absX = cv.convertScaleAbs(x)  # Convert scale
    Scale_absY = cv.convertScaleAbs(y)
    edge_of_detection = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)  # Blend the two directions

    return detection_img_new, edge_of_detection, window_size


'''
Burr removal
'''
def removing_burrs(image, bili=0.1, biaoji=2):
    node = []
    node_inside = []
    node_outside = []

    height, weight = image.shape  # height: image height; weight: image width
    length_burrs = min(height, weight)
    iThin = image.copy()

    for h in range(2, height - 2):
        for w in range(2, weight - 2):
            if (int(image[h, w]) == 255) and (
                int(image[h - 1, w - 1]) + int(image[h - 1, w]) + int(image[h - 1, w + 1]) +
                int(image[h, w + 1]) + int(image[h + 1, w + 1]) + int(image[h + 1, w]) +
                int(image[h + 1, w - 1]) + int(image[h, w - 1]) >= 3 * 255):
                # If current pixel is white and at least 3 of its 8 neighbors are white
                node.append((h, w))  # Store the node coordinates

                node_inside_np = np.where(image[h - 1:h + 2, w - 1:w + 2] == 255)  # Inner ring around the node
                for i in range(len(node_inside_np[0])):
                    image[node_inside_np[0][i] + h - 1, node_inside_np[1][i] + w - 1] = 0
                    iThin[node_inside_np[0][i] + h - 1, node_inside_np[1][i] + w - 1] = 0
                    node_inside.append((node_inside_np[0][i] + h - 1, node_inside_np[1][i] + w - 1))

                node_outside_np = np.where(image[h - 2:h + 3, w - 2:w + 3] == 255)  # Outer ring around the node
                for i in range(len(node_outside_np[0])):
                    image[node_outside_np[0][i] + h - 2, node_outside_np[1][i] + w - 2] = biaoji
                    node_outside.append((node_outside_np[0][i] + h - 2, node_outside_np[1][i] + w - 2))

    num_labels, labels = cv.connectedComponents(iThin)
    for label in range(num_labels):
        coords = np.where(labels == label)
        connected_region = np.where(image[coords] == biaoji, 1, 0)
        if (connected_region.sum() == 1) and (len(connected_region) <= length_burrs * bili):
            # If region has only one endpoint and is shorter than the threshold, it's a burr
            iThin[coords] = 0  # Remove burr

    for i in node:
        iThin[i] = 255  # Restore central node pixels

    for j in node_inside:
        if ((int(iThin[j[0] - 1, j[1] - 1]) + int(iThin[j[0] - 1, j[1]]) + int(iThin[j[0] - 1, j[1] + 1]) == 255) and
            (int(iThin[j[0] + 1, j[1] + 1]) + int(iThin[j[0] + 1, j[1]]) + int(iThin[j[0] + 1, j[1] - 1]) == 255)) or \
           ((int(iThin[j[0] - 1, j[1] - 1]) + int(iThin[j[0], j[1] - 1]) + int(iThin[j[0] + 1, j[1] - 1]) == 255) and
            (int(iThin[j[0] - 1, j[1] + 1]) + int(iThin[j[0], j[1] + 1]) + int(iThin[j[0] + 1, j[1] + 1]) == 255)):
            iThin[j] = 255

    return iThin

'''
Skeleton Extraction
'''
# Since the image has already been converted to a binary image in previous steps, no conversion is needed here
def skeleton_for_crack(recImg):

    list_kernel1 = [[2, 2, 2],
                    [2, 1, 2],
                    [2, 2, 2]]
    kernellist1 = np.array(list_kernel1, np.uint8)

    list_kernel2 = [[2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                    [2, 2, 1, 2, 2],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2]]
    kernellist2 = np.array(list_kernel2, np.uint8)

    kernel = np.ones(shape=[5, 5], dtype=np.uint8)
    recImg = cv.dilate(recImg, kernellist1, iterations=1)

    kernelclose = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # Define structuring element

    # recImg = cv.morphologyEx(recImg, cv.MORPH_OPEN, kernel)  # Opening (optional)
    recImg = cv.morphologyEx(recImg, cv.MORPH_CLOSE, kernellist2)  # Closing: dilation followed by erosion
    _, recImg = cv.threshold(recImg, 200, 255, cv.THRESH_BINARY)
    # cv.imshow('recImg', np.uint8(recImg))
    # cv.waitKey(0)

    # Crack skeletonization method 1: Not usable in current OpenCV version
    # skeleton = cv.ximgproc.thinning(recImg, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)

    # Crack skeletonization method 2
    recImg[recImg == 255] = 1
    skeleton0 = morphology.skeletonize(recImg)
    skeleton = skeleton0.astype(np.uint8) * 255
    # cv.imshow('skeleton', np.uint8(skeleton))

    # Crack skeletonization method 3: produces many artifacts, and may not work
    # skel, distance = morphology.medial_axis(recImg, return_distance=True)
    # dist_on_skel = distance * skel
    # skeleton = dist_on_skel.astype(np.uint8) * 255

    # Crack skeletonization method 4: extremely discontinuous
    # skeleton = np.zeros(recImg.shape, np.uint8)
    # while (True):
    #     if np.sum(recImg) == 0:
    #         break
    #     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    #     recImg = cv.erode(recImg, kernel, None, None, 1)
    #     open_dst = cv.morphologyEx(recImg, cv.MORPH_OPEN, kernel)
    #     result = recImg - open_dst
    #     skeleton = skeleton + result

    # Burr removal method 1: leaves many breakpoints (severely fragmented)
    # kernelopen = cv.getStructuringElement(cv.MORPH_CROSS, (2,1))
    # skeleton = cv.erode(skeleton, kernelopen, iterations=1)

    # Burr removal method 2: removes burrs but not very cleanly
    skeleton = removing_burrs(skeleton)
    # cv.imshow('removing', np.uint8(skeleton))

    # Find coordinates of skeleton pixels (value == 255)
    x_skeleton, y_skeleton = np.where(skeleton == 255)
    return skeleton, x_skeleton, y_skeleton


def width_calculation(edge_of_detection, skeleton_image, x_skeleton, y_skeleton, window_size, original_image,
                      set_of_distance, geduan, x_wholeimage_for_skel, y_wholeimage_for_skel,
                      number_of_blocks, size_of_imageblock, width_of_crack):
    # Crack width calculation
    size_of_mark = 1  # Size of the white marking area
    set_of_distancehere = []
    [height, width, _] = original_image.shape
    img_withmark = original_image.copy()
    need_tobe_calculated = skeleton_image.copy()

    xbaocun = []  # Temporary variable to store x-coordinates of skeleton points to be marked
    ybaocun = []  # Temporary variable to store y-coordinates

    for i in range(len(x_skeleton)):
        X = x_skeleton[i]
        Y = y_skeleton[i]
        if need_tobe_calculated[X, Y] != 0:
            # Use the skeleton point as the center to create a search window
            upper_of_window = max([X - window_size, 0])
            lower_of_window = min([X + window_size, height - 1])
            left_of_window = max([Y - window_size, 0])
            right_of_window = min([Y + window_size, width - 1])
            search_window = edge_of_detection[upper_of_window:lower_of_window, left_of_window:right_of_window]

            x_edgehere, y_edgehere = np.where(search_window == 1)
            pixels_on_edge_thiswindow = len(x_edgehere)

            if pixels_on_edge_thiswindow != 0:
                ditance_between_edge_and_skel = np.zeros(pixels_on_edge_thiswindow)  # Initialize distance array

                # Coordinate transformation
                # Transform X coordinate (global skeleton coordinate → local window coordinate)
                if 0 > X - window_size:
                    center1 = window_size - (X - window_size)
                elif X + window_size > height - 1:
                    center1 = window_size - (X + window_size - (height - 1))
                else:
                    center1 = window_size
                # Transform Y coordinate
                if 0 > Y - window_size:
                    center2 = window_size - (Y - window_size)
                elif Y + window_size > width - 1:
                    center2 = window_size - (Y + window_size - (width - 1))
                else:
                    center2 = window_size

                for i2 in range(pixels_on_edge_thiswindow):
                    ditance_between_edge_and_skel[i2] = math.sqrt(
                        (x_edgehere[i2] - center1) ** 2 + (y_edgehere[i2] - center2) ** 2)  # Euclidean distance
                    if ditance_between_edge_and_skel[i2] == 0:
                        ditance_between_edge_and_skel[i2] = 100  # Avoid zero-distance anomaly

                minimum_of_distance = min(ditance_between_edge_and_skel)
                if minimum_of_distance != 100:
                    xbaocun.append(X)
                    ybaocun.append(Y)
                    set_of_distancehere.append(2 * minimum_of_distance)  # Store calculated width

    set_of_distance.append(set_of_distancehere)
    number_of_pixels = len(set_of_distancehere)

    if number_of_pixels >= 1:
        for im in range(number_of_pixels):
            width_of_crack.append(set_of_distancehere[im])

            # Convert coordinates from current block to global image
            min_x = max(geduan + 1, geduan + xbaocun[im] - size_of_mark)
            max_x = min(geduan + xbaocun[im] + size_of_mark, height)
            min_y = max(ybaocun[im] - size_of_mark + (number_of_blocks - 1) * size_of_imageblock,
                        1 + (number_of_blocks - 1) * size_of_imageblock)
            max_y = min(ybaocun[im] + size_of_mark + (number_of_blocks - 1) * size_of_imageblock,
                        size_of_imageblock + (number_of_blocks - 1) * size_of_imageblock)

            # Mark crack location in white (255, 255, 255)
            img_withmark[min_x:max_x, min_y:max_y, 0] = 255
            img_withmark[min_x:max_x, min_y:max_y, 1] = 255
            img_withmark[min_x:max_x, min_y:max_y, 2] = 255

            # If only one block is used, this part may be redundant, but reserved
            x_wholeimage_for_skel[number_of_blocks - 1] = [i + geduan for i in xbaocun]
            y_wholeimage_for_skel[number_of_blocks - 1] = [j + (number_of_blocks - 1) * size_of_imageblock for j in ybaocun]

    return img_withmark, width_of_crack, set_of_distancehere, set_of_distance, x_wholeimage_for_skel, y_wholeimage_for_skel


def tradition_process(original_image, pixel_resolution=1, geduan=0, number_of_categories=5, threshold=1, number_of_blocks=1):
    img_withmark = original_image  # Set variable and assign value
    [height, width, _] = original_image.shape  # Get image dimensions: height and width
    size_of_imageblock = width  # Set block size based on image width

    # Set coordinate storage for block processing as list (used instead of MATLAB cell arrays)
    x_wholeimage_for_skel = [None] * round(width / size_of_imageblock)  # List of size round(width / block size)
    y_wholeimage_for_skel = [None] * round(width / size_of_imageblock)  # Same for y-coordinates

    set_of_distance = []  # List to store distances per block
    width_of_crack = []  # List to store crack widths per block

    # ========== Main processing flow (modularized functions) ==========
    # ======= Start homomorphic filtering, first process image by blocks and take one block =======

    part_of_original_Img = original_image[geduan:height + 1,
                           (size_of_imageblock * (number_of_blocks - 1)):1 + size_of_imageblock * number_of_blocks,:]




    part_of_original_Img = homomorphic_filter(part_of_original_Img)






    # cv.imshow('imageafter3channels', np.uint8(imageafter3channels))
    # cv.waitKey(0)
    # Homomorphic filtering completed. Input: local image; Output: enhanced image

    # Downsample the image after homomorphic filtering (if downsampling is needed, keep this function)

    img_downsample = downsample1(part_of_original_Img)
    # cv.imshow('img_downsample',np.uint8(img_downsample))
    # cv.waitKey(0)


    # Convert image to vector format. Input: image; Output: an N×3 vector, where each row represents a pixel
    vector_of_data = turntovector(img_downsample)

    # Perform clustering analysis to obtain the crack detection result `detection_img` as a binary image  
    # number_of_categories = kcaise(img_downsample)  # Adaptive clustering can be used here


    detection_img, x_crack, y_crack = dectetion_based_on_clustering(vector_of_data, number_of_categories, threshold,
                                                                    height, width)
    # cv.imshow('detection_img',detection_img)
    # cv.waitKey(0)
    # Input: data in vector form, number of categories, threshold, image height and width  
    # Output: binary crack detection map and corresponding coordinate positions

    detection_img_crack, edge_of_detection, window_size = edgedetection(detection_img)

    skeleton_image_crack, x_skeleton, y_skeleton = skeleton_for_crack(detection_img_crack)
    # cv.imshow('detection_img_crack', detection_img_crack)
    # cv.imshow('edge_of_detection', edge_of_detection)
    # cv.imshow('skeleton_image_crack', skeleton_image_crack)
    # cv.waitKey(0)
    img_withmark, width_of_crack, set_of_distancehere, set_of_distance, x_wholeimage_for_skel, \
    y_wholeimage_for_skel = width_calculation(edge_of_detection, skeleton_image_crack, x_skeleton, y_skeleton,
                                              window_size, original_image, set_of_distance, geduan,
                                              x_wholeimage_for_skel, y_wholeimage_for_skel, number_of_blocks,
                                              size_of_imageblock, width_of_crack)
    actual_distance = [width * pixel_resolution for width in width_of_crack]  # 实际缝宽物理尺寸等于统计值乘以图片像素的分辨率

    return detection_img_crack,skeleton_image_crack,img_withmark,len(actual_distance),width_of_crack

#
# name_of_image=r'D:\Project_tianqin\data\crops_png_432_576\DJI_Crack_0001_([0;432],[0;576]).jpg'   #设置变量，调用图片
# name_of_image=r'D:\Project_tianqin\data\crops_test/aa1baed8d3026e305e81822d68d2757.png'
# name_of_image=r'./data/demo/DJI_Crack_0018_([2592;3024],[1152;1728]).jpg'
# original_image=cv.imread(name_of_image)
# detection_img_crack,skeleton_image_crack,img_withmark,crack_len,actual_distance=tradition_process(original_image)
#
# # cv.imshow('detection_img_crack',detection_img_crack)
# cv.imshow('skeleton_image_crack',skeleton_image_crack)
# cv.imshow('img_withmark',img_withmark)
# print(crack_len)
# print(actual_distance)
# cv.waitKey(0)
def file_filter(f):
    if f[-4:] in ['.jpg', '.png', '.bmp', '.JPG']:
        return True
    else:
        return False
# crops_dir='D:\project_tianqin\data\demo_5\DJI_20220217105417_0010_Z/' # crops分割
# fileNames = os.listdir(crops_dir)
# fileNames = list(filter(file_filter, fileNames))
# for f in fileNames:
#     image = cv.imread(os.path.join(crops_dir, f))
#     detection_img_crack, skeleton_image_crack, img_withmark, crack_len, actual_distance = tradition_process(image)
#     # skeleton_image_crack,_,_=skeleton_for_crack(image)
#     cv.imshow('detection_img_crack',detection_img_crack)
#     print(crack_len)
#     # print(actual_distance)
#
#     # cv.namedWindow('skeleton_image_crack', cv.WINDOW_FREERATIO)
#     # cv.imshow('image', image)
#     cv.imshow('skeleton_image_crack', skeleton_image_crack)
#     # cv.imshow('detection_img_crack', detection_img_crack)
#
#     # cv.imshow('img_withmark', img_withmark)
#     cv.waitKey(0)