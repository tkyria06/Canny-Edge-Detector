import numpy as np
import cv2
from matplotlib import pyplot as plt


def convolution_Multichannel_Extension(img, kernel, border_type, channels):
    # auxiliary function when we have more than one channel
    # split the image and call "convolution_2D" function for each channel
    # then take the results from "convolution_2D" and merge the channels
    if channels == 2:
        ch1, ch2 = cv2.split(img)
        channel_1 = convolution_2D(ch1, kernel, border_type)
        channel_2 = convolution_2D(ch2, kernel, border_type)
        image = cv2.merge([channel_1, channel_2])
    else:
        ch1, ch2, ch3 = cv2.split(img)
        channel_1 = convolution_2D(ch1, kernel, border_type)
        channel_2 = convolution_2D(ch2, kernel, border_type)
        channel_3 = convolution_2D(ch3, kernel, border_type)
        image = cv2.merge([channel_1, channel_2, channel_3])

    return image


def convolution_2D(arr, kernel, border_type):
    kernel_size = kernel.shape[0]
    assert (kernel_size % 2 != 0 and kernel_size > 0)

    channels = 1
    if len(arr.shape) == 2:
        channels == 1
    # If image has more than one channels -> call function: "convolution_Multichannel_Extension"
    elif len(arr.shape) == 3:
        channels = arr.shape[2]
        conv_arr = convolution_Multichannel_Extension(arr,kernel, border_type,channels)
        return conv_arr

    pad_size = int((kernel.shape[0]-1)/2)

    # Flips Sobel Sx Kernel around vertical, horizontal, or both axes
    flip_Y_axis = cv2.flip(kernel, 1)  # 1 means flipping around y-axis
    kernel_FLIP = cv2.flip(flip_Y_axis, 0)  # 0 means flipping around the x-axis

    img_replicate = cv2.copyMakeBorder(src=arr, top=pad_size, bottom=pad_size, left=pad_size, right=pad_size,
                                       borderType=border_type)

    rows, columns = arr.shape[:2]

    # Convolved image initialisation
    conv_arr = np.zeros(shape=(rows, columns))

    # Convolution
    for i in range(rows): # Rows num is from original image, not from padded image -> we don't need to subtract
        for j in range(columns): # Cols num is from original image, not from padded image -> we don't need to subtract
            g = np.sum(np.multiply(kernel_FLIP, img_replicate[i:i + kernel_size, j:j + kernel_size]))
            conv_arr[i][j] = g

    return conv_arr


def gaussian_kernel_2D(ksize, sigma):
    assert(ksize % 2 != 0 and ksize > 0)
    assert (sigma > 0)

    gauss = []
    # Calculate 1D kernel
    for x in range(-int((ksize-1)/2), int((ksize-1)/2)+1):
        gauss.append((1/(sigma*np.sqrt(2*np.pi))) * (np.exp(-0.5 * np.square(x / sigma))))
    # Calculate 2D kernel
    kernel = np.outer(gauss, gauss)
    # All of its elements sum to 1
    kernel = kernel / np.sum(kernel)

    return kernel


def normalisation_0_1(img):
    # Normalise array to be in range [0, 1]
    min_ImgNumber = img.min()
    max_ImgNumber = img.max()

    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            img[i][j] = (img[i][j]-min_ImgNumber) / (max_ImgNumber - min_ImgNumber)

    return img


def normalisation_minus1_plus1(img):
    # Normalise array to be in range [-1, +1]
    min_ImgNumber = img.min()
    max_ImgNumber = img.max()

    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            img[i][j] = 2 * ((img[i][j] - min_ImgNumber) / (max_ImgNumber - min_ImgNumber)) - 1

    return img


def sobel_x(arr):
    sobel_kernel_Sx = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])

    dx = convolution_2D(arr, sobel_kernel_Sx, cv2.BORDER_REPLICATE)

    # Normalized in the range [−1,+1]
    dx = normalisation_minus1_plus1(dx)

    return dx


def sobel_y(arr):
    sobel_kernel_Sy = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])

    dy = convolution_2D(arr, sobel_kernel_Sy, cv2.BORDER_REPLICATE)

    # Normalized in the range [−1,+1]
    dy = normalisation_minus1_plus1(dy)

    return dy


def magnitude(x, y):
    rows, columns = x.shape[:2]

    # magnitude image initialisation
    mag = np.zeros(shape=(rows, columns))

    for i in range(rows):
        for j in range(columns):
            mag[i][j] = np.sqrt(x[i][j] ** 2 + y[i][j] ** 2)

    mag = normalisation_0_1(mag)
    return mag


def direction(x, y):
    arr_dir = np.rad2deg(np.arctan2(y, x))

    return arr_dir

def non_maximum_suppression(arr_mag, arr_dir):
    arr_dir[arr_dir < 0] += 180 # deal with negative angles
    rows, columns = arr_mag.shape[:2]
    arr_local_maxima = arr_mag

    for i in range(1, rows-1):
        for j in range(1,columns-1):
            # In the x axis direction
            if (0 <= arr_dir[i, j] < 22.5):
                neighbors_max_value = max(arr_mag[i, j - 1], arr_mag[i, j + 1])
            # Top right diagonal direction
            elif (22.5 <= arr_dir[i, j] < 67.5):
                neighbors_max_value = max(arr_mag[i - 1, j - 1], arr_mag[i + 1, j + 1])
            # In y-axis direction
            elif (67.5 <= arr_dir[i, j] < 112.5):
                neighbors_max_value = max(arr_mag[i - 1, j], arr_mag[i + 1, j])
            # Top left diagonal direction
            elif (112.5 <= arr_dir[i, j] < 157.5):
                neighbors_max_value = max(arr_mag[i + 1, j - 1], arr_mag[i - 1, j + 1])
            # Restart the cycle - In the x axis direction
            else:
                neighbors_max_value = max(arr_mag[i, j - 1], arr_mag[i, j + 1])

            if arr_mag[i, j] >= neighbors_max_value:
                arr_local_maxima[i, j] = arr_mag[i, j]
            else:
                arr_local_maxima[i, j] = 0.0

    return arr_local_maxima


def hysteresis_thresholding(arr, low_ratio, high_ratio):
    # Copy arr and in the new arr remove zeros. Then calculate mean and find the thresholds
    arr_with_no_zeros = np.copy(arr)
    arr_with_no_zeros[arr_with_no_zeros == 0] = np.nan
    # Mean calculation
    mean = np.nanmean(arr_with_no_zeros)
    # Calculate mean
    min_threshold = mean + (mean * (low_ratio / 255))
    max_threshold = mean + (mean * (high_ratio / 255))

    edges = np.copy(arr)

    sum_arr = 0.0
    max_arr = 0.0

    row, cols = arr.shape[:2]

    for i in range(row):
        for j in range(cols):
            # Strong pixels
            if (arr[i, j] > max_threshold):
                edges[i, j] = 1.0
            # Weak pixels
            elif (arr[i, j] < min_threshold):
                edges[i, j] = 0.0

    while (1):
        for i in range(1, row - 1):
            for j in range(1, cols - 1):
                # Intermediate pixels
                if (arr[i, j] > min_threshold) and (arr[i, j] < max_threshold):
                    # if one of its 8 neighbours has value > max_threshold -> it's an edge
                    if (((edges[i, j - 1]) or
                         (edges[i, j + 1]) or
                         (edges[i - 1, j - 1]) or
                         (edges[i + 1, j + 1]) or
                         (edges[i - 1, j]) or
                         (edges[i + 1, j]) or
                         (edges[i + 1, j - 1]) or
                         (edges[i + 1, j + 1])
                         ) > max_threshold):
                        edges[i, j] = 1.0
                    else:
                        edges[i, j] = 0.0

        sum_arr = np.sum(edges)
        if sum_arr > max_arr:
            max_arr = sum_arr
        else:
            break
        edges = np.uint8(edges)
    return edges


def main():
    # Read image
    img = cv2.imread("Results/building.jpg")
    if img is None:
        print("Image not found.")
        return
    ksize = 3
    pad_size = int((ksize - 1) / 2)
    sigma = pad_size

    assert(ksize % 2 != 0 and ksize > 0)
    assert(sigma > 0)

    # Compute Gaussian Kernel
    gaussian_kernel = gaussian_kernel_2D(ksize, sigma)

    # Compute Blurred Image
    print("Noise Reduction ...")
    image_blur = convolution_2D(img, gaussian_kernel, cv2.BORDER_REPLICATE)

    # Grayscale
    img_gray = cv2.cvtColor(image_blur.astype(np.float32), cv2.COLOR_BGR2GRAY)

    # Plot Gaussian Blurring
    fig = plt.figure("Gaussian Blurring")

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image_blur.astype(np.float32)/255.0,cv2.COLOR_BGR2RGB))
    plt.title("Blurred Image")


    # dI/dx
    print("1st order partial derivatives along x-axis ...")
    sobel_X_image = sobel_x(img_gray)

    # dI/dy
    print("1st order partial derivatives along y-axis ...")
    sobel_Y_image = sobel_y(img_gray)

    # Plot Gaussian Blurring
    fig = plt.figure("Image 1st order partial derivatives")

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(sobel_X_image, cmap='gray')
    plt.title("dI/dx")

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(sobel_Y_image, cmap='gray')
    plt.title("dI/dy")


    # Image gradient magnitude
    print("Image gradient magnitude and direction ...")
    mag = magnitude(sobel_X_image,sobel_Y_image)
    direc = direction(sobel_X_image,sobel_Y_image)


    # Plot gradient magnitude
    fig = plt.figure("Image gradient magnitude")

    plt.subplot(1, 1, 1)
    plt.axis("off")
    plt.imshow(mag, cmap='gray')
    plt.title("Image magnitude")


    arr_local_maxima = non_maximum_suppression(mag,direc)

    # Plot Non-maximum Suppression
    print("Non-maximum Suppression ...")
    fig = plt.figure("Non-maximum Suppression")

    plt.subplot(1, 1, 1)
    plt.axis("off")
    plt.imshow(arr_local_maxima, cmap='gray')
    plt.title("Thinned Edges")


    print("Hysteresis Thresholding ...")
    edges = hysteresis_thresholding(arr_local_maxima,100,200)

    # OpenCv Canny
    edges_Canny = cv2.Canny(np.uint8(image_blur), threshold1=100, threshold2=200)


    # Plot Non-maximum Suppression
    fig = plt.figure("Edge Detection Results")

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(edges_Canny, cmap='gray')
    plt.title("OpenCV Edges")

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(edges, cmap='gray')
    plt.title("Mine Edges")

    plt.show()


if __name__ == "__main__":
    main()