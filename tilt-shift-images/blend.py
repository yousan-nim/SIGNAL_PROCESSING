#!/usr/bin/python
"""Script based on assignment 6 to blend two images."""
import cv2
import math
import os
import numpy as np
import scipy.signal
import shutil


def generating_kernel(parameter):

    kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                       0.25, 0.25 - parameter / 2.0])
    return np.outer(kernel, kernel)


def reduce_img(image):

    # per the instructions, use 0.4 for the kernel generation
    kernel = generating_kernel(0.4)

    # use convolve2d with the image and kernel sent in
    output = scipy.signal.convolve2d(image, kernel, 'same')

    # return every other line and row
    return output[:output.shape[0]:2, :output.shape[1]:2]


def expand(image):

    # per the instructions, use 0.4 for the kernel generation
    kernel = generating_kernel(0.4)

    # make a new array double the size, assign initial values
    output = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    output[:output.shape[0]:2, :output.shape[1]:2] = image

    # use convolve2d to fill in rest
    # multiply by 4 per instructions to scale back up
    output = scipy.signal.convolve2d(output, kernel, 'same') * 4
    return output


def gauss_pyramid(image, levels):

    output = [image]
    for level in range(levels):
        output.append(reduce_img(output[level]))

    return output


def lapl_pyramid(gauss_pyr):

    output = []
    # look over the lists, but ignore the last element since it cannot be
    # subtracted
    for image1, image2 in zip(gauss_pyr[:-1], gauss_pyr[1:]):
        # add in the subtracted difference
        # expand and bind the 2nd image based on the dimentions of the first
        output.append(
            image1 - expand(image2)[:image1.shape[0], :image1.shape[1]])

    # now add the last item back in
    output.append(gauss_pyr[-1])

    return output


def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):

    blended_pyr = []
    for lapl_white, lapl_black, gauss_mask in \
            zip(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
        blended_pyr.append(gauss_mask * lapl_white +
                           (1 - gauss_mask) * lapl_black)

    return blended_pyr


def collapse(pyramid):

    output = pyramid[-1]
    for image in reversed(pyramid[:-1]):
        output = image + expand(output)[:image.shape[0], :image.shape[1]]
    return output


def run_blend(black_image, white_image, mask):

    # Automatically figure out the size
    min_size = min(black_image.shape)
    # at least 16x16 at the highest level.
    depth = int(math.floor(math.log(min_size, 2))) - 4

    gauss_pyr_mask = gauss_pyramid(mask, depth)
    gauss_pyr_black = gauss_pyramid(black_image, depth)
    gauss_pyr_white = gauss_pyramid(white_image, depth)

    lapl_pyr_black = lapl_pyramid(gauss_pyr_black)
    lapl_pyr_white = lapl_pyramid(gauss_pyr_white)

    outpyr = blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
    outimg = collapse(outpyr)

    # blending sometimes results in slightly out of bound numbers.
    outimg[outimg < 0] = 0
    outimg[outimg > 255] = 255
    outimg = outimg.astype(np.uint8)

    return outimg


def get_images(sourcefolder):
    """Rewritten function to collect the three images from three folders."""
    filenames = os.listdir(sourcefolder)
    for photo in filenames:
        black_img = cv2.imread('images/original/' + photo)
        white_img = cv2.imread('images/blur/' + photo)
        mask_img = cv2.imread('images/mask/' + photo)

        if mask_img is None:
            print ( 'Oops! There is no mask of image: ', photo )
            continue
        if white_img is None:
            print ( 'Oops! There is no blurred version of image: ', photo )
            continue

        assert black_img.shape == white_img.shape, \
            "Error - the sizes of orignal and blur are not equal"

        assert black_img.shape == mask_img.shape, \
            "Error - the sizes of the original and the mask are not equal"

        print (photo)
        yield photo, white_img, black_img, mask_img


def main():
    """Given the two images, blend them according to the mask."""
    sourcefolder = 'images/original'
    outfolder = 'images/output'

    if os.path.isdir(outfolder):
        shutil.rmtree(outfolder)
    os.mkdir(outfolder)

    for photo, white_img, black_img, mask_img in get_images(sourcefolder):
        print ("...applying blending")
        black_img = black_img.astype(float)
        white_img = white_img.astype(float)
        mask_img = mask_img.astype(float) / 255

        out_layers = []
        for channel in range(3):
            outimg = run_blend(black_img[:, :, channel],
                               white_img[:, :, channel],
                               mask_img[:, :, channel])
            out_layers.append(outimg)

        outimg = cv2.merge(out_layers)
        cv2.imwrite(os.path.join(outfolder, photo), outimg)
        print ( '...[DONE]' )

if __name__ == "__main__":
    main()