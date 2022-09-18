import cv2
import numpy as np


def addbox(
    image,
    sbcord,
    bbcord=None,
    sbsize=100,
    bbsize=200,
    sbthickness=2,
    bbthickness=1,
    color=(255, 0, 0),
):
    """sb = small box,
    bb = big box,
    sbcord = upper left cordinate of small box
    bbcord = upper left cordinate of big box
    """
    w, h = image.shape[0], image.shape[1]

    # big box.
    if bbcord is None:
        bbcord = (w - bbsize - bbthickness, h - bbsize - bbthickness)
    first_point_b = bbcord
    last_point_b = (bbcord[0] + bbsize, bbcord[1] + bbsize)
    cv2.rectangle(image, first_point_b, last_point_b, color, bbthickness)

    # small box.
    first_point_s = sbcord
    last_point_s = (first_point_s[0] + sbsize, first_point_s[1] + sbsize)
    cv2.rectangle(image, first_point_s, last_point_s, color, sbthickness)

    # crop and combine.
    crop_img = image[
        first_point_s[1]: first_point_s[1] + sbsize,
        first_point_s[0]: first_point_s[0] + sbsize,
    ]
    crop_img = cv2.resize(crop_img, (bbsize, bbsize))
    image[
        first_point_b[0]: first_point_b[0] + bbsize,
        first_point_b[1]: first_point_b[1] + bbsize,
    ] = crop_img

    return image


def convert_color(arr, cmap='viridis', vmin=0, vmax=0.1):
    import matplotlib.cm as cm
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin, vmax)
    rgba = sm.to_rgba(arr, alpha=1)
    return np.array(rgba[:, :, :3])


def addbox_with_diff(input, gt, pt, vmax=0.1, size=100,
                     color=(255, 0, 0),
                     thickness=1,
                     bbthickness=2):

    H, W, C = input.shape

    out = np.zeros((H, W + H // 2, C))
    out[:, :W] = input

    # small box.
    pt1 = pt
    pt2 = (pt1[0] + size, pt1[1] + size)
    cv2.rectangle(out, pt1, pt2, color, thickness)

    # crop.
    bbsize = H // 2
    crop_img = input[pt[1]:pt[1] + size, pt[0]:pt[0] + size]
    crop_img = cv2.resize(crop_img, (bbsize, bbsize))

    # diff
    diff = convert_color(np.abs(input-gt)[:,:,1], vmin=0, vmax=vmax)

    crop_img_diff = diff[pt[1]:pt[1] + size, pt[0]:pt[0] + size]
    crop_img_diff = cv2.resize(crop_img_diff, (bbsize, bbsize))

    out[:H // 2, W:W + H // 2, :] = crop_img
    out[H // 2:, W:W + H // 2, :] = crop_img_diff

    return out
