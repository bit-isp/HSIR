import cv2
import numpy as np

mapbbpox = {
    'Bottom Right': 'br',
    'Bottom Left': 'bl',
    'Top Right': 'ur',
    'Top Left': 'ul',
}

def get_border_box_pt(h, w, size, thickness, pos):
    if pos == 'br':
        return (w-size-thickness, h-size-thickness)
    if pos == 'ur':
        return (w-size-thickness, 0)
    if pos == 'ul':
        return (0, 0)
    if pos == 'bl':
        return (0, h-size-thickness)

def addbox(img, pt, size=100,
           bbsize=200, bbpos='br',
           color=(255, 0, 0),
           thickness=1,
           bbthickness=2):
    H, W = img.shape[0], img.shape[1]

    ptb = get_border_box_pt(H, W, bbsize, bbthickness, pos=bbpos)
    crop_img = img[pt[1]:pt[1]+size, pt[0]:pt[0]+size]
    crop_img = cv2.resize(crop_img, (bbsize, bbsize))
    img[ptb[1]:ptb[1]+bbsize, ptb[0]:ptb[0]+bbsize] = crop_img

    # big box
    pt1 = ptb
    pt2 = (pt1[0]+bbsize, pt1[1]+bbsize)
    cv2.rectangle(img, pt1, pt2, color, bbthickness)

    # small box
    pt1 = pt
    pt2 = (pt1[0]+size, pt1[1]+size)
    cv2.rectangle(img, pt1, pt2, color, thickness)

    return img


def convert_color(arr, cmap='viridis', vmin=0, vmax=0.1):
    import matplotlib.cm as cm
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin, vmax)
    rgba = sm.to_rgba(arr, alpha=1)
    return np.array(rgba[:, :, :3])


def addbox_with_diff(input, gt, pt, vmax=0.1, size=100,
                     color=(255, 0, 0),
                     thickness=1,
                     bbthickness=2,
                     sep=5):

    H, W, C = input.shape

    out = np.zeros((H, W + H // 2 + sep, C))
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

    out[:H // 2, W+sep:W+sep + H // 2, :] = crop_img
    out[H // 2:, W+sep:W+sep + H // 2, :] = crop_img_diff

    return out
