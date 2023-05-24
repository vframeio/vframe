#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################


import logging

LOG = logging.getLogger("VFRAME")
from pathlib import Path
import random
import math
from typing import Optional, Sequence, Union

import cv2 as cv
from PIL import Image, ImageDraw, ImageEnhance

import numpy as np
import imageio
from imagehash import ImageHash
import scipy.fftpack  # imagehash

try:
    # TODO: move into standalone script
    imageio.plugins.freeimage.download()
except Exception as e:
    LOG.warning("Could not download freeimage")

from vframe.utils.misc_utils import oddify, evenify
from vframe.models.geometry import BBox


# -----------------------------------------------------------------------------
#
# Hashing and similarity
#
# -----------------------------------------------------------------------------


def phash(im: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4):
    """Perceptual hash rewritten from https://github.com/JohannesBuchner/imagehash/blob/master/imagehash.py#L197"""
    wh = hash_size * highfreq_factor
    im = cv.resize(im, (wh, wh), interpolation=cv.INTER_NEAREST)
    if len(im.shape) > 2 and im.shape[2] > 1:
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(im, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return ImageHash(diff)


# -----------------------------------------------------------------------------
#
# Affine functions
#
# -----------------------------------------------------------------------------


def rotate_bound(im, angle):
    """Function from PyImageSearch
    from https://github.com/PyImageSearch/imutils/blob/c12f15391fcc945d0d644b85194b8c044a392e0a/imutils/convenience.py#L41
    """
    # grab dimensions of the image and then determine center
    (h, w) = im.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(im, M, (nW, nH))


# -----------------------------------------------------------------------------
#
# Enhance, degrade, filter
#
# -----------------------------------------------------------------------------


def blend(im_bottom: np.ndarray, im_top: np.ndarray, alpha: float) -> np.ndarray:
    """Blend the top image over the bottom image
    :param im_bottom: numpy.ndarray original image
    :param im_top: numpy.ndarray new image
    :param alpha: (float) 0.0 - 1.0 of the new image
    :returns numpy.ndarray blended composite image
    """
    return cv.addWeighted(im_bottom, 1.0 - alpha, im_top, alpha, 1.0)


def equalize(
    im: np.ndarray, fac: float, clip_limit: float = 2.0, grid_size: tuple = (8, 8)
) -> np.ndarray:
    """Equalize histograms using CLAHE.
    Applying in RGB space yields cleaner white balance and less contrast
    Applying in YUV/LAB space yields higher/improved contrast
    :param im: numpy.ndarray BGR image
    :param alpha_range: alpha range for blending
    :returns numpy.ndarray BGR image
    """
    # BGR to YUV to BGR
    im_dst = cv.cvtColor(im, cv.COLOR_BGR2YUV)
    # with CLAHE
    # clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    clahe = cv.createCLAHE(clip_limit, grid_size)
    im_dst[:, :, 0] = clahe.apply(im_dst[:, :, 0])
    im_dst = cv.cvtColor(im_dst, cv.COLOR_YUV2BGR)
    # with histograms
    # im_dst[:, :, 0] = cv.equalizeHist(im_yuv[:, :, 0])  # equalize Y channel histogram
    # im_dst = cv.cvtColor(im, cv.COLOR_BGR2LAB)
    # im_dst[:, :, 0] = clahe.apply(im_dst[:, :, 0])
    # im_dst[:, :, 1] = clahe.apply(im_dst[:, :, 1])
    # im_dst[:, :, 2] = clahe.apply(im_dst[:, :, 2])
    # im_dst = cv.cvtColor(im_dst, cv.COLOR_LAB2BGR)
    im_dst = blend(im, im_dst, fac)
    return im_dst


def compress(im: np.ndarray, fac, compression_type="JPEG"):
    """Degrade image using JPEG or WEBP compression
    :param im: (numpy.ndarray) BGR image
    :param compression_type: (str) image extension (jpg, wep)
    :param fac: image compression where 1.0 maps to quality=0 and 0.0 maps to quality=100
    """
    q_flag = (
        cv.IMWRITE_WEBP_QUALITY
        if compression_type == "WEBP"
        else cv.IMWRITE_JPEG_QUALITY
    )
    im_type = "jpg" if compression_type == "JPEG" else "webp"
    quality = int(np.interp(fac, [0.0, 1.0], (0, 100)))
    _, im_enc = cv.imencode(f".{im_type}", im, (int(q_flag), quality))
    im_dst = cv.imdecode(im_enc, cv.IMREAD_UNCHANGED)
    return im_dst


def compress_jpg(im: np.ndarray, fac: float) -> np.ndarray:
    """Degrade image using JPEG or WEBP compression
    :param im: (numpy.ndarray) BGR image
    :param fac: image compression where 1.0 maps to quality=0 and 0.0 maps to quality=100
    """
    return compress(im, fac, "JPEG")


def compress_webp(im: np.ndarray, fac: float) -> np.ndarray:
    """Degrade image using WEBP compression
    :param im: (numpy.ndarray) BGR image
    :param fac: image compression where 1.0 maps to quality=0 and 0.0 maps to quality=100
    """
    return compress(im, fac, "WEBP")


def blur_motion_v(im: np.ndarray, fac: float) -> np.ndarray:
    """Degrade image using vertical motion blur"""
    w, h = im.shape[:2][::-1]
    k = max(1, int((fac * 0.01125) * max(w, h)))  # 0.01, 0.016
    # k = max(1, int((fac * 0.05) * max(w,h)))  # 0.01, 0.016
    k = k + 1 if k % 2 else k
    kernel_v = np.zeros((k, k))
    kernel_v[:, int((k - 1) / 2)] = np.ones(k)  # Fill middle row with ones
    kernel_v /= k  # Normalize
    im_dst = cv.filter2D(im, -1, kernel_v)
    return im_dst


def blur_motion_h(im: np.ndarray, fac: float) -> np.ndarray:
    """Degrade image using horizontal motion blur"""
    w, h = im.shape[:2][::-1]
    k = max(1, int((fac * 0.01125) * max(w, h)))  # 0.01, 0.016
    # k = max(1, int((fac * 0.05) * max(w,h)))  # 0.01, 0.016
    k = k + 1 if k % 2 else k
    kernel_h = np.zeros((k, k))
    kernel_h[int((k - 1) / 2), :] = np.ones(k)  # Fill middle row with ones
    kernel_h /= k  # Normalize
    im_dst = cv.filter2D(im, -1, kernel_h)
    return im_dst


def blur_bilateral(im: np.ndarray, fac: float) -> np.ndarray:
    """Degrade image using bilateral blurring. This reduces texture and noise."""
    fac = np.interp(fac, [0.0, 1.0], [0.0, 0.1])
    dim_max = max(im.shape[:2])
    k = max(1, int(fac * dim_max))
    k = k if k % 2 else k + 1
    radius = k // 5
    # blur = cv2.bilateralFilter(img,9,75,75)
    im_dst = cv.bilateralFilter(im, radius, k, k)
    return im_dst


def blur_gaussian(im: np.ndarray, fac: float) -> np.ndarray:
    """Degrade image using Gaussian blur. This reduces texture and noise."""
    fac = np.interp(fac, [0.0, 1.0], [0.0, 0.1])
    dim_max = max(im.shape[:2])
    k = max(1, int(fac * dim_max))
    k = k if k % 2 else k + 1
    # dst = cv.blur(src, (i, i))
    im_dst = cv.GaussianBlur(im, (k, k), 0)
    return im_dst


def rescale(
    im: np.ndarray,
    fac: float,
    interp_down: int = cv.INTER_CUBIC,
    interp_up: int = cv.INTER_CUBIC,
) -> np.ndarray:
    """Degrades image by reducing scale then rescaling to original size"""
    w, h = im.shape[:2][::-1]
    nw, nh = (max(1, int(fac * w)), max(1, int(fac * h)))
    im_dst = resize(im, width=nw, height=nh, interp=interp_down)
    im_dst = resize(im_dst, width=w, height=h, force_fit=True, interp=interp_up)
    return im_dst


def _enhance(
    im: np.ndarray, enhancement: ImageEnhance._Enhance, fac: float
) -> np.ndarray:
    """Transform image using Pillow enhancements
    :param im: numpy.ndarray
    :param enhancement:Enhance
    :param amt: float
    :returns numpy.ndarray
    """
    return ensure_np(enhancement(ensure_pil(im)).enhance(fac))


def sharpness(im: np.ndarray, fac: float) -> np.ndarray:
    """Adjust sharpness
    :param im: numpy.ndarray
    :param fac: normalized float
    :returns numpy.ndarray
    """
    amt = np.interp(fac, [0.0, 1.0], (0, 20))
    return _enhance(im, ImageEnhance.Sharpness, amt)


def brightness(im: np.ndarray, fac: float) -> np.ndarray:
    """Increase brightness
    :param im: numpy.ndarray
    :param fac: normalized float
    :returns numpy.ndarray
    """
    amt = np.interp(fac, [0.0, 1.0], (1.0, 2.0))
    return _enhance(im, ImageEnhance.Brightness, amt)


def darkness(im: np.ndarray, fac: float) -> np.ndarray:
    """Darken image
    :param im: numpy.ndarray
    :param fac: normalized float
    :returns numpy.ndarray
    """
    amt = np.interp(fac, [0.0, 1.0], (0.0, -2.0))
    return _enhance(im, ImageEnhance.Brightness, amt)


def contrast(im: np.ndarray, fac: float) -> np.ndarray:
    """Increase contrast
    :param im: numpy.ndarray
    :param fac: normalized float
    :returns numpy.ndarray
    """
    amt = np.interp(fac, [0.0, 1.0], (0, 3))
    return _enhance(im, ImageEnhance.Contrast, amt)


def shift(im: np.ndarray, fac: float) -> np.ndarray:
    """Degrades image by superimposing image with offset xy and applies random blend"""
    w, h = im.shape[:2][::-1]
    max_px = int(max(im.shape[:2]) * 0.02)
    D = max(1, int(np.interp(fac, [0.0, 1.0], (0, max_px))))
    rad = random.uniform(0, 2 * math.pi)
    dx = int(math.cos(rad) / D)
    dy = int(math.sin(rad) / D)
    # pad
    im_dst = cv.copyMakeBorder(im, D, D, D, D, cv.BORDER_CONSTANT, value=[0, 0, 0])
    # paste
    x1, y1, x2, y2 = list(np.array([0, 0, w, h]) + np.array([dx, dy, -dx, -dy]))
    # crop
    xyxy = list(np.array([0, 0, w, h]) + np.array([D, D, -D, -D]))
    bbox = BBox(*xyxy, w + D, h + D)
    im_dst = crop_roi(im_dst, bbox)
    # scale
    im_dst = resize(im_dst, width=w, height=h, force_fit=True)
    # blend
    alpha = random.uniform(0.1, 0.35)
    return blend(im, im_dst, alpha)


def chromatic_aberration(
    im: np.ndarray, fac: float, channel: int = 0, max_distance: int = 5
) -> np.ndarray:
    """Scale-shift color channel and then superimposes it back into image
    :param channel: int for BGR channel 0 = B
    """
    # TODO: use shift method to overlay channels
    channel = channel if channel else random.randint(0, 2)
    w, h = im.shape[:2][::-1]
    im_c = im.copy()
    # dx,dy = value_range
    dx = np.interp(fac, [0.0, 1.0], (0, max_distance))
    dy = np.interp(fac, [0.0, 1.0], (0, max_distance))
    # inner crop
    xyxy = list(np.array([0, 0, w, h]) + np.array([dx, dy, -dx, -dy]))
    bbox = BBox(*xyxy, w, h)
    im_c = crop_roi(im_c, bbox)
    # resize back to original dims
    im_c = resize(im_c, width=w, height=h, force_fit=True)
    # add color channel
    im_dst = im.copy()
    im_dst[:, :, channel] = im_c[:, :, channel]
    return im_dst


def grayscale(im: np.ndarray, fac: float) -> np.ndarray:
    return blend(im, gray2bgr(bgr2gray(im)), fac)


# -----------------------------------------------------------------------------
#
# Image data type conversions
#
# -----------------------------------------------------------------------------


def np2pil(im: Union[np.ndarray, Image.Image], swap: bool = True) -> Image:
    """Ensure image is Pillow format
    :param im: image in numpy or format
    :returns image in Pillow RGB format
    """
    try:
        im.verify()
        LOG.warn("Expected Numpy received PIL")
        return im
    except:
        if swap:
            if len(im.shape) == 2:
                color_mode = "L"
            elif im.shape[2] == 4:
                im = bgra2rgba(im)
                color_mode = "RGBA"
            elif im.shape[2] == 3:
                im = bgr2rgb(im)
                color_mode = "RGB"
        else:
            color_mode = "RGB"
        return Image.fromarray(im.astype("uint8"), color_mode)


def pil2np(im: Image, swap=True):
    """Ensure image is Numpy.ndarry format
    :param im: image in numpy or format
    :returns image in Numpy uint8 format
    """
    if type(im) == np.ndarray:
        LOG.warn("Expected PIL received Numpy")
        return im
    im = np.asarray(im, np.uint8)
    if swap:
        if len(im.shape) == 2:
            # grayscale, ignore swap and return current image
            return im
        elif len(im.shape) > 2 and im.shape[2] == 4:
            im = bgra2rgba(im)
        elif im.shape[2] == 3:
            im = bgr2rgb(im)
    return im


def is_pil(im: Union[Image.Image, np.ndarray]) -> bool:
    """Ensures image is Pillow format
    :param im: image
    :returns bool if is
    """
    try:
        im.verify()
        return True
    except:
        return False


def is_np(im: Union[Image.Image, np.ndarray]) -> bool:
    """Checks if image if numpy"""
    return type(im) == np.ndarray


def ensure_np(im: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Lazily force image type to numpy.ndarray"""
    return pil2np(im) if is_pil(im) else im


def ensure_pil(im: Union[Image.Image, np.ndarray]) -> Image:
    """Lazily force image type to"""
    return np2pil(im) if is_np(im) else im


def num_channels(im: Union[Image.Image, np.ndarray]) -> int:
    """Number of channels in numpy.ndarray image"""
    return im.shape[2] if len(im.shape) > 2 else 1


def is_grayscale(im: np.ndarray, threshold: int = 5) -> bool:
    """Returns True if image is grayscale
    :param im: (numpy.array) image
    :return (bool) of if image is grayscale"""
    b = im[:, :, 0]
    g = im[:, :, 1]
    mean = np.mean(np.abs(g - b))
    return mean < threshold


def crop_roi(im: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crops ROI
    :param im: (np.ndarray) image BGR
    :param bbox: (BBox)
    :returns (np.ndarray) BGR image ROi
    """
    dim = im.shape[:2][::-1]
    x1, y1, x2, y2 = bbox.xyxy_int
    im_roi = im[y1:y2, x1:x2]
    return im_roi


def blur_bboxes(im: np.ndarray, bboxes, fac=0.33, iters=1):
    """Blur ROI
    :param im: (np.ndarray) image BGR
    :param bbox: (BBox)
    :param cell_size: (int, int) pixellated cell size
    :returns (np.ndarray) BGR image
    """
    if not bboxes:
        return im
    elif not type(bboxes) == list:
        bboxes = list(bboxes)

    for bbox in bboxes:
        dim = im.shape[:2][::-1]
        x1, y1, x2, y2 = bbox.xyxy_int
        im_roi = im[y1:y2, x1:x2]
        h, w, c = im_roi.shape
        ksize = int(max(fac * w, fac * h))
        ksize = ksize if ksize % 2 else ksize + 1
        for n in range(iters):
            im_roi = cv.blur(im_roi, ksize=(ksize, ksize))
            # im_roi = cv.GaussianBlur(im_roi, (ksize,ksize), 0)
        im[y1:y2, x1:x2] = im_roi
    return im


def pixellate_bboxes(im: np.ndarray, bboxes, cell_size=(5, 6), expand_per=0.0):
    """Pixellates ROI using Nearest Neighbor inerpolation
    :param im: (numpy.ndarray) image BGR
    :param bbox: (BBox)
    :param cell_size: (int, int) pixellated cell size
    :returns (numpy.ndarray) BGR image
    """
    if not bboxes:
        return im
    elif not type(bboxes) == list:
        bboxes = list(bboxes)

    for bbox in bboxes:
        if expand_per > 0:
            bbox = bbox.expand_per(expand_per)
        x1, y1, x2, y2 = bbox.xyxy_int
        im_roi = im[y1:y2, x1:x2]
        h, w, c = im_roi.shape
        # pixellate
        im_roi = cv.resize(im_roi, cell_size, interpolation=cv.INTER_NEAREST)
        im_roi = cv.resize(im_roi, (w, h), interpolation=cv.INTER_NEAREST)
        im[y1:y2, x1:x2] = im_roi

    return im


def mk_mask(bboxes, shape="ellipse", blur_kernel_size=None, blur_iters=1):
    bboxes = bboxes if isinstance(bboxes, list) else [bboxes]
    # mk empty mask
    im_mask = create_blank_im(*bboxes[0].dim, 1)
    # draw mask shapes
    color = (255, 255, 255)
    for bbox in bboxes:
        if shape == "rectangle":
            im_mask = cv.rectangle(im_mask, bbox.p1.xy_int, bbox.p2.xy_int, color, -1)
        elif shape == "circle":
            im_mask = cv.circle(im_mask, bbox.cxcy_int, bbox.w, color, -1)
        elif shape == "ellipse":
            im_mask = cv.ellipse(
                im_mask, bbox.cxcy_int, bbox.wh_int, 0, 0, 360, color, -1
            )
    # blur if k
    k = blur_kernel_size
    if k:
        k = oddify(k)
        for i in range(blur_iters):
            im_mask = cv.GaussianBlur(im_mask, (k, k), k, k)
    return im_mask


def mask_composite(im: np.ndarray, im_masked, im_mask):
    """Masks two images together using grayscale mask
    :param im: the base image
    :param im_masked: the image that will be masked on top of the base image
    :param im_mask: the grayscale image used to mask
    :returns (numpy.ndarray): masked composite image
    """
    im_mask_alpha = im_mask / 255.0
    im = im_mask_alpha[:, :, None] * im_masked + (1 - im_mask_alpha)[:, :, None] * im
    return (im).astype(np.uint8)


def blur_bbox_soft(
    im: np.ndarray,
    bbox,
    iters=1,
    expand_per=-0.1,
    multiscale=True,
    mask_k_fac=0.125,
    im_k_fac=0.33,
    shape="ellipse",
):
    """Blurs objects using multiple blur scale per bbox"""
    if not bbox:
        return im
    bboxes = bbox if isinstance(bbox, list) else [bbox]
    bboxes_mask = [b.expand_per(expand_per, keep_edges=True) for b in bboxes]
    if multiscale:
        # use separate kernel size for each bboxes (slower but more accurate)
        im_blur = im.copy()
        dim = im.shape[:2][::-1]
        im_mask = create_blank_im(*dim, 1)
        for bbox in bboxes_mask:
            # create a temp mask, draw shape, blur, and add to cummulative mask
            k = min(bbox.w_int, bbox.h_int)
            k_mask = oddify(
                int(k * mask_k_fac)
            )  # scale min bbox dim for desired blur intensity
            im_mask_next = mk_mask(
                bbox, shape=shape, blur_kernel_size=k_mask, blur_iters=iters
            )
            bounding_rect = cv.boundingRect(im_mask_next)
            bbox_blur = BBox.from_xywh(*bounding_rect, *bbox.dim)
            im_mask = cv.add(im_mask, im_mask_next)
            # blur the masked area bbox in the original image

        k = max([min(b.w_int, b.h_int) for b in bboxes])
        k_im = oddify(int(k * im_k_fac))  # scaled image blur kernel
        im_blur = cv.GaussianBlur(im, (k_im, k_im), k_im / 4, 0)
    else:
        # use one kernel size for all bboxes (faster but less accurate)
        k = max([min(b.w_int, b.h_int) for b in bboxes])
        k_im = oddify(int(k * im_k_fac))  # scaled image blur kernel
        k_mask = oddify(
            int(k * mask_k_fac)
        )  # scale min bbox dim for desired blur intensity
        im_mask = mk_mask(
            bboxes, shape=shape, blur_kernel_size=k_mask, blur_iters=iters
        )
        im_blur = cv.GaussianBlur(im, (k_im, k_im), k_im, 0, k_im)

    # iteratively blend image
    k = max([min(b.w_int, b.h_int) for b in bboxes])
    k_im = oddify(int(k * im_k_fac))  # scaled image blur kernel
    im_dst = im.copy()
    for i in range(iters):
        im_alpha = im_mask / 255.0
        im_dst = im_alpha[:, :, None] * im_blur + (1 - im_alpha)[:, :, None] * im_dst
        im_mask = cv.GaussianBlur(im_mask, (k_im, k_im), k_im, 0)

    # im_dst = mask_composite(im: np.ndarray, im_blur, im_mask)

    return (im_dst).astype(np.uint8)


# -----------------------------------------------------------------------------
#
# Image writers
#
# --------------------------------------------------------------------------


def write_animated_gif(
    fp,
    frames,
    format="GIF",
    save_all=True,
    optimize=True,
    duration=40,
    loop=0,
    verbose=False,
):
    """Writes animated GIF using PIL"""
    frames[0].save(
        fp,
        append_images=frames[1:],
        format=format,
        save_all=save_all,
        optimize=optimize,
        duration=duration,
        loop=loop,
    )
    if verbose:
        LOG.info(f"Wrote: {fp}")
        s = Path(fp).stat().st_size // 1000
        LOG.info(f"{Path(fp).name}: {s:,}KB, {len(frames)} frames")


# -----------------------------------------------------------------------------
#
# Placeholder images
#
# -----------------------------------------------------------------------------


def create_blank_im(w: int, h: int, c: int = 3, dtype=np.uint8) -> np.ndarray:
    """Creates blank np image
    :param w: width
    :param h: height
    :param c: channels
    :param dtype: data type
    :returns (np.ndarray)
    """
    dim = [h, w] if c == 1 else [h, w, c]
    return np.zeros(dim, dtype=np.uint8)


def create_random_im(
    w: int, h: int, c: int = 3, low: int = 0, high: int = 255, dtype=np.uint8
) -> np.ndarray:
    """Creates blank np image
    :param w: width
    :param h: height
    :param c: channels
    :param dtype: data type
    :returns (np.ndarray)
    """
    if c == 1:
        im = np.random.randint(low, high, (h * w)).reshape((h, w)).astype(dtype)
    elif c == 3:
        im = np.random.randint(low, high, (h * w * c)).reshape((h, w, c)).astype(dtype)
    else:
        im = None  # TODO handle error
    return im


# -----------------------------------------------------------------------------
#
# Resizing
#
# -----------------------------------------------------------------------------


def resize(
    im: np.ndarray, width=None, height=None, force_fit=False, interp=cv.INTER_LINEAR
):
    """FIXME: Resizes image, scaling issues with floating point numbers
    :param im: (nump.ndarray) image
    :param width: int
    :param height: int
    :param interp: INTER_AREA, INTER_BITS, INTER_BITS2, INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR,
      INTER_NEAREST_EXACT, INTER_MAX, INTER_NEAREST, INTER_NEAREST_EXACT
    :returns (nump.ndarray) image
    """
    if not width and not height:
        return im
    else:
        im_width, im_height = im.shape[:2][::-1]
        if width and height:
            if force_fit:
                # scale_x = width / im_width
                # scale_y = height / im_height
                w, h = width, height
            else:
                scale_x = min(width / im_width, height / im_height)
                scale_y = scale_x
                w, h = int(scale_x * im_width), int(scale_y * im_height)
        elif width and not height:
            scale_x = width / im_width
            scale_y = scale_x
            w, h = int(scale_x * im_width), int(scale_y * im_height)
        elif height and not width:
            scale_y = height / im_height
            scale_x = scale_y
            w, h = int(scale_x * im_width), int(scale_y * im_height)
        w, h = int(w), int(h)
        return cv.resize(im, (w, h), interpolation=interp)


# -----------------------------------------------------------------------------
#
# OpenCV aliases
#
# -----------------------------------------------------------------------------


def bgr2gray(im: np.ndarray) -> np.ndarray:
    """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (BGR)
    :returns Numpy.ndarray (Gray)
    """
    return cv.cvtColor(im, cv.COLOR_BGR2GRAY)


def gray2bgr(im: np.ndarray) -> np.ndarray:
    """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (Gray)
    :returns Numpy.ndarray (BGR)
    """
    return cv.cvtColor(im, cv.COLOR_GRAY2BGR)


def bgr2rgb(im: np.ndarray) -> np.ndarray:
    """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (BGR)
    :returns Numpy.ndarray (RGB)
    """
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)


def rgb2bgr(im: np.ndarray) -> np.ndarray:
    """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (RGB)
    :returns Numpy.ndarray (RGB)
    """
    return cv.cvtColor(im, cv.COLOR_RGB2BGR)


def bgra2rgba(im: np.ndarray) -> np.ndarray:
    """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (BGRA)
    :returns Numpy.ndarray (RGBA)
    """
    return cv.cvtColor(im, cv.COLOR_BGRA2RGBA)


def rgba2bgra(im: np.ndarray) -> np.ndarray:
    """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (RGB)
    :returns Numpy.ndarray (RGB)
    """
    return cv.cvtColor(im, cv.COLOR_RGBA2BGRA)


def bgr2luma(im: np.ndarray) -> np.ndarray:
    """Converts BGR image to grayscale Luma
    :param im: np.ndarray BGR uint8
    :returns nd.array GRAY uint8
    """
    im_y, _, _ = cv.split(cv.cvtColor(im, cv.COLOR_BGR2YUV))
    return im_y


# -----------------------------------------------------------------------------
#
# Visualize
#
# -----------------------------------------------------------------------------


def montage(im_arr, n_cols=3):
    # temporary function
    n_index, height, width, intensity = im_arr.shape
    n_rows = n_index // n_cols
    assert n_index == n_rows * n_cols
    result = (
        im_arr.reshape(n_rows, n_cols, height, width, intensity)
        .swapaxes(1, 2)
        .reshape(height * n_rows, width * n_cols, intensity)
    )
    return result


# -----------------------------------------------------------------------------
#
# Image file loading
#
# -----------------------------------------------------------------------------


def load_hdr(fp):
    """Loads HDR image in .hdr or .exr format
    :param fp: (str) filepath
    :returns numpy.ndarray in BGR format
    """
    if Path(fp).suffix.lower() == ".exr":
        im = cv.imread(fp, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    else:
        im = imageio.imread(fp, format="HDR-FI")  # RGB
        im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
    im = np.power(im, 1 / 2.2)  # gamma correction
    im = np.clip(im, 0, 1)
    return (im * 255).astype(np.uint8)


def load_heif(fp: Union[str, Path]) -> Image.Image:
    """Loads HEIF (High Efficient Image Format) image into
    :param fp: (str) filepath
    :returns
    """
    try:
        import pyheif

        d = pyheif.read(fp)
        return Image.frombytes(d.mode, d.size, d.data, "raw", d.mode, d.stride)
    except Exception as e:
        return None


# -----------------------------------------------------------------------------
#
# Enumerated transform types
#
# -----------------------------------------------------------------------------

# pixel-level transforms
IMAGE_TRANSFORMS = {
    "compress-jpg": compress_jpg,
    "compress-webp": compress_webp,
    "equalize": equalize,
    "blur-v": blur_motion_v,
    "blur-h": blur_motion_h,
    "blur-bilateral": blur_bilateral,
    "blur": blur_gaussian,
    "rescale": rescale,
    "brighten": brightness,
    "darken": darkness,
    "sharpness": sharpness,
    "contrast": contrast,
    "shift": shift,
    "chromatic-aberration": chromatic_aberration,
    "grayscale": grayscale,
}
