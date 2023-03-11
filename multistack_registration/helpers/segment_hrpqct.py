import itertools
import math
import numpy as np
from scipy import ndimage as ndi
from skimage._shared.utils import check_nD
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects
from helpers.resize_reposition_image import crop_pad_aims, crop_pad_image
from pint import Quantity

def threshold_local_3d(image, block_size, method='gaussian',
                      offset=0, mode='reflect', param=None, cval=0):
  if block_size % 2 == 0:
    raise ValueError("The kwarg ``block_size`` must be odd! Given "
                      "``block_size`` {0} is even.".format(block_size))
  check_nD(image, 3)
  thresh_image = np.zeros(image.shape, 'double')
    
  if method == 'generic':
      ndi.generic_filter(image, param, block_size,
                          output=thresh_image, mode=mode, cval=cval)
  elif method == 'gaussian':
      if param is None:
          # automatically determine sigma which covers > 99% of distribution
          sigma = (block_size - 1) / 6.0
      else:
          sigma = param
      ndi.gaussian_filter(image, sigma, output=thresh_image, mode=mode,
                          cval=cval)
  elif method == 'mean':
      mask = 1. / block_size * np.ones((block_size,))
      # separation of filters to speedup convolution
      ndi.convolve1d(image, mask, axis=0, output=thresh_image, mode=mode,
                      cval=cval)
      ndi.convolve1d(thresh_image, mask, axis=1, output=thresh_image,
                      mode=mode, cval=cval)
      ndi.convolve1d(thresh_image, mask, axis=2, output=thresh_image,
                      mode=mode, cval=cval)
  elif method == 'median':
      ndi.median_filter(image, block_size, output=thresh_image, mode=mode,
                        cval=cval)
  else:
      raise ValueError("Invalid method specified. Please use `generic`, "
                        "`gaussian`, `mean`, or `median`.")

  return thresh_image - offset

def adaptive_threshold(density,low_threshold=190,high_threshold=450):
    
    filtered_density = gaussian(density,sigma=1)
    
    low_mask = filtered_density>low_threshold
    local_thresh = threshold_local_3d(density*low_mask,13,method = 'mean')

    low_image = (filtered_density*low_mask)>local_thresh
    high_image = filtered_density>high_threshold

    return remove_small_objects(high_image | low_image, min_size=64)


def segment_hrpqct(density, voxelsize, masks=[], thresholds=[]):
    
    voxelsize = voxelsize.to('um').magnitude[0]
    
    if not isinstance(thresholds,list):
        thresholds = [thresholds]
    if not isinstance(masks,list):
        masks = [masks]
        
    masks = [np.asarray(m) for m in masks]
    density = np.asarray(density)     
    
    if voxelsize<80: #XtremeCT II
        filtered_density = gaussian(density,sigma=0.8,truncate=1.2)
        segmented_density = np.zeros_like(density)
        for thr, mask in zip(thresholds, masks):
            segmented_density[(filtered_density>thr) & (mask>0)] = 1
        
    else: #XtremeCT I
        trab_threshold = 190
        cort_threshold = 450
        adaptive = adaptive_threshold(density, low_threshold=trab_threshold, high_threshold=cort_threshold)
        segmented_density = np.zeros_like(density)
        for thr, mask in zip(thresholds, masks):
            segmented_density[(adaptive>0) & (mask>0)] = 1
    
    return segmented_density        

def segment_hrpqct_aim(density_aim, mask_aims = [], thresholds=[]):
    
    voxelsize=density_aim.voxelsize

    if not isinstance(thresholds,list):
        thresholds = [thresholds]
    if not isinstance(mask_aims,list):
        mask_aims = [mask_aims]
        
    for mask in mask_aims:
        mask.data = crop_pad_image(
            density_aim.data,mask.data,
            ref_img_position=density_aim.position,
            resize_img_position=mask.position)
    
    masks = [np.asarray(m.data) for m in mask_aims]
    density = np.asarray(density_aim.data)
    
    segmented = segment_hrpqct(density, voxelsize, masks=masks, thresholds=thresholds)
    
    return segmented

