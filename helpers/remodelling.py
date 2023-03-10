from copy import deepcopy
from itertools import combinations
import numpy as np
import pandas as pd
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_erosion
from helpers.reporting import dftotxt
from helpers.segment_hrpqct import segment_hrpqct


def remodelling(
    series,
    key_image='dens',
    key_mask=['trab', 'cort'],
    baseline=0,
    thresholds=[320, 450],
    rem_thr=225,
    min_size=12,
    distance=3,
    regto=0
):
    """
    Calculates bone remodelling rates for given timepoints and masks.

    Args:
        series: An object that contains time-series image data.
        key_image (str): The key of the image to use for density calculations.
        key_mask (list[str]): The keys of the masks to use for segmentation.
        baseline (int): The index of the baseline timepoint.
        thresholds (list[float]): The threshold values to use for segmentation.
        rem_thr (float): The threshold for detecting bone remodelling.
        min_size (int): The minimum size of clusters to consider for remodelling.
        distance (int): The distance between voxels to consider for clustering.
        regto (int): The index of the timepoint to register to.

    Returns:
        tuple: A tuple containing the docstring, dataframes, and datanames.

    """
    if key_mask is None:
        print('provide mask')
        return None
    
    if thresholds is None:
        thresholds = [320, 450]

    docstring = ''
    dataframes = []
    datanames = []

    # Create a common region as the intersection of image domains
    # (this erosion is necessary for slanted surfaces)
    images = series.get(key_image, to=regto)
    common_region = np.all([np.any(im.domain, axis=0) for im in images], axis=0)
    common_region = binary_erosion(common_region)

    for thr, key in zip(thresholds, key_mask):
        masks = series.get(key, to=regto)

        # Limit the segmentations to the common and mask region
        seg_data = []
        for im, mask in zip(images, masks):
            seg = segment_hrpqct(
                im.data * common_region,
                voxelsize=im.voxelsize,
                masks=mask.data,
                thresholds=thr
            )
            seg_data.append(seg)

        # Limit the density image to the common and mask region
        densities = [im.data * mask.data * common_region for im, mask in zip(images, masks)]
        
        

        # Initiate the remodelling matrices based on the number of timepoints
        dfs = []
        remodelling_images = []
        for baseline, followup in combinations(range(series.nTimepoints), 2):
            baseline_seg = seg_data[baseline] > 0.5
            followup_seg = seg_data[followup] > 0.5
            binary_f = (followup_seg > 0) & (baseline_seg == 0)
            binary_r = (followup_seg == 0) & (baseline_seg > 0)
            gray_f = (densities[followup] - densities[baseline]) > rem_thr
            gray_r = (densities[followup] - densities[baseline]) < -rem_thr
            fv_bv = np.sum(baseline_seg)

            # Calculate formation and resorption
            formation = np.sum(remove_small_objects(binary_f & gray_f, min_size=min_size)) / fv_bv
            resorption = np.sum(remove_small_objects(binary_r & gray_r, min_size=min_size)) / fv_bv
            
            remodelling_image = deepcopy(images[baseline])
            remodelling_image.data = np.zeros_like(baseline_seg)
            remodelling_image.data[resorption>0] = 1
            remodelling_image.data[baseline_seg>0] = 2
            remodelling_image.data[formation>0] = 3
            remodelling_image.path = remodelling_image.path.replace('.AIM','REGTO_{}_REM_B{}_F{}.AIM'.format(regto, baseline,followup))
            remodelling_image.timepoint = regto
            remodelling_images.append(remodelling_image)

            # Save the results to the dataframe
            dfs.append({
                't0': baseline,
                't1': followup,
                'for': formation,
                'res': resorption,
                'BV': fv_bv
            })
                
        # Format the remodelling rates for saving
        df_rem = pd.DataFrame(dfs)
        dataframes.append(df_rem)
        datanames.append(f'TABLE_4_GRAY_{key}_{rem_thr}_REGTO{regto}')
        df_rem = df_rem.round(4)
        docstring += f'\n{dftotxt(df_rem, name=f"Table 4 {key}: formation (FV/BV) and resorption (RV/BV) between Timepoints with clusters >{min_size} voxel >{rem_thr} mg/ccm")}'
    
    print(docstring)
    return docstring, dataframes, datanames, remodelling_images