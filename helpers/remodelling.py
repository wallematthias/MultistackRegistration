from copy import deepcopy
from itertools import combinations
import numpy as np
import pandas as pd
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_erosion
import SimpleITK as sitk
#import termplotlib as tpl
import numpy as np
from skimage.measure import regionprops, regionprops_table
from scipy import stats
import pandas as pd
import numpy as np
from scipy import ndimage
from helpers.reporting import dftotxt
from helpers.imdebug import imdebug
from helpers.segment_hrpqct import segment_hrpqct



def remodelling(
        series,
        keyImage='dens',
        keyMask=[
            'trab',
            'cort'],
    baseline=0,
    thresholds=[
            320,
            450],
        remThr=225,
        min_size=12,
        distance=3,
        regto=0,
        repoducability='DIST'):

    if thresholds is None:
         print('provide threshold')

    if keyMask is None:
        print('provide mask')
        
    datanames = []
    dataframes = []
    docstring = ''
    
    # Transform the image according to transformation matrices
    images = series.get(keyImage, to=regto)  
    
    # Calculate a common region as 1 eroded overlay between image regions 
    # (this erosion is necessary for slanted surfaces)
    common_region = binary_erosion(
        np.all([np.any(im.domain, axis=0) for im in series.get(keyImage, to=regto)], axis=0))
    
    for thr, key in zip(thresholds, keyMask):
        
        # Transform the image according to transformation matrices
        masks = series.get(key, to=regto)

        # Limit the segmentations to the common and mask region (maybe add gauss filter here)
        segData = [segment_hrpqct(im.data * common_region, voxelsize=im.voxelsize, masks=mask.data, thresholds=thr) for im, mask in zip(images,masks)]
        imdebug(segData)
        
        # Limit the density image to the common and mask region
        densities = [im.data * mask.data * common_region for im, mask in zip(images,masks)]
        
        # Initiate the remodellling matrices based on the number of timepoints
        dfs = []
        for baseline, followup in combinations(
                range(0, series.nTimepoints), 2):

            # Get segmented baseline data
            baseline_seg = segData[baseline] > 0.5
            
            # Get segmented followup data
            followup_seg = segData[followup] > 0.5

            # Calcualte formation and resorption
            binaryF = (followup_seg>0) & (baseline_seg==0)
            binaryR = (followup_seg==0) & (baseline_seg>0)
            grayF = (densities[followup] - densities[baseline]) > (remThr)
            grayR = (densities[followup] - densities[baseline]) < (-remThr)
            
            # Here we assemble the matrices for formation/resortpion
            dfs.append(
                {
                    't0': baseline,
                    't1': followup,
                    'for': np.sum(remove_small_objects(binaryF & grayF, min_size=min_size))/ np.sum(baseline_seg), #divide the FV/BV
                    'res': np.sum(remove_small_objects(binaryR & grayR, min_size=min_size))/ np.sum(baseline_seg), #divide the RV/BV
                    'BV': np.sum(baseline_seg) #Save BV
                }
            )
                
                 
            if 0: #will most likely be removed soon
                # This section here calcualtes advanced spatial markers
                grayIntensity = densities[followup] - densities[baseline]

                # Label formation sites
                label_im, nb_labels = ndimage.label(binaryF)

                # Regionprops table
                regions = regionprops_table(label_im,intensity_image=grayIntensity,properties=(
                    'area',"max_intensity",'min_intensity','mean_intensity', 'major_axis_length','minor_axis_length','centroid'))
                # These are not added to the docstring
                dataframes.append(pd.DataFrame(regions))
                datanames.append('TABLE_6_F_CLUSTERS_{}_{}TO{}_REGTO{}'.format(key,followup,baseline,regto))

                # Label resorption sites
                label_im, nb_labels = ndimage.label(binaryR)
                # Regionprops table
                regions = regionprops_table(label_im,intensity_image=grayIntensity,properties=(
                    'area',"max_intensity",'min_intensity','mean_intensity','major_axis_length','minor_axis_length','centroid'))
                # These are not added to the docstring
                dataframes.append(pd.DataFrame(regions))
                datanames.append('TABLE_7_R_CLUSTERS_{}_{}TO{}_REGTO{}'.format(key, followup,baseline,regto))
        
        
        #format the remodelling string for the docstring
        dfRem = pd.DataFrame(dfs)

        
        # append this table for saving
        dataframes.append(dfRem)
        datanames.append(
            'TABLE_4_GRAY_{}_{}_REGTO{}'.format(
                key,
                remThr,regto))  
        
        # Format and save remodelling rates in docstring
        dfRem = dfRem.round(4)
        docstring += '\n' + \
            dftotxt(dfRem, name='Table 4 {}: formation (FV/BV) and resorption (RV/BV) between Timepoints with clusters >{} voxel >{} mg/ccm'.format(key, min_size,remThr))

    print(docstring)
    return docstring, dataframes, datanames

