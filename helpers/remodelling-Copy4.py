from copy import deepcopy
from itertools import combinations
from ifb_framework.pipelines.hrpqct.multistack_registration.helpers.reporting import dftotxt
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
    


    # Calculate a common region as 1 eroded overlay between image regions 
    # (this erosion is necessary for slanted surfaces)
    common_region = binary_erosion(
        np.all([np.any(im.domain, axis=0) for im in series.get(keyImage, to=regto)], axis=0))
    
    for thr, key in zip(thresholds, keyMask):
        
        # Deep copy image and create a gauss filtered copy
        series.data['_filt_'+keyImage] =deepcopy(series.data[keyImage])
        series.data['_filt_'+keyImage] = [im.filter(
            'SmoothingRecursiveGaussian') for im in series.data['_filt_'+keyImage]]          
        
        # limit the segmented image to the masked region before transformation
        series.data['_filt_'+keyImage] = [im.mask(m.data) for im, m in zip(series.data['_filt_'+keyImage], series.data[key])]
        
        # Segment image based on Gauss filtered image using masks thresholds provided
        series.data['_seg'] = [
            im > thr for im in series.data['_filt_'+keyImage]]
        

        # Transform the image according to transformation matrices
        segImages = series.get('_seg', to=regto)
    

        # Limit segmented data to the common region 
        segData = [(im.data > 0.5) * common_region  for im in segImages]
        
        # Limit the density image to the common region
        densities = [(im.data * common_region)
                     for im in series.get('_filt_'+keyImage, to=regto)]
        
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
        #dfRem = dfRem.rename_axis('>{} mg/ccm'.format(remThr))
        #dfRem = dfRem.reset_index(drop=True)
        
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

