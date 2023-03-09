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
        thresholds = [320, ]

    if keyMask is None:
        masks = [1, ]
        keyMask = ['']
    else:
        masks = [series.get(key, to=regto) for key in keyMask]

    datanames = []
    dataframes = []
    docstring = ''
    
      # Deep copy image and create a gauss filtered copy
    series.data['_filt_'+keyImage] =deepcopy(series.data[keyImage])
    series.data['_filt_'+keyImage] = [im.filter(
        'SmoothingRecursiveGaussian') for im in series.data['_filt_'+keyImage]]

    # Segment image based on Gauss filtered image using masks thresholds provided
    series.data['_seg'] = [
        deepcopy(im) > thr for im in series.data['_filt_'+keyImage]]

    # limit the segmented image to the masked region
    series.data['_seg'] = [im.mask(mask) for im in series.data['_seg']]

    # Transform the image according to transformation matrices
    segImages = series.get('_seg', to=regto)
    
    

    for mask, thr, key in zip(masks, thresholds, keyMask):
        
  
        
        # Calculate a common region as 1 eroded overlay between image regions 
        # (this erosion is necessary for slanted surfaces)
        common_region = binary_erosion(
            np.all([np.any(im.domain, axis=0) for im in segImages], axis=0))

        # Limit segmented data to the common region 
        segData = [(im.data > 0.5) * common_region  for im in segImages]
        
        # Limit the density image to the common region
        densities = [(im.data * common_region)
                     for im in series.get('_filt_'+keyImage, to=regto)]
        
        # Initiate the remodellling matrices based on the number of timepoints
        matrTrueRem = [[0] * (series.nTimepoints)
                       for i in range(series.nTimepoints)]
        matrTrueFR = [[0] * (series.nTimepoints)
                      for i in range(series.nTimepoints)]
        matrBinRem = [[0] * (series.nTimepoints)
                      for i in range(series.nTimepoints)]
        matrBinFR = [[0] * (series.nTimepoints)
                     for i in range(series.nTimepoints)]
        matrMinRem = [[0] * (series.nTimepoints)
                      for i in range(series.nTimepoints)]
        matrMinFR = [[0] * (series.nTimepoints)
                     for i in range(series.nTimepoints)]

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
            
            
            # This here calcualtes advanced spatial markers
            grayIntensity = densities[followup] - densities[baseline]
            
            label_im, nb_labels = ndimage.label(binaryF)
            regions = regionprops_table(label_im,intensity_image=grayIntensity,properties=(
                'area',"max_intensity",'min_intensity','mean_intensity', 'major_axis_length','minor_axis_length','centroid'))
            dataframes.append(pd.DataFrame(regions))
            datanames.append('TABLE_6_F_CLUSTERS_{}_{}TO{}_REGTO{}'.format(key,followup,baseline,regto))

            label_im, nb_labels = ndimage.label(binaryR)
            regions = regionprops_table(label_im,intensity_image=grayIntensity,properties=(
                'area',"max_intensity",'min_intensity','mean_intensity','major_axis_length','minor_axis_length','centroid'))
            dataframes.append(pd.DataFrame(regions))
            datanames.append('TABLE_7_R_CLUSTERS_{}_{}TO{}_REGTO{}'.format(key, followup,baseline,regto))
            

            # Here we assemble the matrices for formation/resortpion
            matrTrueRem[baseline][followup] = remove_small_objects(
                binaryF & grayF, min_size=min_size)
            matrTrueRem[followup][baseline] = remove_small_objects(
                binaryR & grayR, min_size=min_size)
            
            matrTrueFR[baseline][followup] = np.sum(
                matrTrueRem[baseline][followup])
            matrTrueFR[followup][baseline] = - \
                np.sum(matrTrueRem[followup][baseline])

            matrBinRem[baseline][followup] = remove_small_objects(
                binaryF, min_size=min_size)
            matrBinRem[followup][baseline] = remove_small_objects(
                binaryR, min_size=min_size)
            matrBinFR[baseline][followup] = np.sum(
                matrBinRem[baseline][followup])
            matrBinFR[followup][baseline] = - \
                np.sum(matrBinRem[followup][baseline])

            matrMinRem[baseline][followup] = remove_small_objects(
                grayF & (binaryF == 0), min_size=min_size)
            matrMinRem[followup][baseline] = remove_small_objects(
                grayR & (binaryR == 0), min_size=min_size)
            matrMinFR[baseline][followup] = np.sum(
                matrMinRem[baseline][followup])
            matrMinFR[followup][baseline] = - \
                np.sum(matrMinRem[followup][baseline])

        boneVol = []
        totalVol = []
        for i in range(0, series.nTimepoints):
            boneVol.append(np.sum(segData[i] > 0.5))

        # We divide the matrices by the original bone volume and round 
        dfRem = pd.DataFrame(matrTrueFR).div(boneVol)
        dfRem = dfRem.round(4)
        dfRem['BV'] = boneVol
        dfRem = dfRem.rename_axis('>{} mg/ccm'.format(remThr))
        dfRem = dfRem.reset_index()
        docstring += '\n' + \
            dftotxt(dfRem, name='Table 4 {}: formation (FV/BV) and resorption (RV/BV) between Timepoints with clusters >{} voxel'.format(key, min_size))

        dfBin = pd.DataFrame(matrBinFR).div(boneVol)
        dfBin = dfBin.round(4)
        dfBin['BV'] = boneVol
        dfBin = dfBin.rename_axis('Binary')
        dfBin = dfBin.reset_index()
        docstring += '\n' + dftotxt(dfBin)

        dfMin = pd.DataFrame(matrMinFR).div(boneVol)
        dfMin = dfMin.round(4)
        dfMin['BV'] = boneVol
        dfMin = dfMin.rename_axis('Mineral')
        dfMin = dfMin.reset_index()
        docstring += '\n' + dftotxt(dfMin)

        dataframes += [dfRem, dfMin, dfBin]
        datanames += [
            'TABLE_4_GRAY_{}_{}_REGTO{}'.format(
                key,
                remThr,regto),
            'TABLE_4_MNRL_{}_REGTO{}'.format(key,regto),
            'TABLE_4_BINARY_{}_REGTO{}'.format(key,regto)]

    '''
    # THis option will be depriciated (way too complicated) 
    
    def iou(arr1, arr2):
    intersect = np.sum(arr1 & arr2)
    union = np.sum(arr1 | arr2)
    if union == 0:
        return 0
    else:
        return intersect / union

    def hdistance(arr1, arr2):
        filter = sitk.HausdorffDistanceImageFilter()
        filter.Execute(
            sitk.GetImageFromArray(
                arr1.astype(int)), sitk.GetImageFromArray(
                arr2.astype(int)))
        return filter.GetAverageHausdorffDistance()

    
        if repoducability is not None:
            titles = [
                '{}: Reproducability ({}) for remodelling sites with clusters >{} voxel'.format(
                    key, repoducability, min_size), None, None]
            for name, matrix, title in zip(['GRAY_{}'.format(remThr), 'BINARY', 'MNRL'], [
                                           matrTrueRem, matrBinRem, matrMinRem], titles):
                firstEdges = []
                secondEdges = []
                ious = []
                for [b1, f1], [b2, f2] in combinations(
                        (combinations(range(0, series.nTimepoints), 2)), 2):
                    if ((b1 - series.baseline) < (distance + 1)
                        ) & ((b2 - series.baseline) < (distance + 1)):
                        overlap = min(f1, f2) - max(b1, b2)
                        interval = max(f1, f2) - min(b1, b2)
                        if overlap > 0:
                            firstEdges.append((b1, f1))
                            secondEdges.append((b2, f2))

                test_keys = [
                    str(f) +
                    '>' +
                    str(b) for [
                        b,
                        f] in np.unique(
                        firstEdges +
                        secondEdges,
                        axis=0)]
                test_values = [i for i in range(len(test_keys))]

                mapping = {test_keys[i]: test_values[i]
                           for i in range(len(test_keys))}
                iouMatrix = [[np.nan] * (len(test_keys))
                             for i in range(len(test_keys))]
                for (b1, f1), (b2, f2) in zip(firstEdges, secondEdges):
                    if repoducability == 'DIST':
                        iouMatrix[mapping['{}>{}'.format(f2, b2)]][mapping['{}>{}'.format(
                            f1, b1)]] = - hdistance(matrix[f1][b1], matrix[f2][b2])
                        iouMatrix[mapping['{}>{}'.format(f1, b1)]][mapping['{}>{}'.format(
                            f2, b2)]] = hdistance(matrix[b1][f1], matrix[b2][f2])
                    elif repoducability == 'IOU':
                        iouMatrix[mapping['{}>{}'.format(f2, b2)]][mapping['{}>{}'.format(
                            f1, b1)]] = - iou(matrix[f1][b1], matrix[f2][b2])
                        iouMatrix[mapping['{}>{}'.format(f1, b1)]][mapping['{}>{}'.format(
                            f2, b2)]] = iou(matrix[b1][f1], matrix[b2][f2])

                df = pd.DataFrame(
                    iouMatrix,
                    index=mapping.keys(),
                    columns=mapping.keys())
                df = df.round(4)
                df.dropna(how='all', axis=1, inplace=True)
                df = df.replace(np.nan, '', regex=True)
                df = df.rename_axis(name)
                df = df.reset_index()

                dataframes.append(df)
                datanames.append('TABLE_4_REPRO_{}_{}_REGTO{}'.format(key, name,regto))

                docstring += '\n' + dftotxt(df, name=title)
    '''
    print(docstring)
    return docstring, dataframes, datanames

