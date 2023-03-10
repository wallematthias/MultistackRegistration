import os
import warnings
import numpy as np
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import SimpleITK as sitk
import pandas as pd
from copy import deepcopy
from datetime import datetime
from scipy.ndimage import grey_closing
from skimage.filters import gaussian
from scipy.ndimage import binary_dilation
from skimage.morphology import remove_small_objects

from pint import Quantity

from helpers import aim
from helpers.resize_reposition_image import crop_pad_image
from helpers.registration import Registration
from helpers.remodelling import remodelling
from helpers.transformation import Transformation
from helpers.reporting import table1, table2, table3, table5, randomart, finish

warnings.filterwarnings("ignore")


def generate_settings():

    settings = {}
    settings['baseline'] = [0]
    settings['interpolator'] = 'linear'
    settings['Timelapse sequence'] = None
    settings['Stackcorrect numof images'] = 2
    settings['timelapsed interpolation'] = 3
    settings['synthetic interpolation'] = [3, 23, 5]
    settings['order'] = ['timelapse','synthetic']

    settings['Optimizer (timelapse)'] = 'powell'
    settings['Metric (timelapse)'] = 'correlation'
    settings['Sampling (timelapse)'] = 0.01
    settings['MaxIter (timelapse)'] = 100
    settings['Interpolator (timelapse)'] = 'linear'
    settings['Initial rotation (timelapse)'] = None
    settings['Initial translation (timelapse)'] = None
    settings['Shrink factors (timelapse)'] = [12, 8, 4, 2, 1, 1]
    settings['Smoothing sigmas (timelapse)'] = [0, 0, 0, 0, 1, 0]

    settings['Optimizer (stackcorrect)'] = 'powell'
    settings['Metric (stackcorrect)'] = 'correlation'
    settings['Sampling (stackcorrect)'] = 0.1
    settings['MaxIter (stackcorrect)'] = 100
    settings['Interpolator (stackcorrect)'] = 'linear'
    settings['Initial rotation (stackcorrect)'] = [0, 0, 0]
    settings['Initial translation (stackcorrect)'] = [0, 0, 0]
    settings['Shrink factors (stackcorrect)'] = [1, 1]
    settings['Smoothing sigmas (stackcorrect)'] = [1, 0]
    settings['Overlap region'] = 15

    settings['Remodelling threshold'] = 225
    settings['Minimum cluster size'] = 12
    settings['Report depth'] = 3

    return settings


def T(inp: list) -> list:
    return [list(sublist) for sublist in list(zip(*inp))]


def asslice(a):
    return tuple([slice(a[0][0], a[0][1]), slice(
        a[1][0], a[1][1]), slice(a[2][0], a[2][1])])


def pad(im, dim):
    x, y, z = im.shape
    padded_im = np.zeros(dim)
    padded_im[0:x, 0:y, 0:z] = im
    return padded_im


def getfilter(filter: str):

    gaussfilt = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussfilt.SetSigma(0.8)

    filters = {
        "AdditiveGaussianNoise": sitk.AdditiveGaussianNoiseImageFilter(),
        "Bilateral": sitk.BilateralImageFilter(),
        "BinomialBlur": sitk.BinomialBlurImageFilter(),
        "BoxMean": sitk.BoxMeanImageFilter(),
        "BoxSigmaImageFilter": sitk.BoxSigmaImageFilter(),
        "CurvatureFlow": sitk.CurvatureFlowImageFilter(),
        "DiscreteGaussian": sitk.DiscreteGaussianImageFilter(),
        "LaplacianSharpening": sitk.LaplacianSharpeningImageFilter(),
        "Mean": sitk.MeanImageFilter(),
        "Median": sitk.MedianImageFilter(),
        "Normalize": sitk.NormalizeImageFilter(),
        "RecursiveGaussian": sitk.RecursiveGaussianImageFilter(),
        "ShotNoise": sitk.ShotNoiseImageFilter(),
        "SmoothingRecursiveGaussian": gaussfilt,
        "SpeckleNoise": sitk.SpeckleNoiseImageFilter(),
    }
    return filters[filter]


class TimelapsedImage:

    def __init__(self, stackHeight=None):
        self.data = None
        self.timepoint = None
        self.stack = None
        self.stacks = None
        self.voxelsize = None
        self.domain = None
        self.units = None
        self.processing_log = None
        self.position = None
        self.stackHeight = stackHeight
        self.binary = False
        self.path = None
        self.shape = None
        self.type = np.int16

    def load_aim(self, path: str):
        aimFile = aim.load_aim(path)



        self.data = aimFile.data.magnitude.astype(self.type)
        
        if len(np.unique(self.data)) < 5:
            self.data = self.data > 0
            self.binary = True
            
        if self.stackHeight is None:
            self.stackHeight = self.data.shape[2]

        self.units = aimFile.data.units
        self.processing_log = aimFile.processing_log
        self.position = aimFile.position
        self.voxelsize = aimFile.voxelsize
        self.timepoint = self._getCreationDate(aimFile.processing_log)
        self.path = path
        self.shape = self.data.shape
        self.stacks = [
            (slice(0, None), slice(0, None), slice(i, i + self.stackHeight))
            for i in range(0, self.data.shape[2], self.stackHeight)]

        # Save stacks and the "domains" of the stacks
        self.stack = self._genStack(self.data.astype(self.type))
        self.domain = self._genStack(np.ones_like(self.data).astype(bool))

        #print('Loaded {}-stack AIM file ({}): {}'.format(len(self.stacks), self.timepoint, path))

    def _genStack(self, data):
        stacks = []
        for stack in self.stacks:
            tmp = np.zeros_like(data).astype(self.type)
            tmp[stack] = data[stack]
            stacks.append(tmp)
        return stacks

    def _getCreationDate(self, processing_log: str):

        # Create datetime string
        datetime_str = processing_log['CreationDate']

        # call datetime.strptime to convert
        # it into datetime datatype
        datetime_obj = datetime.strptime(datetime_str,
                                         '%d-%b-%Y %H:%M:%S.%f')

        return datetime_obj

    def save_aim(self, path: str, arr=None):
        print('Writing {}'.format(path.replace('.AIM','.mha')).ljust(133, '-'))
        if arr is None:
            arr = self.data

        if self.binary:
            arr = arr > 0.5  # remove border artefacts
        
        aim.write_aim(
            aim.AIMFile(arr,
                        self.processing_log,
                        self.voxelsize,
                        self.position),
            path)
        
       

    def filter(self, filt='XCT2'):
        if filt == 'XCT2':
            filtImage = deepcopy(self)
            filtImage.data = gaussian(
                float(
                    filtImage.data),
                sigma=0.8,
                truncate=1.25)
            filtImage.stack = filtImage._genStack(filtImage.data)
            return filtImage
        else:
            filtImage = deepcopy(self)
            sitkimage = sitk.GetImageFromArray(filtImage.data)
            filtImage.data = sitk.GetArrayFromImage(
                getfilter(filt).Execute(sitkimage))
            filtImage.stack = filtImage._genStack(filtImage.data)
            return filtImage

    def mask(self, other):
        mask = deepcopy(self)
        mask.data[other == 0] = 0
        for ind, _ in enumerate(self.stack):
            mask.stack[ind][other == 0] = 0
        return mask

    def __add__(self, other):
        addition = deepcopy(self)
        addition.data = addition.data + other.data
        addition.stack = [s + o for s, o in zip(addition.stack, other.stack)]
        return addition

    def __or__(self, other):
        comp = deepcopy(self)
        comp.data = (comp.data > 0) | (comp.data > 0)
        comp.stack = [s | o for s, o in zip(comp.stack, other.stack)]
        comp.domain = [
            s | o for s,
            o in zip(
                comp.domain,
                other.domain)]  # added
        return comp

    def __mul__(self, other):
        # this does actually osmething funny and "combines images"
        xorx = deepcopy(self)
        for ind, stack in enumerate(other.stack):
            index = (xorx.domain[ind] == 0) & (other.domain[ind] > 0)
            xorx.stack[ind][index] = other.stack[ind][index]
            xorx.domain[ind][index] = other.domain[ind][index]
        return xorx

    def __lt__(self, other):
        thr = deepcopy(self)
        thr.data = thr.data < other
        thr.stack = thr._genStack(thr.data)
        thr.path = thr.path.replace('.AIM', '_SEG.AIM')
        thr.binary = True
        return thr

    def __gt__(self, other):
        thr = deepcopy(self)
        thr.data = thr.data > other
        thr.stack = thr._genStack(thr.data)
        thr.path = thr.path.replace('.AIM', '_SEG.AIM')
        thr.binary = True
        return thr


class TimelapsedImageSeries:
    def __init__(self, stackHeight=None):
        e = datetime.now()

        self.data = {}
        self.shapes = []
        self.timepoints = {}
        self.transform = Transformation()
        self.interpolator = None
        self.filtering = None
        self.stacktype = None
        self.movingTimepoints = None
        self.fixedTimepoints = None
        self.nTimepoints = None
        self.sequence = None
        self.results = {}
        self.filledByTimelapse = {}
        self.filledSynthetic = {}
        self.saveTimepoint = None
        self.report = randomart(e)
        self.dataframes = []
        self.datanames = []
        self.loaded_transform = False
        print(self.report)

    def shape(self):
        return np.amax(self.shapes, axis=0)

    def addData(
            self,
            data: dict,
            stackHeight=None,
            sortby=None,
            min_pad=5) -> None:

        for key, items in data.items():

            if key not in self.data:
                self.data[key] = []
                self.timepoints[key] = []
            if isinstance(items, list):
                for item in items:
                    tmp = TimelapsedImage(stackHeight)
                    tmp.load_aim(item)
                    self.data[key].append(tmp)
                    self.timepoints[key].append(tmp.timepoint)
                    # self.shapes.append(tmp.data.shape)
            else:
                tmp = TimelapsedImage(stackHeight)
                tmp.load_aim(items)
                self.data[key].append(tmp)
                self.timepoints[key].append(tmp.timepoint)
                # self.shapes.append(tmp.data.shape)

        # adds an all 1 mask if no mask is provided    
        if len(self.data.keys())==1:
            print('Adding all 1 mask')
            self.data['FULL_MASK'] = deepcopy(self.data[next(iter(self.data))])
            for i, im in enumerate(self.data['FULL_MASK']):
                self.data['FULL_MASK'][i].data=np.ones_like(self.data['FULL_MASK'][i].data)
                self.data['FULL_MASK'][i].binary=True
                self.data['FULL_MASK'][i].path=self.data['FULL_MASK'][i].path.replace('.AIM','_FULL_MASK.AIM')
            self.timepoints['FULL_MASK'] = self.timepoints[next(iter(self.data))]

        if sortby is None:
            sortby = next(iter(self.data))

        aquisitionTimes = self.timepoints[sortby]
        self.nTimepoints = len(aquisitionTimes)


        for key in data.keys():
            _, data[key] = (list(t) for t in zip(
                *sorted(zip(aquisitionTimes, data[key]))))

        self.cropSameSize(min_pad=min_pad)

        self.transform.imshape = self.shape()
        report, df, names = table1(self)
        self.dataframes += df
        self.datanames += names
        self.report += report

    def cropSameSize(self, min_pad=5):
        # Crops to the first mask key provided

        if len(list(self.data.keys())) > 1:
            ref_ims = [np.pad(ref_im.data, ((min_pad, min_pad), (min_pad, min_pad), (
                0, 0)), 'constant') for ref_im in self.data[list(self.data.keys())[1]]]
            ref_pos = [np.subtract(ref_im.position, [min_pad, min_pad, 0])
                       for ref_im in self.data[list(self.data.keys())[1]]]

            for ind, (ref_im_min_pad, ref_im_min_pad_pos) in enumerate(
                    zip(ref_ims, ref_pos)):
                for key in self.data.keys():
                    self.data[key][ind].data = crop_pad_image(
                        ref_im_min_pad,
                        self.data[key][ind].data,
                        ref_img_position=ref_im_min_pad_pos,
                        resize_img_position=self.data[key][ind].position)
                    self.data[key][ind].stack = self.data[key][ind]._genStack(
                        self.data[key][ind].data)
                    self.data[key][ind].domain = self.data[key][ind]._genStack(
                        np.ones_like(self.data[key][ind].data))
                    # preserving this stuff
                    self.data[key][ind].position = ref_im_min_pad_pos
                    self.shapes.append(ref_im_min_pad.shape)

    def addRegistration(
            self,
            timelapse: Registration = None,
            stackcorr: Registration = None,
            keyImage=None,
            keyMask=None,
            sequence=None,
            nImages=None,
            nOverlap=20,
            **kwargs):
        self.timelapse = timelapse
        self.stackcorr = stackcorr
        self.keyImage = keyImage
        self.keyMask = keyMask
        self.sequence = sequence

        if self.keyMask == []:
            self.keyMask = ['FULL_MASK',]

        if self.keyImage is None:
            self.keyImage = next(iter(self.data))

        self.setSequence(self.sequence)
        self.baseline = self.baselines[0]

        if nImages is None:
            nImages = self.nTimepoints

        self.nImages = nImages
        self.nOverlap = nOverlap

        report, df, names = table2(self)
        self.dataframes += df
        self.datanames += names
        self.report += report

    def runRegistration(self, **kwargs):
        if self.timelapse is not None:
            self.registerTimelapse()

        if self.stackcorr is not None:
            self.registerStacks()

        report, df, names = table3(self)
        self.dataframes += df
        self.datanames += names
        self.report += report

    def setSequence(self, sequence=None):

        timepoints = [t for t in range(0, self.nTimepoints)]

        if sequence is None:
            self.baselines = [timepoints[0]] * len(timepoints[1:])
            self.followups = timepoints[1:]
        elif len(sequence) == 1:
            self.baselines = [timepoints[sequence[0]], ] * len(timepoints[1:])
            self.followups = [t for t in timepoints if t != sequence[0]]
        else:
            self.baselines = [t[0] for t in sequence[:self.nTimepoints-1]]
            self.followups = [t[1] for t in sequence[:self.nTimepoints-1]]

    def loadTransform(self, path: str):
        self.loaded_transform = True
        #self.transform = Transformation()
        self.transform.loadTransform(path)
        
        if tuple(self.shape()) != tuple(self.transform.imshape):
            raise IOError('Loaded Images compound dimension: {}, the loaded registration was run with {} and is not compatible. To fix this use the original files you used for running the registration or update the registration'.format(tuple(self.shape()),tuple(self.transform.imshape)))
        

    def save(self, t, path: str, interpolator='linear', order=['timelapse', 'synthetic'], n_images=3, min_size=3, max_size=23, step=5, **kwargs):

        filename = self.keyImage.replace(
            '.AIM',
            '').replace(
            '?',
            '').replace(
            '*',
            '')
        self.transform.saveTransform(os.path.join(path, filename))
  
        #save raw densities with new size
        for temp in range(0, self.nTimepoints):
            raw_im = self.get(self.keyImage,temp)
            raw_im.save_aim(
                os.path.join(
                    path,
                    os.path.basename(
                        raw_im.path).replace(
                        '.AIM',
                        '_DENSITY.AIM')))
                
        #save common masks
        for key in self.keyMask:
            #if key !='FULL_MASK':
                for temp in range(0, self.nTimepoints):
                    masks = self.get(key, to=temp, stackCorr=False)
                    comm_mask = deepcopy(masks[temp])
                    comm_mask.data = np.all([mask.data>0 for mask in masks],axis=0)
                    comm_mask.save_aim(
                        os.path.join(
                            path,
                            os.path.basename(
                                comm_mask.path).replace(
                                '.AIM',
                                '_COMM.AIM'.format(temp))))
        
        # This is the stack corr common mask now 
        for key in self.keyMask:
            if key !='FULL_MASK':
                masks = self.get(key, to=t, stackCorr=True)
                comm_mask = deepcopy(masks[t])
                comm_mask.data = np.all([mask.data>0 for mask in masks],axis=0)
                comm_mask.save_aim(
                    os.path.join(
                        path,
                        os.path.basename(
                            comm_mask.path).replace(
                            '.AIM',
                            '_COMM_REGTO_{}.AIM'.format(t))))

        # Save remodelling images 
        for im in self.remodelling_images:
            im.save_aim(
                os.path.join(
                        path,
                        os.path.basename(
                            im.path)))


        originaldata = [
            item for item in list(self.data.keys()) if ('_seg' not in item) and ('_filt' not in item)]
        for key in originaldata:
            if key !='FULL_MASK':
                images = self.get(key, to=t, order=order, n_images=n_images, min_size=min_size, max_size=max_size,step=step, interpolator=interpolator) 
                for image in images:
                    image.save_aim(
                        os.path.join(
                            path,
                            os.path.basename(
                                image.path).replace(
                                '.AIM',
                                '_REGTO_{}.AIM'.format(t))))
                    

        report, df, names = table5(self)
        self.dataframes += df
        self.datanames += names
        self.report += report
        self.saveTimepoint = t

        for x, df in zip(self.datanames, self.dataframes):
            sample_str = x  # ''.join(item for item in x if item.isalnum())
            saveto = os.path.join(
                path,
                filename,
                filename + '_' + sample_str + '.csv')

            df.to_csv(str(saveto))

        e = datetime.now()
        self.report += finish(e)

        with open(os.path.join(path, filename + '_REGTO_{}_TIMELAPSE_LOG.txt'.format(t)), 'w') as f:
            f.write(self.report)

        print(finish(e))

    def get(
            self,
            keys,
            t=None,
            to=None,
            order=[],
            interpolator='linear',
            n_images=3,
            min_size=3,
            max_size=23,
            step=5,
            stackCorr=True):

        if not isinstance(keys, list):
            keys = [keys]

        if t is not None:
            timepoints = [t, ]
            data = [[deepcopy(self.data[key][t]), ] for key in keys]
        else:
            timepoints = [t for t in range(0, self.nTimepoints)]
            data = [deepcopy(self.data[key][:]) for key in keys]

        merged = np.sum(data, axis=0)

        padded = [self._padIm(m) for m in merged]

        if to is not None:
            transformed = []
            for ind, time in enumerate(timepoints):
                transformedIm = padded[ind]
                self.transform.transform(
                    transformedIm,
                    source=time,
                    target=to,
                    interpolator=interpolator,
                    stackCorr=stackCorr)
                transformedIm.data = deepcopy(self.transform.resampledImage)
                transformedIm.domain = deepcopy(
                    self.transform.resampledDomains)
                transformedIm.stack = deepcopy(self.transform.resampledStacks)
                transformed.append(transformedIm)

            transformed = self.interpolate(
                transformed,
                keys,
                n_images=n_images,
                order=order,
                min_size=min_size,
                max_size=max_size,
                step=step)

        else:
            transformed = padded

        if len(transformed) == 1:
            return transformed[0]
        else:
            return transformed

    def registerTimelapse(self, **kwargs):

        timelapse = self.timelapse
        temp_initial_transform = deepcopy(timelapse.initial_transform)
        # This adds the "timelapse" registration to the queue.
        for fixed_t, moving_t in zip(self.baselines, self.followups):

            fixed_im = self.get(self.keyImage, fixed_t)
            moving_im = self.get(self.keyImage, moving_t)

            if self.keyMask is not None:
                fixed_mask = self.get(self.keyMask, fixed_t)
                moving_mask = self.get(self.keyMask, moving_t)

                if len(fixed_mask.stacks) != len(moving_mask.stacks):
                    raise Exception('Masks have different number of stacks')

            if len(fixed_im.stacks) != len(moving_im.stacks):
                raise Exception('Images have different number of stacks')

            
            initial_stack_guess = None
            timelapse.initial_transform = temp_initial_transform
            
            for stackInd, stackIm in reversed(list(enumerate(fixed_im.stacks))):
                if not self.transform.exists(
                        source=moving_t, target=fixed_t, stack=stackInd):
                    # Set registration content
                    timelapse.setRegistrationParamaters(
                        fixed_im.stack[stackInd], moving_im.stack[stackInd])

                    # Set registratioin mask
                    if self.keyMask is not None:
                        timelapse.setRegistrationMask(
                            fixed_mask.stack[stackInd], moving_mask.stack[stackInd])
                    
                    if initial_stack_guess is not None:
                        timelapse.initial_transform = initial_stack_guess
                    
                    # We have to execute right away
                    print('\nTimelapse: {} > {}  for stack {} (target: {} with mask: {})'.format(
                        moving_t, fixed_t, stackInd, self.keyImage, '+'.join(self.keyMask)).ljust(133, '-'))

                    reg_im = timelapse.execute()

                    self.transform.addTransform(
                        timelapse.get_transform(),
                        source=moving_t,
                        target=fixed_t,
                        stack=stackInd,
                        metric=timelapse.reg.GetMetricValue())
                    
                    initial_stack_guess = timelapse.get_transform().GetParameters()
                    
    def registerStacks(self, **kwargs):

        if self.loaded_transform:
            print('Stack correction was loaded')
        else:

            stackcorr = self.stackcorr

            timepoints = [t for t in range(0, self.nTimepoints)]
            timedistance = [abs(t - self.baseline)
                            for t in range(0, self.nTimepoints)]

            _, using_timepoints = (list(t) for t in zip(
                *sorted(zip(timedistance, timepoints))))

            images = [self.get(self.keyImage, t, to=self.baseline)
                      for t in using_timepoints[:self.nImages]]
            superStacks = np.prod(images, axis=0).stack

            if self.keyMask is not None:
                masks = [self.get(self.keyMask, t, to=self.baseline)
                         for t in using_timepoints[:self.nImages]]
                superStackMasks = np.prod(masks, axis=0).stack

            individualStackTransforms = []
            individualMetrics = []
            compositeMetrics = [1, ]
            compositeStackTransforms = [sitk.Euler3DTransform(), ]

            for stackInd in range(0, len(superStacks[:-1])):

                stackcorr.setRegistrationParamaters(
                    superStacks[stackInd], superStacks[stackInd + 1])

                if self.keyMask is not None:
                    overlap = (
                        binary_dilation(superStackMasks[stackInd] > 0, iterations=self.nOverlap) &
                        binary_dilation(superStackMasks[stackInd + 1] > 0, iterations=self.nOverlap))

                    stackcorr.setRegistrationMask(
                        (superStackMasks[stackInd] > 0) & overlap, (superStackMasks[stackInd + 1] > 0) & overlap)

                print('\nStack Correction: {} target: {} with mask: {}) for baseline: {} using timepoints: {}'.format(
                    stackInd + 1, self.keyImage, '+'.join(self.keyMask), self.baseline, using_timepoints[:self.nImages]).ljust(133, '-'))

                # We have to execute right away
                registered_stack = stackcorr.execute()

                # We add the transforms and composite them including all previous
                individualStackTransforms.append(stackcorr.get_transform())
                individualMetrics.append(stackcorr.reg.GetMetricValue())

                compositeTransform = sitk.CompositeTransform(
                    individualStackTransforms)
                compositeTransform.FlattenTransform()

                compositeStackTransforms.append(compositeTransform)
                compositeMetrics.append(np.mean(individualMetrics))

            for indx, (correction, metric) in enumerate(
                    zip(compositeStackTransforms, compositeMetrics)):

                for t in timepoints:
                    forwardTransform = self.transform.getTransform(
                        source=t, target=self.baseline, stack=indx)
                    backwardTransform = self.transform.getTransform(
                        source=self.baseline, target=t, stack=0)
                    timelapsed_sc = sitk.CompositeTransform(
                        [backwardTransform, correction, forwardTransform])  # They are applied in reverse order
                    timelapsed_sc.FlattenTransform()

                    # Losing my shit over this inverse but it seems to work
                    self.transform.addTransform(
                        timelapsed_sc, source=t, target=t, stack=indx, metric=metric)

    def interpolate(
            self,
            images: list,
            keys,
            order=[
                'timelapse',
                'synthetic'],
            n_images=3,
            min_size=3,
            max_size=23,
            step=5):

        if not isinstance(images, list):
            images = [images, ]
        self.filledByTimelapse[''.join(keys)] = [0, ] * len(images)
        self.filledSynthetic[''.join(keys)] = [0, ] * len(images)

        for filling in order:
            print('Filling missing data with {} data'.format(
                filling).ljust(133, '-'))

            if filling == 'timelapse':
                imagesUnfilt = deepcopy(images)
                images = self._timelapsedFilling(images, n_images=n_images)

                self.filledByTimelapse[''.join(keys)] = [np.sum(abs(image.data - unfilt.data) > 0) / np.sum(
                    unfilt.data > 0) * 100 for image, unfilt in zip(images, imagesUnfilt)]

            elif filling == 'synthetic':
                imagesUnfilt = deepcopy(images)
                images = self._syntheticFilling(
                    images, min_size=3, max_size=23, step=5)

                self.filledSynthetic[''.join(keys)] = [np.sum(abs(image.data - unfilt.data) > 0) / np.sum(
                    unfilt.data > 0) * 100 for image, unfilt in zip(images, imagesUnfilt)]

        return images

    def remodelling(
            self,
            key_image='dens',
            key_mask=[
                'trab',
                'cort'],
            baseline=0,
            thresholds=[
                320,
                450],
        rem_thr=225,
        min_size=12,
        distance=3,
        regto=0):

        if key_mask == []:
            key_mask = ['FULL_MASK',]

        docstring, dataframes, datanames, remodelling_images = remodelling(self, key_image=key_image, key_mask=key_mask, baseline=baseline,
                                                       thresholds=thresholds, rem_thr=rem_thr, min_size=min_size, distance=distance, regto=regto)
        self.report += docstring
        self.dataframes += dataframes
        self.datanames += datanames
        self.remodelling_images = remodelling_images

    def _syntheticFilling(self, images: list, min_size=3, max_size=23, step=5):

        filledimages = []
        for image in images:
            image = deepcopy(image)
            filledImageGaps = deepcopy(image.data)
  
            # iteratively filling the image
            for i in range(min_size, max_size, step):
                # Closing the image but only filling the gaps
                filledImageGaps = grey_closing(
                    image.data, size=(3, 3, i), mode='mirror')
                missing = remove_small_objects(
                    (image.data == 0), min_size=3 * 3 * (i + 1)) #added missingMask here
                image.data[missing] = filledImageGaps[missing]
                # This line is as there is some "natural" zeros in the filled
                # image

            filledimages.append(image)

        return filledimages

    def _timelapsedFilling(self, images: list, n_images=3):

        filledimages = []
        for indx, image in enumerate(images):

            # Starting the timelapsed interpolation
            filledim = deepcopy(image)
            missingMask = np.any(image.domain, axis=0) == 0
            
            
            fill = np.zeros_like(image.data)
            includedImages = self._nClosestIm(images, indx, n_images)
            for im in includedImages:
                fill[missingMask & (fill == 0)
                     ] = im.data[missingMask & (fill == 0)]

            filledim.data = image.data + fill
            
            filledimages.append(filledim)

        return filledimages

    def _nClosestIm(self, images, indx, n_images):
        timepoints = [image.timepoint for image in images]
        differences = [abs(image.timepoint - images[indx].timepoint)
                       for image in images]
        include = np.isin(timepoints, [x for _, x in sorted(
            zip(differences, timepoints))][:n_images + 1])
        includedImages = [im for im, incl in zip(images, include) if incl]
        return includedImages

    def _padIm(self, image: TimelapsedImage):
        image.data = self._padArr(image.data)
        image.stack = [self._padArr(s) for s in image.stack]
        image.domain = [self._padArr(d) for d in image.domain]
        return image

    def _padArr(self, im):
        dim = self.shape()
        x, y, z = im.shape
        padded_im = np.zeros(dim)
        padded_im[0:x, 0:y, 0:z] = im
        return padded_im
