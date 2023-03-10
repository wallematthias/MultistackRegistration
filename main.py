import os
import yaml
import warnings
import numpy as np
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import sys
sys.path.append("helpers")

from helpers.registration import Registration
from helpers.remodelling import remodelling
from helpers.transformation import Transformation
from helpers.timelapse import TimelapsedImage, TimelapsedImageSeries

def main():
    '''
     python '/cluster/home/wallem/framework/ifb_framework/pipelines/hrpqct/multistack_registration/main.py' --keyImage 'INSR_171_DR_C?.AIM' --keyMask CORT_MASK TRAB_MASK --thresholds 450 320

    '''
    parser = ArgumentParser(description='Registration Pipeline')
    parser.add_argument(
        '--input',
        type=str,
        default='.',
        help='input path will be used with glob')
    parser.add_argument('--keyImage', type=str, help='image key')
    parser.add_argument('--keyMask', nargs='+', default=None, help='mask key')
    parser.add_argument('--transform', default=None, help='path to transform')
    parser.add_argument(
        '--output',
        default='.',
        help='Results will be written here')
    parser.add_argument(
        '--stackHeight',
        type=int,
        default=168,
        help='Results will be written here')

    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=int,
        default = None,
        help='segmentation thresholds within the masks that were provided')

    parser.add_argument(
        '--options',
        default=None)
    
    args = parser.parse_args()

    
    if args.options is None:
        if os.path.exists(
            os.path.join(
                os.path.dirname(
                Path(__file__)),
                'options.yml')):
            # Generate options
            p = os.path.join(os.path.dirname(Path(__file__)), 'options.yml')
            with open(p, 'r') as stream:
                settings = yaml.safe_load(stream)
        else:
            p = os.path.join(os.path.dirname(Path(__file__)), 'options.yml')
            with open(p, 'w') as stream:
                settings = generate_settings()
                yaml.dump(settings, stream, sort_keys=False)
    else:
        with open(os.path.abspath(args.options), 'r') as stream:
            settings = yaml.safe_load(stream)
    
    
    # print(settings)
    files = []

    # Importing the main file first
    files.append(sorted(glob(
        os.path.join(
            os.path.abspath(args.input),
            '{}'.format(args.keyImage)))))

    # importing as many masks as provided
    if args.keyMask is not None:
        for mask in args.keyMask:
            mask = args.keyImage.replace('?', '*{}*'.format(mask))
            foundpaths = sorted(glob(
                os.path.join(
                    os.path.abspath(args.input),
                    '{}'.format(mask))))
            if len(foundpaths) != len(files[0]):
                print(mask)
                raise TypeError('Not enough masks found')
            else:
                files.append(foundpaths)
    else:
        args.keyMask = []

    # parsing the input dict
    input_dict = dict(zip([args.keyImage] + args.keyMask, files))

    # print(input_dict)

    # if not specified write to same path (it should never overwrite)
    if args.output is None:
        args.output = args.input
    else:
        args.output = os.path.abspath(args.output)

    # Checking for output transforms
    default_transform = os.path.join(
        os.path.abspath(
            args.output),
        ('.'.join(
            args.keyImage.split('.')[
                :-
                1])).replace(
                    '?',
                    '').replace(
                        '*',
            ''))
    print(default_transform)
    
    if os.path.exists(default_transform) & (args.transform is None):
        print('Using transforms found in default path: {}'.format(default_transform))
        args.transform = default_transform

    # Inititalising and adding the datas
    series = TimelapsedImageSeries()
    series.addData(input_dict, stackHeight=args.stackHeight)

    timelapse = Registration(
        sampling=settings['Sampling (timelapse)'],
        num_of_iterations=settings['MaxIter (timelapse)'])
    timelapse.setOptimizer(
        settings['Optimizer (timelapse)'])
    timelapse.setMultiResolution(
        shrinkFactors=settings['Shrink factors (timelapse)'],
        smoothingSigmas=settings['Smoothing sigmas (timelapse)'])
    timelapse.setInterpolator(
        settings['Interpolator (timelapse)'])
    timelapse.setSimilarityMetric(
        settings['Metric (timelapse)'])

    if settings['Initial rotation (timelapse)'] is not None:
        timelapse.setInitialTransform(
            settings['Initial rotation (timelapse)'],
            settings['Initial translation (timelapse)'])

    stackcorr = Registration(
        sampling=settings['Sampling (stackcorrect)'],
        num_of_iterations=settings['MaxIter (stackcorrect)'])
    stackcorr.setOptimizer(
        settings['Optimizer (stackcorrect)'])
    stackcorr.setMultiResolution(
        shrinkFactors=settings['Shrink factors (stackcorrect)'],
        smoothingSigmas=settings['Smoothing sigmas (stackcorrect)'])
    stackcorr.setInterpolator(
        settings['Interpolator (stackcorrect)'])
    stackcorr.setSimilarityMetric(
        settings['Metric (stackcorrect)'])
    if settings['Initial rotation (stackcorrect)'] is not None:
        stackcorr.setInitialTransform(
            settings['Initial rotation (stackcorrect)'],
            settings['Initial translation (stackcorrect)'])

    series.addRegistration(
        timelapse=timelapse,
        stackcorr=stackcorr,
        keyMask=args.keyMask,
        sequence = settings['Timelapse sequence'],
        nOverlap=settings['Overlap region'],
        nImages=settings['Stackcorrect numof images'])

    if args.transform is not None:
        series.loadTransform(args.transform)

    series.runRegistration(
        interpolator=settings['interpolator'])

    if settings['Remodelling threshold'] is not None:
        series.remodelling(
            key_image=args.keyImage,
            key_mask=args.keyMask,
            baseline=series.baseline,
            thresholds=args.thresholds,
            rem_thr=settings['Remodelling threshold'],
            min_size=settings['Minimum cluster size'],
            distance=settings['Report depth'],
            regto=series.baseline,
            repoducability=settings['Repoducability'])

    series.save(
        series.baseline,
        args.output,
        interpolator=settings['interpolator'],
        order=settings['order'], 
        n_images=settings['timelapsed interpolation'], 
        min_size=settings['synthetic interpolation'][0], 
        max_size=settings['synthetic interpolation'][1], 
        step=settings['synthetic interpolation'][2]
    )


if __name__ == "__main__":
    main()
