import os
import yaml
import numpy as np
from glob import glob
import networkx as nx
import SimpleITK as sitk
from copy import deepcopy
from skimage.filters import gaussian
from helpers.registration import Registration, strToSitkInterpolator


class Transformation:

    def __init__(self):
        self.data = {}
        self.imshape = None

        # Initiate temporary views
        self.resampledImage = None
        self.resampledDomains = None
        self.resampledStacks = None

    def addTransform(self, transform, source=0, target=0, stack=0, metric=0):

        # If the "stackindex" does not exist add a new graph (for each stack)
        if stack not in self.data:
            self.data[stack] = nx.DiGraph()  # Make Graph

        # Adding transform
        self.data[stack].add_edge(
            source,
            target,
            transform=transform,
            finalMetric=metric,
            label='{}TO{}'.format(
                source,
                target))

        # Adding inverse transform
        if source != target:
            self.data[stack].add_edge(
                target,
                source,
                transform=transform.GetInverse(),
                finalMetric=metric,
                label='{}TO{}'.format(
                    source,
                    target))

        # Adding source eigentransform if not exist
        if self.data[stack].has_edge(source, source) == False:
            self.data[stack].add_edge(
                source,
                source,
                transform=sitk.Euler3DTransform(),
                finalMetric=1,
                label='{}TO{}'.format(
                    source,
                    source))

        # Adding target egientransform if not exist
        if self.data[stack].has_edge(target, target) == False:
            self.data[stack].add_edge(
                target,
                target,
                transform=sitk.Euler3DTransform(),
                finalMetric=1,
                label='{}TO{}'.format(
                    target,
                    target))

    def getTransform(
            self,
            source=0,
            target=0,
            stack=0,
            metric=False,
            verbose=False,
            stackCorr=True):

        if self.exists(source, target, stack):
            # Get shortest path as combination of transformaitons
            sp = nx.shortest_path(
                self.data[stack],
                source=source,
                target=target)
            pathGraph = nx.path_graph(sp)  # does not pass edges attributes

            # Extract the edge metrics again
            transformations = [self.data[stack].edges[ea[0],
                                                      ea[1]]['transform'] for ea in pathGraph.edges()]
            metrics = [self.data[stack].edges[ea[0], ea[1]]
                       ['finalMetric'] for ea in pathGraph.edges()]
            labels = [self.data[stack].edges[ea[0], ea[1]]['label']
                      for ea in pathGraph.edges()]

            # The stackTransformations will be saved in the
            # "eigentransformation"
            if stackCorr: #here was a problem that the common masks were not correct I hope this fixes it but I think maybe these should be saved within each transformation matrix so that it is proper
                stackTransforms = [
                    self.data[stack].edges[target, target]['transform'], ]
            else:
                stackTransforms = [
                    sitk.Euler3DTransform(), ]                
            stackMetric = [
                self.data[stack].edges[target, target]['finalMetric'], ]
            stackLabel = [self.data[stack].edges[target, target]['label'], ]

            # Making the composite transform
            composite_transform = sitk.CompositeTransform(
                stackTransforms + transformations)
            composite_transform.FlattenTransform()

            # Printing stuff
            if verbose:
                print(
                    "Stack: {}. Shortest transformation path: {} with metrics: {}: ".format(
                        stack, sp, stackMetric + metrics))
                print(stackLabel + labels)

            if metric:
                return composite_transform, stackMetric + metrics, list(sp)

            else:
                return composite_transform

        else:
            # Just return something in case
            #print('Warning: Returning empty transformation')
            return sitk.Euler3DTransform()

    def transform(
            self,
            image,
            source,
            target,
            interpolator='linear',
            filtering=False,
            stacktype=None,
            stackCorr=True) -> sitk.Image:
        '''
        Apply the transform to a different image

        Args:
            transform_arr
        Returns:
            Image: registered image
        '''

        # This function first transforms stacks
        self.resampledStacks, self.resampledDomains = self.transformStacks(
            image, source, target, stackCorr=stackCorr)

        # This function combines them
        self.resampledImage = self.combineStacks(
            self.resampledStacks, self.resampledDomains)

        return self.resampledImage

    def transformStacks(
            self,
            image,
            source,
            target,
            interpolator='linear',
            filtering=False,
            sharpEdges=True,
            stackCorr=True):

        # These are only temporary based on the last transformation requested
        #print('Using interpolator '+interpolator)
        self.resampledStacks = []
        self.resampledDomains = []

        for indx, (stack, domain) in enumerate(zip(image.stack, image.domain)):
            # Get transformation
            transform = self.getTransform(
                source=source, target=target, stack=indx, stackCorr=stackCorr)

            # Cast ITK images
            im = sitk.Cast(
                sitk.GetImageFromArray(
                    stack.astype(int)),
                sitk.sitkFloat32)
            im_domain = sitk.Cast(
                sitk.GetImageFromArray(
                    domain.astype(int)),
                sitk.sitkFloat32)

            # Resample Images and get array
            resampled_stack = sitk.GetArrayFromImage(
                sitk.Resample(
                    im,
                    im,
                    transform,
                    strToSitkInterpolator(interpolator),
                    0.0,
                    im.GetPixelID()))
            resampled_domain = sitk.GetArrayFromImage(
                sitk.Resample(
                    im_domain,
                    im_domain,
                    transform,
                    strToSitkInterpolator(interpolator),
                    0.0,
                    im_domain.GetPixelID())) == 1

            if filtering & (transform.GetParameters() ==
                            transform.GetInverse().GetParameters()):
                print('filteringing')
                resampled_stack = gaussian(
                    resampled_stack, sigma=0.8, truncate=1.25)

            # Crop stack to domain (remove border artefacts)
            if sharpEdges:
                resampled_stack *= resampled_domain

            # Append resampled images temporary
            self.resampledStacks.append(resampled_stack)
            self.resampledDomains.append(resampled_domain)

        return self.resampledStacks, self.resampledDomains

    def combineStacks(self, stacks, domains, stacktype=None):

        # Initiate the resampled image
        combinedIm = np.zeros_like(stacks[0])

        # stacktype bot prefers the bottom stacks over the top stack
        if stacktype == 'bot':
            for indx, (stack, domain) in enumerate(zip(stacks, domains)):
                combinedIm[domain] = stack[domain]

        # All other stacktypes prefer the "top" stack
        else:
            for indx, (stack, domain) in enumerate(zip(stacks, domains)):
                fillIndx = (domain > 0) & (combinedIm == 0)
                combinedIm[fillIndx] = stack[fillIndx]

        return combinedIm

    def exists(self, source=0, target=0, stack=0):
        try:
            # detect shortest path for requested transform
            sp = nx.shortest_path(
                self.data[stack],
                source=source,
                target=target)
            #print('Transform {} to {} for stack {} is reachable'.format(source,target,stack))
            return 1

        except Exception as e:
            try:
               # print('Requested transform {} to {} for stack {}'.format(source,target,stack))

                # detect all reachable nodes
                reach = np.sort(
                    list(
                        nx.shortest_path(
                            self.data[stack],
                            source=source).keys()))
               # print('Reachable timepoints {} to {} for stack {}'.format(source, reach, stack))
            except BaseException:
                pass
               # print("Transform does not exist")

            return 0

    def saveTransform(self, path):

        data = deepcopy(self.data)

        dataDict = {}
        for key, value in data.items():
            dataDict[key] = nx.to_dict_of_dicts(value)

        if not os.path.exists(path):
            os.makedirs(path)

        for key1, lvl2 in dataDict.items():
            for key2, lvl3 in lvl2.items():
                for key3, lv4 in lvl3.items():
                    filename = '{}_{}_{}_{}.tfm'.format(
                        key1, key2, key3, os.path.basename(path))
                    transform = dataDict[key1][key2][key3]['transform']
                    transform = self._simplifyCompositeTransform(transform)
                    sitk.WriteTransform(
                        transform, os.path.join(
                            path, filename))

                    with open(os.path.join(path, filename.replace('.tfm', '.yml')), 'w') as handle:
                        yaml.dump({'metric': float(
                            dataDict[key1][key2][key3]['finalMetric'])}, handle, default_flow_style=True)

        shape = [int(a) for a in self.imshape]
        with open(os.path.join(path, os.path.basename(path) + '.yml'), 'w') as handle:
            yaml.dump({'imshape': shape}, handle, default_flow_style=True)

    def loadTransform(self, path):

        try:
            with open(os.path.join(path, os.path.basename(path) + '.yml'), "r") as stream:
                self.imshape = yaml.load(
                    stream, Loader=yaml.FullLoader)['imshape']

            tpaths = glob(os.path.join(path, '*tfm'))

            for tpath in tpaths:
                sitktransform = sitk.CompositeTransform(
                    sitk.ReadTransform(tpath))
                with open(tpath.replace('.tfm', '.yml'), "r") as stream:
                    metric = yaml.load(
                        stream, Loader=yaml.FullLoader)['metric']
                stack, source, target = os.path.basename(tpath).split('_')[:3]
                print(
                    'Adding: stack: {} from {} to {} with metric={}'.format(
                        int(stack),
                        int(source),
                        int(target),
                        format(
                            float(metric),
                            '.4f')).ljust(
                        133,
                        '-'))
                if source == target:
                    self.addTransform(
                        sitktransform,
                        source=int(source),
                        target=int(target),
                        stack=int(stack),
                        metric=float(metric))   # dont know why
                else:
                    self.addTransform(
                        sitktransform,
                        source=int(source),
                        target=int(target),
                        stack=int(stack),
                        metric=float(metric))
        except BaseException:
            pass

    def _simplifyCompositeTransform(self, CompositeTransform):
        CompositeTransform = sitk.CompositeTransform(CompositeTransform)
        CompositeTransform.FlattenTransform()
        numOf = CompositeTransform.GetNumberOfTransforms()
        nonEmpty = []
        for n in range(numOf):
            if CompositeTransform.GetNthTransform(n).GetParameters(
            ) != CompositeTransform.GetNthTransform(n).GetInverse().GetParameters():
                nonEmpty.append(CompositeTransform.GetNthTransform(n))

        if len(nonEmpty) > 0:
            CompositeTransform = sitk.CompositeTransform(nonEmpty)
        else:
            CompositeTransform = sitk.CompositeTransform(
                [sitk.Euler3DTransform()])

        CompositeTransform.FlattenTransform()
        return CompositeTransform
