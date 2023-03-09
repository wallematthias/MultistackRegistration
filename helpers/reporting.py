from itertools import permutations, combinations
import pandas as pd
import os
import numpy as np
import re
from io import StringIO
#import termplotlib as tpl
from skimage.measure import regionprops, regionprops_table
from scipy import stats
from scipy import ndimage

def dftotxt(df, name=None, width=130):
    current_size = len(df.to_markdown(index=False).split('\n')[0])
    min_width = np.max([0, width - current_size])

    col_len = len(df.to_markdown(index=False).split('\n')[0].split('|')[1])
    new_name = str(df.columns[0]).ljust(min_width + col_len)
    df = df.rename(columns={df.columns[0]: new_name})

    data = str(df.to_markdown(index=False, maxcolwidths=min_width))
    if name is None:
        header = '\n' + '\n'.rjust(width, '-')
    else:
        header = '+ ' + '\n'.rjust(width - 2, '=') + \
            name + '\n' + '\n'.rjust(width, '-')
    return header + data


def txttodf(data):
    markdown = str('\n'.join(data.split('\n')[3:]))
    tablename = str(''.join(data.split('\n')[1])).replace('  ', '')
    df = pd.read_csv(
        StringIO(markdown.replace(' ', '')),  # Get rid of whitespaces
        sep='|',
        index_col=0
    ).dropna(
        axis=1,
        how='all'
    ).iloc[1:].reset_index(drop=True)

    df.index.name = tablename
    return df


def table1(series):
    voxelsize = []
    filename = []
    creation_time = []
    mask = []
    shape = []
    stacks = []
    timepoints = []
    for key in series.data.keys():
        for ind in range(0, len(series.data[key])):
            timepoints.append(ind)
            voxelsize.append(
                '{:.4}'.format(
                    series.data[key][ind].voxelsize.to('um').magnitude[0]))
            filename.append(os.path.basename(series.data[key][ind].path))
            creation_time.append(
                str(series.data[key][ind].timepoint.replace(microsecond=0)))
            mask.append(bool(series.data[key][ind].binary))
            shape.append(tuple(series.data[key][ind].data.shape))
            stacks.append('n={} (z={})'.format(len(series.data[key][ind].stacks), [
                          stack[2].stop - stack[2].start for stack in series.data[key][ind].stacks]))

    data = {
        'Input files: (Out: + *{})'.format('_REG/FILLED.AIM'): filename,
        't': timepoints,
        'date': creation_time,
        'size [um]': voxelsize,
        'dim': shape,
        'stacks': stacks,
        'binary': mask}
    data_frame = pd.DataFrame(data)
    data = dftotxt(data_frame, 'Table 1: Input/Output files')
    print(data)
    return data, [data_frame, ], ['TABLE_1']


def table2(series):

    keys = [
        'Optimizer',
        'Metric',
        'Sampling',
        'MaxIter',
        'Interpolator',
        'Initial Transform',
        'Multiresolution',
        'Registration Image',
        'Registration Mask',
        'Registration Sequence']
    values_time = [
        series.timelapse.optimizer,
        series.timelapse.metric,
        series.timelapse.sampling,
        series.timelapse.num_of_iterations,
        series.timelapse.interpolatorstring,
        series.timelapse.initial_transform,
        '\nShrink factors: {}\nSmoothing sigmas: {}'.format(
            series.timelapse.shrinkFactors,
            series.timelapse.smoothingSigmas),
        series.keyImage,
        ' & '.join(
            series.keyMask),
        'baseline: {}, followups: {}'.format(
            series.baselines,
            series.followups)]

    if series.stackcorr is not None:
        values_stack = [
            series.stackcorr.optimizer,
            series.stackcorr.metric,
            series.stackcorr.sampling,
            series.stackcorr.num_of_iterations,
            series.stackcorr.interpolatorstring,
            series.stackcorr.initial_transform,
            '\nShrink factors: {}\nSmoothing sigmas: {}'.format(
                series.stackcorr.shrinkFactors,
                series.stackcorr.smoothingSigmas),
            series.keyImage,
            ' & '.join(
                series.keyMask),
            'n={} -> {} with overlap n={}'.format(
                series.nTimepoints,
                series.baseline,
                series.nOverlap)]
        datacorr = {
            'Settings': keys,
            'Timelapse': values_time,
            'Stackcorrect': values_stack}

    else:
        datacorr = {'Settings': keys, 'Timelapse': values_time}
    inputdata = pd.DataFrame(datacorr)
    df = pd.DataFrame(inputdata)

    data = dftotxt(df, 'Table 2: Registration setup', width=130)

    print(data)
    return data, [df, ], ['TABLE_2']


def table3(series):
    avails = []
    stacks = []
    timelapses = []
    stackcorrs = []
    transpaths = []

    for stack in series.transform.data.keys():
        for a, b in combinations(range(series.nTimepoints + 1), r=2):
            try:
                # if 1:
                transform, metric, nodes = series.transform.getTransform(
                    source=a, target=b, stack=stack, verbose=False, metric=True)
                avails += ['{} > {}'.format(a, b)]
                stacks += [stack]
                timelapses += [np.mean(metric[1:])]
                stackcorrs += [metric[0]]
                transpaths += [' > '.join([str(n) for n in nodes])]
            except BaseException:
                pass

    if not np.all(stackcorrs == 1):
        data = {
            'From > To': transpaths,
            'Metric (timelapse)': timelapses,
            'Metric (stackcorr)': stackcorrs,
            'Stack': stacks}
    else:
        data = {
            'From > To': transpaths,
            'Metric (timelapse)': timelapses,
            'Stack': stacks}
    df = pd.DataFrame(data).sort_values(['Stack', 'From > To'])

    data = dftotxt(df, 'Table 3: Registration results', width=130)
    print(data)
    return data, [df, ], ['TABLE_3']


def table5(series):
    fills = []
    synfills = []
    timepoints = []
    images = []
    for key in series.filledByTimelapse.keys():
        for ind in range(len(series.filledByTimelapse[key])):
            try:
                images.append(os.path.basename(series.data[key][ind].path))
                timepoints.append(ind)
                fills.append(series.filledByTimelapse[key][ind])
                synfills.append(series.filledSynthetic[key][ind])
            except BaseException:
                pass

    data = {
        'Image': images,
        'Timepoints [>{}]'.format(
            series.saveTimepoint): timepoints,
        'Interpolation [%] (timelapse)': fills,
        'Interpolation [%] (synthetic)': synfills}
    df = pd.DataFrame(data)
    data = dftotxt(df, 'Table 5: Interpolation results', width=130)
    print(data)
    return data, [df, ], ['TABLE_5']


def table6(series):


    label_im, nb_labels = ndimage.label(bin_formation)
    regions = regionprops_table(label_im,intensity_image=rem,properties=('area',"max_intensity",'min_intensity','mean_intensity','major_axis_length','minor_axis_length','centroid'))
    df_raw =pd.DataFrame(regions)
    df_raw.to_csv('table_6_')
    df = df_raw.sort_values(['area'])
    df = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]
    grapstr = ""

    grapstr += 'Table 6: Bone Formation '.ljust(130,'=')
    grapstr +=('\nX-Coord [voxel] \t\t\t\t\t Y-Coord [voxel]  \t\t\t\t Z-Coord [voxel]\n')

    subplot = tpl.subplot_grid([1,3])
    counts, bin_edges = np.histogram(df['centroid-0'])
    subplot[0,0].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['centroid-1'])
    subplot[0,1].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['centroid-2'])
    subplot[0,2].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    grapstr += subplot.get_string()

    grapstr +=('\nMin density change [mg/ccm] \t\t\t Max density change [mg/ccm] \t\t\t Mean density change [mg/ccm]\n')
    subplot = tpl.subplot_grid([1,3])
    counts, bin_edges = np.histogram(df['min_intensity'])
    subplot[0,0].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['max_intensity'])
    subplot[0,1].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['mean_intensity'])
    subplot[0,2].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    grapstr += subplot.get_string()

    df = df[(np.abs(stats.zscore(df)) < 1).all(axis=1)]
    grapstr+=('\nSize [voxel] \t\t\t\t\t Minor Axis [voxel] \t\t\t\t\t Major Axis [voxel]\n')
    subplot = tpl.subplot_grid([1,3])
    counts, bin_edges = np.histogram(df['area'])
    subplot[0,0].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['major_axis_length'])
    subplot[0,1].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['minor_axis_length'])
    subplot[0,2].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    grapstr += subplot.get_string()







def table7(series):
    label_im, nb_labels = ndimage.label(bin_resorption)
    regions = regionprops_table(label_im,intensity_image=rem,properties=('area',"max_intensity",'min_intensity','mean_intensity','major_axis_length','minor_axis_length','centroid'))
    df_raw2 =pd.DataFrame(regions)
    df_raw2.to_csv('table_7_')
    df = df_raw2.sort_values(['area'])
    df = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]

    grapstr += '\n Table 7: Bone Resorption '.ljust(130,'=')
    grapstr +=('\nX-Coord [voxel] \t\t\t\t\t Y-Coord [voxel]  \t\t\t\t Z-Coord [voxel]\n')

    subplot = tpl.subplot_grid([1,3])
    counts, bin_edges = np.histogram(df['centroid-0'])
    subplot[0,0].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['centroid-1'])
    subplot[0,1].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['centroid-2'])
    subplot[0,2].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    grapstr += subplot.get_string()

    grapstr +=('\nMin density change [mg/ccm] \t\t\t Max density change [mg/ccm] \t\t\t Mean density change [mg/ccm]\n')
    subplot = tpl.subplot_grid([1,3])
    counts, bin_edges = np.histogram(df['min_intensity'])
    subplot[0,0].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['max_intensity'])
    subplot[0,1].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['mean_intensity'])
    subplot[0,2].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    grapstr += subplot.get_string()

    df = df[(np.abs(stats.zscore(df)) < 1).all(axis=1)]
    grapstr+=('\nSize [voxel] \t\t\t\t\t Minor Axis [voxel] \t\t\t\t\t Major Axis [voxel]\n')
    subplot = tpl.subplot_grid([1,3])
    counts, bin_edges = np.histogram(df['area'])
    subplot[0,0].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['major_axis_length'])
    subplot[0,1].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    counts, bin_edges = np.histogram(df['minor_axis_length'])
    subplot[0,2].hist(counts, bin_edges, orientation="horizontal", force_ascii=False, max_width=5)
    grapstr += subplot.get_string()

    print(grapstr)


def randomart(e):

    art = ('Hi? Can you register me?\n' +
           ' .-. .-.       .-. .-.\n' +
           '(   Y   )     (   Y   )\n' +
           '  |   |         |   |\n' +
           '  | 00|_       _|00 |\n' +
           '  |  ,__)     (__,  |\n' +
           '  |,_|           L_,|\n' +
           '  ||               ||\n' +
           '  | \\_,         ,_/ |\n' +
           '  |   |         |   |\n' +
           '  |   |         |   |\n' +
           ' (  A  )       (  A  )\n' +
           "'-' '-'       '-' '-'\n")
    center = "\n".join(line.center(125) for line in art.split("\n"))
    timestring = 'Multistack Registration v.1.0.0 {}\n'.format(e).center(125)

    return '\n'.rjust(130, '=') + center + ' \n' + \
        timestring + '\n' + '\n'.rjust(130, '=')


def finish(e):
    timestring = 'Multistack Registration v.1.0.0 {}\n'.format(e).center(125)
    return '\n' + '\n'.rjust(130, '=') + ' \n' + \
        timestring + '\n' + '\n'.rjust(130, '=')
