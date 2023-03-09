import numpy as np
import warnings
from skimage.transform import rescale
from pint import Quantity
from aim import pad_to_common_coordinate_system

def crop_pad_image(reference_image, resize_image, ref_img_position=None,
                   resize_img_position=None, delta_position=None, padding_value=None):
    '''
    Function which resizes one image, using the reference image and the change
    of position. Resizing is done by cropping and padding. The resized images
    has the same position and shape as the reference image.

    Parameters
    ----------
    reference_image : 2D- or 3D-array
        Reference Image, according to which resize_image is cropped and padded.
    resize_image : 2D- or 3D-array
        Image which is resized.
    ref_img_position : list of ints, optional
        Position of reference_image, default is None.
    resize_img_position : list of ints, optional
        Position of resize_image, default is None.
    delta_position : list of ints, optional
        Difference between the positions, default is None. Vector pointing from reference_image position to position of resize_image.
    padding_value : int, float, optional
        Value with which array is padded, default is None. If padding_value is None, padding value is set 0.

    Returns
    -------
    resized_image : 2D- or 3D-array
        Resized imaged which has same shape and position as reference image.

    Examples
    --------
    >>> crop_pad_image(reference_image= np.zeros((5,5)),
    ...        resize_image=np.ones((3,3)), delta_position=[2,1])
    [[0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0.]
    [0. 1. 1. 1. 0.]
    [0. 1. 1. 1. 0.]
    [0. 1. 1. 1. 0.]]

    >>> crop_pad_image(reference_image=np.zeros((5,5)),
    ...        resize_image=np.ones((3,3)), delta_position=[1,3], padding_value=2)
    [[2. 2. 2. 2. 2.]
    [2. 2. 2. 1. 1.]
    [2. 2. 2. 1. 1.]
    [2. 2. 2. 1. 1.]
    [2. 2. 2. 2. 2.]]

    '''
    if (ref_img_position is not None or resize_img_position is not None) and delta_position is not None:
        error_message = 'When specifiying delta position, not additional position is needed.'

        raise ValueError(error_message)
    elif (ref_img_position is None or resize_img_position is None) and delta_position is None:
        error_message = 'Positions of both images must be specified.'

        raise ValueError(error_message)

    # calculate delta_position from the two given positions
    if delta_position is None:
        delta_position = np.subtract(resize_img_position, ref_img_position)

    if padding_value is None:
        padding_value = 0

    # calculate delta_position for points in arrays with [x=-1,y=-1,z=-1]
    delta_position_end = np.subtract(np.shape(reference_image), np.add(delta_position, np.shape(resize_image)))

    # establish where to pad and where to slice array
    delta_position_slice = np.zeros(np.shape(delta_position)[0], dtype=int)
    for idx, val in enumerate(delta_position):
        if val >= 0:
            continue
        else:
            delta_position[idx] = 0
            delta_position_slice[idx] = abs(val)

    # establish where to pad and where to slice array
    delta_position_slice_end = np.zeros(np.shape(delta_position)[0], dtype=int)
    for idx, val in enumerate(delta_position_end):
        if val >= 0:
            continue
        else:
            delta_position_end[idx] = 0
            delta_position_slice_end[idx] = val

    # solve problem when there was no slicing from the end. any number causes slicing (as index is exclusive)
    delta_position_slice_end_z = None
    for axis, val in enumerate(delta_position_slice_end):
        if val != 0:
            if axis == 0:
                delta_position_slice_end_x = val
            elif axis == 1:
                delta_position_slice_end_y = val
            elif axis == 2:
                delta_position_slice_end_z = val
        else:
            if axis == 0:
                delta_position_slice_end_x = None
            elif axis == 1:
                delta_position_slice_end_y = None

    delta_position_slice_tuple = tuple(slice(x, y) for x, y in zip(delta_position_slice, [delta_position_slice_end_x, delta_position_slice_end_y, delta_position_slice_end_z]))

    # bring pad width into correct shape for np.pad function
    pad_width = np.transpose(np.asarray([delta_position, delta_position_end]))

    conversion_to_units = isinstance(resize_image, Quantity) and isinstance(padding_value, Quantity) and (padding_value.units.is_compatible_with(resize_image.units))

    values_resize_image = resize_image.magnitude if isinstance(resize_image, Quantity) else resize_image

    # slicing depends on number of dimensions
    if np.ndim(reference_image) in (2, 3):
        resized_image = np.pad(values_resize_image, pad_width, 'constant', constant_values=(padding_value.to(resize_image.units).magnitude
                                                                                            if conversion_to_units else padding_value,))[delta_position_slice_tuple]
    else:
        error_message = "Function currently only supports arrays with 2 or 3 dimensions."
        raise ValueError(error_message)

    # adapt memory layout of array (output was neither c_contiguous nor f_contiguous)
    resized_image = np.ascontiguousarray(resized_image)

    if isinstance(resize_image, Quantity):
        return Quantity(resized_image, resize_image.units)
    else:
        return resized_image


def crop_pad_aims(list_aims, index_image=0):
    """
    Function which resizes aim files to the original image shape after padding.

    Parameters
    ----------
    list_aims : list
        List of :class:`AIMFile` to import and resize
    index_image : int, optional
        Position of the reference image in the list. Default is 0.

    Returns
    -------
    resized_images : list
        Resized images which have same shape and position as the original image.
    """

    orig_shape = (list_aims[index_image]).data.shape

    reference_empty_image = np.zeros(orig_shape)

    # First pad to take into account the coordinates
    padded = pad_to_common_coordinate_system(list_aims)

    resized_images = []

    # Then crop and reshape to reference image
    for image in padded[0]:
        resized_images.append(crop_pad_image(reference_empty_image, image, delta_position=np.array(list_aims[index_image].position) - np.array(padded[1])))
    
    return resized_images


def get_full_slices(registered_slices, threshold=0.005):
    """
    Returns the lower and upper bound for registered stacks
    which only contain full bone slices (no half cut off etc.).

    When stack registering, some stacks might have minor rotations around
    the x- or y-axis. This means that the top and bottom most slices might
    actually only contain partial scanner output and void otherwise. This
    function gets rid of those slices so that at the end, only slices which
    cover the entire bone region are included.

    Parameters
    ----------
    registered_slices : bool nd_array
        Output from the radius ladder registration approach. Basically contains
        the rotated shapes (regular hexahedrals) of the initial stacks as a
        boolean mask.
    threshold : float
        The threshold to apply to the gradient of the image to find the
        "edge" where the full slice section starts / ends.

    Returns
    -------
    bounding_slices : slice
        A slice which can be applied to the corresponding density image z-dimension

    """
    if not np.any(registered_slices):
        warnings.warn("The registered slices given to get_full_slices are empty (all false). "
                      "Returned slice object will not alter its corresponding image.")

    max_fill = np.prod(registered_slices.shape[:2])
    filling = [np.sum(registered_slices[:, :, x])/max_fill for x in range(registered_slices.shape[2])]
    relevant_vals = [x > 0.5 for x in filling]
    lower_bound = np.argmax(relevant_vals)
    upper_bound = len(relevant_vals) - 1 - np.argmax(relevant_vals[::-1])
    vals = [float(x) for x in filling][lower_bound:upper_bound + 1]
    x_vals = list(range(lower_bound, upper_bound + 1))
    vals_g = np.gradient(vals)
    final_slices = np.logical_and(np.abs(vals_g) < threshold, np.abs(vals_g) > 0)
    lower_bound = np.argmax(final_slices)
    if lower_bound != 0:
        lower_bound -= 1
    upper_bound = len(final_slices) - 1 - np.argmax(final_slices[::-1])
    if upper_bound != (len(final_slices) - 1):
        upper_bound += 1
    lower_bound = x_vals[lower_bound]
    upper_bound = x_vals[upper_bound]

    return slice(lower_bound, upper_bound + 1)


def rescale_image(image_data, current_voxel_size, target_voxel_size, order=1, anti_aliasing=True, preserve_range=True, channel_axis=None):
    '''
    Function which rescale one image, from the current voxel size to a target voxel size given as arguments.

    Parameters
    ----------
    image_data : 2D- or 3D-array
        Image that we want to rescale. If the input image has units, the same unit will be returned with the output array
    current_voxel_size : Quantity
        Quantity with the value and unit of the current voxel size of the data
    target_voxel_size : Quantity
        Quantity with the value and unit of the target voxel size that we wish to rescale the image
    order : int
        Order of the interpolation function that we want ot use for the rescaling operation
    anti_aliasing : bool
        Whether to apply a Gaussian filter to smooth the image prior to down-scaling
        It is crucial to filter when down-sampling the image to avoid aliasing artifacts. (from Scikit.Image Doc)
    preserve_range : bool
        Whether to keep the range of values in the image.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds to channels.

    Returns
    -------
    rescaled_image : 2D- or 3D-array
        Resized imaged according to the settings passed as arguments

    Examples
    --------
    >>> a = np.ones((2,2));  a [0,1]=-1;  a[1,0]=-1; a
    array([[ 1., -1.],
       [-1.,  1.]])
    >>> rescale_image(a, current_voxel_size = Quantity(2, 'um'), target_voxel_size = Quantity(1,'um'), order = 0, anti_aliasing=True)
    array([[ 1.,  1., -1., -1.],
       [ 1.,  1., -1., -1.],
       [-1., -1.,  1.,  1.],
       [-1., -1.,  1.,  1.]])
    '''

    # Compute the scale factor to use in the rescaling
    # Ex: If the image has a VS = 2 um, and a target VS = 1 um, we need to double
    # the number of voxels to cover the same length.
    current_voxel_size = current_voxel_size.to('m')
    target_voxel_size = target_voxel_size.to('m')
    scale_factor = current_voxel_size.magnitude/target_voxel_size.magnitude

    if type(image_data) != np.ndarray:
        rescaled_image = rescale(image_data.magnitude, scale_factor, order=order, anti_aliasing=anti_aliasing, preserve_range=preserve_range, channel_axis=channel_axis)
        return Quantity(rescaled_image, image_data.units)
    else:
        rescaled_image = rescale(image_data, scale_factor, order=order, anti_aliasing=anti_aliasing, preserve_range=preserve_range, channel_axis=channel_axis)
        return rescaled_image


def pad_center_image(input_image, pad_margin=0, pad_value=0, target_side_size='max'):
    '''
    Function to pad an image to square dimensions while keeping the object centered.
    The side of the square corresponds to the largest dimension of the image, to which
    `pad_margin` can be additionally added.

    Parameters
    ----------
    input_image : np.ndarray or Quantity
        2D or 3D image to be padded.

    pad_margin : int
        Value of the additional margin to add around the image.

    pad_value : int
        Value to assign to the padded voxels.

    target_side_size : str or int
        Dimension of the side of the image. If 'max', the max shape value is used and
    the `pad_margin` can be additionally added. If int, the given value is used and the 
    `pad_margin` can be additionally added. If the int is lower than the maximum dimension,
    the size will default to 'max'.

    Returns:
    --------
    padded_image : np.ndarray or Quantity
        Input image padded with the specifications passed as argument.
    '''

    image = input_image
    flag_quantity = False
    if type(image) != np.ndarray:
        image_units = image.units
        flag_quantity = True
        image = input_image.magnitude

    image_ndim = len(image.shape)
    even_padding = np.insert(np.array(image.shape) % 2, np.arange(image_ndim), 0).reshape(image_ndim, 2)
    even_pad_image = np.pad(image, pad_width=even_padding, constant_values=pad_value)

    max_image_size = np.amax(even_pad_image.shape)
    target_side_size = max_image_size if target_side_size == 'max' else np.maximum(max_image_size, target_side_size)
    axis_pad = (np.repeat([np.absolute(even_pad_image.shape[i] - target_side_size) + 2*pad_margin for i in range(image_ndim)], 2)//2).reshape(image_ndim, 2)
    
    padded_image = np.pad(even_pad_image, pad_width=axis_pad, constant_values=pad_value)

    if flag_quantity:
        return Quantity(padded_image, image_units)
    else:
        return padded_image

