import itk
import numpy as np
from pint import Quantity

class AIMFile:
    '''
    Python representation of an AIM file. This class
    stores all attributes that can be read and written from
    an AIM file.
    '''

    def __init__(self, data, processing_log, voxelsize, position=None):
        '''
        Constructor that creates an AIM file object

        :param data: Raw image data
        :type data: 3D numpy :class:`array <numpy.ndarray>`
        :param processing_log: Processing log found in AIM files
        :type processing_log: str
        :param voxelsize: The voxelsize in each dimension
        :type voxelsize: A :class:`list` or :class:`tuple`
                         or :class:`numpy.ndarray` of length 1 or 3
        :param position: The position of the image (default is (0,0,0))
        :type position: A :class:`list` or :class:`tuple` of length 3
        '''
        self.data = data
        self.processing_log = processing_log
        self.voxelsize = voxelsize
        self.position = (0, 0, 0) if position is None else position


def load_aim(filepath):
  
    image = itk.imread(filepath)
    arr = np.transpose(np.asarray(image), (1, 2, 0))

    data= Quantity(arr,'mg/cm**3')
    processing_log= dict(image)

    voxelsize= Quantity(processing_log['spacing'],'mm')
    position= np.round(processing_log['origin']/processing_log['spacing']).astype(int)

    return AIMFile(data, processing_log, voxelsize, position)

def write_aim(aim_file, file_path):
  
    image = itk.GetImageFromArray(aim_file.data)

    # Create a new itk.MetaDataDictionary object
    itk_metadata_dict = itk.MetaDataDictionary()

    # Iterate through the dictionary items and set them on the itk_metadata_dict
    for key, value in aim_file.processing_log.items():
    # Convert the value to a string if it is not already a string
        if not isinstance(value, str):
            value = str(value)
        itk_metadata_dict[key] = value


    itk.imwrite(image, file_path.replace('.AIM','.mha'))
  
    return 1


def pad_to_common_coordinate_system(aim_files, padding_values=None):
    '''
    Takes a list of AIMFile objects and returns their data arrays padding each array
    as necessary to make them all have the same size and reside in the same coordinate
    system.

    :param aim_files: AIM files to convert
    :type aim_files: tuple or list of :any:`AIMFile` objects
    :param padding_values: (optionally) padding values to use with each AIM file.
                           One padding value must be given for each array.
    :type padding_values: list or tuple of appropriate :any:`ifb_framework.Quantity`
                          and the new position of the new coordinate system

    :raises: ValueError if the number of given padding_values does not match the number
             of given AIM files or if the aim files have different voxel sizes
    '''
    if padding_values is None:
        padding_values = [Quantity(0, aim_file.data.units)
                          for aim_file in aim_files]

    if len(padding_values) != len(aim_files):
        error_message = (
            'There must be as many padding values as there are AIM-files. But: ' +
            'No. of AIM-files: {}, no. of padding values: {}'.format(
                len(aim_files),
                len(padding_values)))
        raise ValueError(error_message)

    for aim_file1, aim_file2 in zip(aim_files[:-1], aim_files[1:]):
        voxelsizes1 = aim_file1.voxelsize
        voxelsizes2 = aim_file2.voxelsize.to(aim_file1.voxelsize.units)
        if not np.allclose(voxelsizes1.magnitude,
                           voxelsizes2.magnitude, rtol=1e-3):
            error_message = (
                'Found different voxel-sizes: {} and {}'.format(voxelsizes1, voxelsizes2))
            raise ValueError(error_message)

    # Remove units as padding does not support units
    _padding_values = [
        Quantity(val).to(
            aim_file.data.units).magnitude for val, aim_file in zip(
            padding_values, aim_files)]

    # Find out the cube which fully contains all images
    min_coordinate_corner = None
    max_coordinate_corner = None

    for aim_file in aim_files:
        if min_coordinate_corner is None:
            min_coordinate_corner = np.array(aim_file.position)
        else:
            min_coordinate_corner = np.minimum(
                min_coordinate_corner, aim_file.position)

        current_file_max_dimensions = np.array(
            aim_file.position) + aim_file.data.shape

        if max_coordinate_corner is None:
            max_coordinate_corner = current_file_max_dimensions
        else:
            max_coordinate_corner = np.maximum(
                max_coordinate_corner, current_file_max_dimensions)


    # Pad and collect new arrays and return them
    return_data = []

    for aim_file, padding_value in zip(aim_files, _padding_values):
        new_arr = np.pad(
            aim_file.data,
            list(zip(
                aim_file.position - min_coordinate_corner,
                max_coordinate_corner - aim_file.position - aim_file.data.shape)),
            'constant',
            constant_values=((padding_value,) * 2,) * 3)
        return_data.append(Quantity(new_arr, aim_file.data.units))

    return return_data, min_coordinate_corner
