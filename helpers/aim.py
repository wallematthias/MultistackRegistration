import itk
import numpy as np
from pint import Quantity
import re

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


def load_aim(filepath: str) -> AIMFile:
    """
    Load an AIM file and convert it into a standardized format.

    Args:
        filepath (str): The path to the AIM file that needs to be loaded.

    Returns:
        AIMFile: An object that contains the image data in a standardized format.

    Raises:
        Exception: If the file cannot be loaded.

    """
  
    try:
        # Try to read the file as an image file
        image = itk.imread(filepath)

        # Convert the image to a NumPy array and transpose it to change the axis order
        arr = np.transpose(np.asarray(image), (1, 2, 0))

        # Get the calibration constants from the processing log and use them to convert the Hounsfield units to density values
        mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = get_aim_calibration_constants_from_processing_log(filepath)
        density = convert_hounsfield_to_mgccm(arr, mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept)

        # Convert the density values to a Quantity object with units of mg/cm^3
        data= Quantity(density, 'mg/cm**3')

        # Create a dictionary containing the processing log and add the density slope and intercept to it
        processing_log = dict(image)
        processing_log['density_slope'] = density_slope
        processing_log['density_intercept'] = density_intercept

    except:
        # If the file cannot be read as an image file, assume it is a binary mask file
        print('Reading mask Data')
        image = itk.imread(filepath, itk.UC)
        
        # Convert the mask data to a Quantity object with units of 'dimensionless'
        data= Quantity(np.transpose(np.asarray(image) > 0, (1, 2, 0)).astype(int),'dimensionless')
        
        # Create a dictionary containing the processing log
        processing_log = dict(image)

    # Extract the voxel size and position from the processing log and create a Quantity object for the voxel size
    voxelsize = Quantity(processing_log['spacing'], 'mm')
    position = np.round(processing_log['origin'] / processing_log['spacing']).astype(int)

    # Return an AIMFile object with the data, processing log, voxel size, and position
    return AIMFile(data, processing_log, voxelsize, position)


def write_aim(aim_file: AIMFile, file_path: str) -> None:
    """
    Write an AIM file from an AIMFile object.

    Args:
        aim_file (AIMFile): An object that contains the image data in a standardized format.
        file_path (str): The path where the AIM file needs to be saved.

    Returns:
        None

    """
  
    # Convert the data in the AIMFile object to an itk image
    image = itk.GetImageFromArray(np.asarray(aim_file.data).astype(float))

    # Create a new itk.MetaDataDictionary object
    itk_metadata_dict = itk.MetaDataDictionary()

    # Iterate through the dictionary items in the AIMFile processing log and set them on the itk_metadata_dict
    for key, value in aim_file.processing_log.items():
        # Convert the value to a string if it is not already a string
        if not isinstance(value, str):
            value = str(value)
        # If the value is empty, replace it with None
        if not value:
            value = None

        itk_metadata_dict[key] = value

    # Set the itk_metadata_dict on the itk image
    image.SetMetaDataDictionary(itk_metadata_dict)

    # Write the itk image to a file with the extension '.mha'
    file_path = file_path.replace('.aim','.AIM')
    itk.imwrite(image, file_path.split('.AIM')[0]+'.mha')


def get_aim_calibration_constants_from_processing_log(filename):
    '''Get the calibration constants from a AIM processing log'''

    with open(filename, 'rb') as file:
        processing_log = str(file.read(3072))

    mu_scaling_match = re.search(r'Mu_Scaling\s+(\d+)', processing_log)
    hu_mu_water_match = re.search(r'HU: mu water\s+(\d+.\d+)', processing_log)
    density_slope_match = re.search(r'Density: slope\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)', processing_log)
    density_intercept_match = re.search(r'Density: intercept\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)', processing_log)

    mu_scaling = int(mu_scaling_match.group(1))
    hu_mu_water = float(hu_mu_water_match.group(1))
    hu_mu_air = 0
    density_slope = float(density_slope_match.group(1))
    density_intercept = float(density_intercept_match.group(1))

    return mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept


def convert_hounsfield_to_mgccm(hounsfield, mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept):
    """
    Converts a value in Hounsfield units to milligrams per cubic centimeter (mg/cc).

    Args:
        hounsfield (float): The value to be converted, in Hounsfield units (HU).
        mu_scaling (float): The scaling factor used to convert from native units to linear attenuation coefficients (LAC).
        hu_mu_water (float): The HU value of water, used to calculate the slope for the conversion from HU to LAC.
        hu_mu_air (float): The HU value of air, used to calculate the slope for the conversion from HU to LAC.
        density_slope (float): The slope of the linear equation used to convert from LAC to density in mg/cc.
        density_intercept (float): The intercept of the linear equation used to convert from LAC to density in mg/cc.

    Returns:
        float: The value converted to mg/cc.
    """
    
    # do some conversions
    
    slope_native_to_hounsfield = 1000.0 / (mu_scaling * (hu_mu_water - hu_mu_air))
    intercept_native_to_hounsfield = -1000.0 * hu_mu_water / (hu_mu_water - hu_mu_air)
    slope_native_to_density = density_slope / mu_scaling
    intercept_native_to_density = density_intercept
    slope_hounsfield_to_density = slope_native_to_density / slope_native_to_hounsfield
    intercept_hounsfield_to_density = intercept_native_to_density - slope_native_to_density * intercept_native_to_hounsfield / slope_native_to_hounsfield
    
    density = slope_hounsfield_to_density * hounsfield + intercept_hounsfield_to_density
    
    return density


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
