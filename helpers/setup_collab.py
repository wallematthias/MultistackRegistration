import sys
import os
from google.colab import drive

def setup_colab(drive_directory, drive_name, working_directory):
    drive.mount(drive_directory)

    %load_ext autoreload
    %autoreload 2
    os.chdir(os.path.join(drive_directory, drive_name, working_directory))

    if not os.path.exists('multistack_registration'):
        !git clone https://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@github.com/wallematthias/multistack_registration.git
    else:
        os.chdir(os.path.join(drive_directory,drive_name,working_directory,'multistack_registration'))
        !git pull
        os.chdir(os.path.join(drive_directory,drive_name,working_directory))

    !{sys.executable} -m pip install itk-ioscanco xarray zarr tqdm pooch pint PyYAML SimpleITK pandas scikit-image matplotlib tabulate 'itkwidgets[all]>=1.0a21' > /dev/null