import os
import numpy as np


def create_mock_data(base_dir, parameters, years, num_files_per_year, times, height, width):
    """
    Creates a mock directory structure with random data for testing.

    Args:
    - base_dir (str): Base directory to create the mock data.
    - parameters (list of str): List of parameter names.
    - years (list of str): List of years.
    - num_files_per_year (int): Number of files to create per year.
    - times (int): Number of time steps.
    - height (int): Height of the spatial data.
    - width (int): Width of the spatial data.
    """
    for param in parameters:
        param_parts = param.split('_')
        if len(param_parts) > 1:
            param_dir = os.path.join(base_dir, param_parts[0], param_parts[1])
        else:
            param_dir = os.path.join(base_dir, param)

        for year in years:
            year_dir = os.path.join(param_dir, str(year))
            os.makedirs(year_dir, exist_ok=True)

            for i in range(num_files_per_year):
                filename = f"data_{year}_{i:03d}.npy"
                file_path = os.path.join(year_dir, filename)

                data = np.random.rand(times, height, width)

                np.save(file_path, data)


# Example usage
base_dir = 'mock_data'
parameters = ['date', 'lat', 'lon', 'sp', 't2m', 'intensity', 'u10', 'v10', 'tcw', 'z',
              'ta_250', 'ta_300', 'ta_500', 'ta_850',
              'ua_250', 'ua_300', 'ua_500', 'ua_850',
              'va_250', 'va_300', 'va_500', 'va_850',
              'za_250', 'za_300', 'za_500', 'za_850']
years = ['2021', '2022', '2023']
num_files_per_year = 10
times = 24
height = 32
width = 32

create_mock_data(base_dir, parameters, years, num_files_per_year, times, height, width)
