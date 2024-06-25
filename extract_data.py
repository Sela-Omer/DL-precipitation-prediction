import tarfile
import os

from tqdm import tqdm


def extract_selected_years(archive_path, output_dir, selected_years):
    """
    Extracts specific years from a tar.gz archive, maintaining the original directory structure and filenames.

    Args:
    - archive_path (str): Path to the tar.gz archive.
    - output_dir (str): Directory to extract the files to.
    - selected_years (list of str): List of years to extract.
    """
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in tqdm(archive):
            # Check if the member is a file (not a directory)
            if member.isfile():
                # Extract the year from the file path
                for year in selected_years:
                    if year in member.name:
                        archive.extract(member, path=output_dir)
                        break


# Example usage
archive_path = 'G:/precip_prediction_data/tensors.tar.gz'
output_dir = 'G:/precip_prediction_data/tensors'
selected_years = ['2000', '2001', '2002']

extract_selected_years(archive_path, output_dir, selected_years)
