import os
import numpy as np

clouds_path = os.path.join('/home/shared/Data/cloud_masks')
sen3_path = os.path.join('/home/shared/Data/OLCI/GoM/')

def filter_masks_by_percentage(low_bound=0, high_bound=25):
    """
        This function filters cloud masks by the percentage of cloud cover.
        It returns a list of files that are within the boundaries.
        
        Note: the masks must be in the format *_PERCENT{percentage}.npy
        
        Args:
            low_bound: the lower bound of the percentage of cloud cover
            high_bound: the upper bound of the percentage of cloud cover
        Returns:
            files: a list of files that are within the boundaries
    """
    files = os.listdir(clouds_path)
    print("There are {} cloud masks in the directory".format(len(files)))
    #Get all numpy files
    files = [f for f in files if f.endswith('.npy')]
    #Map files where the percent is above the lower bound
    files = [f for f in files if int(f.split('PERCENT')[1].split('.')[0]) >= low_bound]
    #Map files where the percent is below the upper bound
    files = [f for f in files if int(f.split('PERCENT')[1].split('.')[0]) <= high_bound]
    #Both operations could be condensed into one line but for readability they are separated

    print("There are {} cloud masks between {} and {} percent of coverage".format(len(files), low_bound, high_bound))
    return files

def fetch_sen3(path):
    """
        This function fetches the sen3 files from the directory.
        It looks into all the .SEN3 folders and returns the paths to the ones where the matrix of cloud batches is empty (no clouds).

        Args:
        path: the path to the directory containing the .SEN3 folders

        Returns:
        files: a list of paths to the .SEN3 folders where the matrix of cloud batches is empty

    """
    files = []
    for dirs in os.listdir(path):
        for name in os.listdir(os.path.join(path,dirs)):
            if name.endswith('.SEN3'):
                if(np.load(os.path.join(path,dirs, name, 'cloud_batch_coords_256.npy')).size == 0):
                    files.append(os.path.join(path, name))
    print("There are {} .SEN3 folders with no clouds".format(len(files)))
    return files


