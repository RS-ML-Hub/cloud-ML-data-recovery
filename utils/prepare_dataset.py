import os
import numpy as np
import xarray as xr
import xbatcher as xb

clouds_path = os.path.join('/home/shared/Data/cloud_masks')
sen3_path = os.path.join('/home/shared/Data/OLCI/GoM/')

def filter_masks_by_percentage(clouds_path, low_bound=0, high_bound=25):
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
    return files[1:101]

def fetch_cloud_masks(file_list):
    """
        This function fetches the cloud masks from the directory.
        It loads the masks as numpy arrays and concatenates them into a tensor.

        Args:
        file_list: a list of files to be loaded

        Returns:
        tensor: a tensor containing all the masks in the list of size 256x256xnb_clouds
    """
    tens = np.load(os.path.join(clouds_path,file_list[0]))[:,:,np.newaxis]
    for f in file_list[1:]:
        tens = np.concatenate([tens,np.load(os.path.join(clouds_path,f))[:,:,np.newaxis]],axis=2)
    return tens

def list_sen3(path):
    """
        This function fetches the sen3 files from the directory.
        It looks into all the .SEN3 folders and returns the paths to the ones where the matrix of cloud batches is empty (no clouds).

        Args:
        path: the path to the directory containing the .SEN3 folders

        Returns:
        files: a list of paths to the .SEN3 folders where the matrix of cloud batches is empty

    """
    batch_dirs_list = []
    for dirs in os.listdir(path):
        for dir in os.listdir(os.path.join(path,dirs)):
            if dir.endswith('.SEN3'):
                for subdir in os.listdir(os.path.join(path,dirs,dir)):
                    if subdir == "no_neg_256":
                        for batch_dir in os.listdir(os.path.join(path,dirs,dir,subdir)):
                            batch_dirs_list.append(os.path.join(path,dirs,dir, subdir, batch_dir))    
    print("There are {} batches with no clouds".format(len(batch_dirs_list)))
    return batch_dirs_list

def fetch_sen3(file_list):
    #Load all wanted nc bands as one 3D array then concatenate all of them in a tensor
    ds = open_olci(file_list[0])
    for batch in file_list[1:]:
        ds = xr.concat((ds, open_olci(batch)), dim='batch')
    return ds

def mask_image(sen3_file, cloud_file):
    for i in range(1,13):
        sen3_file["Oa%s_reflectance" % str(i).zfill(2)] = sen3_file["Oa%s_reflectance" % str(i).zfill(2)] * cloud_file

def open_olci(path):
    arr = []
    for i in range(1, 13):
        x = xr.open_dataset(os.path.join(path, "oa%s_reflectance.nc" % str(i).zfill(2)))
        arr.append(x)
    return xr.merge(arr)


clouds = fetch_cloud_masks(filter_masks_by_percentage(clouds_path))
ds = fetch_sen3([list_sen3(sen3_path)[0]])
ds