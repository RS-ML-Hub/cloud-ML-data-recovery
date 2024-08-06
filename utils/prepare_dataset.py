import os
import numpy as np
import xarray as xr
import shutil
import pandas
import random

def filter_masks_by_percentage(clouds_path, low_bound=0, high_bound=25, num_clouds=100, type_mask="scatter", direction="/home/etienne/cloud-ML-data-recovery/data/masks/"):
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
    cloud_df = pandas.read_csv("/home/shared/Data/cloud_masks_meta.csv")
    print("There are {} cloud masks in the directory".format(len(files)))
    filtered_clouds = cloud_df[(cloud_df["cloud_class"] == type_mask)]
    percentage = filtered_clouds.apply(lambda x: 100*x["missing_pixels"]/(256*256), axis=1)
    filtered_clouds = filtered_clouds[(percentage >= low_bound) & (percentage <= high_bound)]

    samples = filtered_clouds.sample(num_clouds)["filename"].values

    #copy file from each samples into new dir

    for sample in samples:
        source_path = os.path.join(clouds_path, sample)
        destination_path = os.path.join(direction, sample)
    
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
    return

def fetch_cloud_masks(clouds_path, file_list):
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
        ds = xr.concat((ds, open_olci(batch)), dim='samples')
    return ds

def mask_images(sen3_files, cloud_files):
    cloud_index = np.random.randint(0, cloud_files.shape[2], sen3_files["samples"].shape[0])
    sen_cloudy = xr.zeros_like(sen3_files)
    sen_cloudy = sen_cloudy.assign(mask = xr.zeros_like(sen3_files["Oa01_reflectance"]))
    for b in range(0, sen3_files["samples"].shape[0]):
        for i in range(1,12):
            #normalize then apply mask
            sen3_files["Oa%s_reflectance" % str(i).zfill(2)][b] = (np.log(sen3_files["Oa%s_reflectance" % str(i).zfill(2)][b]) - np.min(np.log(sen3_files["Oa%s_reflectance" % str(i).zfill(2)][b])))/(np.max(np.log(sen3_files["Oa%s_reflectance" % str(i).zfill(2)][b])) - np.min(np.log(sen3_files["Oa%s_reflectance" % str(i).zfill(2)][b])))
            sen_cloudy["Oa%s_reflectance" % str(i).zfill(2)][b] = sen3_files["Oa%s_reflectance" % str(i).zfill(2)][b] *(1- cloud_files[:,:,cloud_index[b]])
        sen_cloudy["mask"][b] = cloud_files[:,:,cloud_index[b]]
    return sen_cloudy        

def open_olci(path):
    arr = []
    for i in range(1, 12):
        x = xr.open_dataset(os.path.join(path, "oa%s_reflectance.nc" % str(i).zfill(2)))
        arr.append(x)
    return xr.merge(arr)

def prepare_dataset(sen3_path_str='/home/shared/Data/OLCI/GoM/' , cloud_path_str='/home/shared/Data/cloud_masks/', direction="./data/", low_bound=0, high_bound=25, num_samples=100, num_clouds=100, type_mask="scatter"):
    sen3_path = os.path.join(sen3_path_str)
    #clouds_path = os.path.join(cloud_path_str)
    #filter_masks_by_percentage(clouds_path, low_bound, high_bound, num_clouds)
    #clouds = fetch_cloud_masks(clouds_path, clouds_list)
    #print("Cloud masks loaded")
    sen3_list = list_sen3(sen3_path)
    sen3_list_reduced = random.sample(sen3_list, num_samples)
    print("SEN3 files selected")
    ds = fetch_sen3(sen3_list_reduced)
    #print("SEN3 files loaded")
    #print("Masking images...")
    #dsc = mask_images(ds, clouds,freq_files)
    #print("Images masked")
    return ds

def main():
    filter_masks_by_percentage("/home/shared/Data/cloud_masks/", low_bound=15, high_bound=25, num_clouds=1000, type_mask="scatter")


if __name__ == "__main__":
    main()