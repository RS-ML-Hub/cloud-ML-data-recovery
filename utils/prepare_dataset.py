import os

clouds_path = os.path.join('/home/shared/Data/cloud_masks')

def filter_masks_by_percentage(low_bound=0, high_bound=25):
    files = os.listdir(clouds_path)
    print("There are {} cloud masks in the directory".format(len(files)))
    #Get all numpy files
    files = [f for f in files if f.endswith('.npy')]
    #Map files where the percent is above the lower bound
    files = [f for f in files if int(f.split('PERCENT')[1].split('.')[0]) >= low_bound]
    #Map files where the percent is below the upper bound
    files = [f for f in files if int(f.split('PERCENT')[1].split('.')[0]) <= high_bound]
    print("There are {} cloud masks between {} and {} percent".format(len(files), low_bound, high_bound))
    return files


