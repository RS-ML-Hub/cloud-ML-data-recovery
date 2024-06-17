import os
import numpy as np
import netCDF4
import matplotlib.pyplot as plt

path = os.path.join('/home/shared/Data/OLCI/L2','S3A_OL_2_WFR____20160425T151510_20160425T151710_20210703T072556_0119_003_239______MAR_R_NT_003.SEN3','Oa01_reflectance.nc' )
Dataset = netCDF4.Dataset(path)

#TODO Actually load cloudless data and apply cloud masks to them