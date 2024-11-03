##We have preprocessed fMRI data (time series corresponding to 64 times 64 times 33 voxels) and a list of changepoints in time
##We want to create brain networks corresponding to time segment segmented by changepoints

##We will define region of interests in brain (corresponds to node in brain networks)
##msdl atlas(39ROIs) is used to define region of interests (ROIs)
##Nodes are connected if the correlation of corresponding brian regions is over 1/2

##This program outputs adjacency matrices (indicating connectivity networks) and correlation matrices as text files

from nilearn import datasets
from nilearn import image
from nilearn.maskers import NiftiMapsMasker
import numpy as np
from nilearn.connectome import ConnectivityMeasure
import numpy as np

#Atlas
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]

correlation_measure = ConnectivityMeasure(kind="correlation")

#"change_points.txt" is a list of change point is 
filename = "change_points.txt"
#first change point for each subject
cp_col1 = np.loadtxt(filename, dtype=int, usecols=range(1,2))
#cp_col1 is NumPy array containing the first changepoints for all the subjects

#second change point for each subject
cp_col2 = np.loadtxt(filename, dtype=int, usecols=range(2,3))


def cut_ts(c1,c2):
    #take time series from time=c1 to time=c2
    #set c1=3, c2=first changepoint-1
    #set c1=first changepoint, c2=second changepoint-1
    #set c1=second changepoint, c2=225
    ts=np.zeros((c2-c1+1,39))
    #time series has 39 regions of interests
    for i in range(c2-c1+1):
        ts[i]=time_series[i+c1-1]
        #ith row of ts is replaced by i+c1-1th row if time_series
    return ts

import os

#'reg_funcs' stores preprocessed fMRI data
#output matrices (adjacent and correlation) are stored in a folder 'matrix'
idx=-1
folder_path = 'reg_funcs'
for filename in os.listdir(folder_path):
    idx+=1
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        image_path = os.path.join(folder_path, filename)
        # Load the fMRI data
        data = image.load_img(image_path)
        #take average time seeries at each ROI
        masker = NiftiMapsMasker(
            maps_img=atlas_filename,
            standardize="zscore_sample",
            memory_level=1,
            memory="nilearn_cache",
            verbose=5,
            detrend=True
        )
        masker.fit(data)
        time_series = masker.transform(data)

        #time_series[0] has values at time 1 for all 39 regions
        if 3<cp_col2[idx]<225:
            #if subject has 2 changepoints, get 3 graphs
            cut_time_series=cut_ts(3,cp_col1[idx]-1)
            corr_mat1 = correlation_measure.fit_transform([cut_time_series])[0]
            adj_mat = np.where(corr_mat1<0.5, 0, 1)
            adj_mat1 = adj_mat - np.identity(39)
            np.savetxt(f"./matrix/corr_1_{idx}_cp.txt", corr_mat1)
            np.savetxt(f"./matrix/adj_1_{idx}_cp.txt", adj_mat1)

            cut_time_series=cut_ts(cp_col1[idx],cp_col2[idx]-1)
            corr_mat2 = correlation_measure.fit_transform([cut_time_series])[0]
            adj_mat = np.where(corr_mat2<0.5, 0, 1)
            adj_mat2 = adj_mat - np.identity(39)
            np.savetxt(f"./matrix/corr_2_{idx}_cp.txt", corr_mat2)
            np.savetxt(f"./matrix/adj_2_{idx}_cp.txt", adj_mat2)

            cut_time_series=cut_ts(cp_col2[idx],225)
            #data has time 1 to time 225
            corr_mat3 = correlation_measure.fit_transform([cut_time_series])[0]
            adj_mat = np.where(corr_mat3<0.5, 0, 1)
            adj_mat3 = adj_mat - np.identity(39)
            np.savetxt(f"./matrix/corr_3_{idx}_cp.txt", corr_mat3)
            np.savetxt(f"./matrix/adj_3_{idx}_cp.txt", adj_mat3)

        elif cp_col2[idx]==0:
            #if subject has no change point, get 1 graph
            cut_time_series=cut_ts(3,225)
            #data has time 1 to 225
            corr_mat4 = correlation_measure.fit_transform([cut_time_series])[0]
            adj_mat = np.where(corr_mat4<0.5, 0, 1)
            adj_mat4 = adj_mat - np.identity(39)
            np.savetxt(f"./matrix/corr_0_{idx}_no.txt", corr_mat4)
            np.savetxt(f"./matrix/adj_0_{idx}_no.txt", adj_mat4)



        

    



        








