##Find Single subject maps of seed-to-voxel correlation
##Seed region is a sphere of radius 8 (in mm) located in the Posterior Cingulate Cortex.
##PCC is considered to be part of the Default Mode Network.
from nilearn.maskers import NiftiSpheresMasker
from nilearn import image
from nilearn.maskers import NiftiMasker
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt



# Sphere radius in mm
sphere_radius = 8

# Sphere center in MNI-coordinate
sphere_coords = [(0, -52, 18)]

data= image.load_img('filtered_func_data.nii.gz')
seed_masker = NiftiSpheresMasker(sphere_coords, radius=sphere_radius, detrend=True,
standardize=True, verbose=1, memory="nilearn_cache", memory_level=2)

# extract the mean time series within the seed region
seed_time_series = seed_masker.fit_transform(data)

#extract the mean time series in each voxel
#smoothing the signal with a smoothing kernel of 6mm.
brain_masker = NiftiMasker(smoothing_fwhm=6, detrend=True, standardize=True, verbose=1, memory="nilearn_cache", memory_level=2)
brain_time_series = brain_masker.fit_transform(data)

#Now that we have two arrays (mean signal in seed region, signal for each voxel), we can correlate the two to each other. This can be done with the dot product between the two matrices.
seed_based_correlations = np.dot(brain_time_series.T, seed_time_series)
seed_based_correlations /= seed_time_series.shape[0]
#tranform the correlation array back to a Nifti image.
seed_based_correlation_img = brain_masker.inverse_transform(seed_based_correlations.T)
#plot
display = plotting.plot_stat_map(seed_based_correlation_img, threshold=0.333, cut_coords=sphere_coords[0])
display.add_markers(marker_coords=sphere_coords, marker_color='black', marker_size=200)
plotting.show()
