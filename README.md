# SAR_IMERG_Collocation

This work presents a framework for generating collocated Sentinel-1 SAR and GPM IMERG 
datasets to study the influence of rainfall on SAR backscatter over oceanic regions.

The developed pipeline integrates SAR preprocessing, IMERG extraction, spatial collocation, 
land masking, exploratory analysis, and visualization into a flexible workflow that can be 
applied to different Sentinel-1 scenes and weather events. 

The use of NetCDF and Parquet formats enabled efficient storage and processing of 
multidimensional SAR observations and large collocated datasets, while stride-based 
sampling and KDTree-based nearest-neighbor matching helped maintain computational 
efficiency. 

Additional analyses, including IMERG-SAR overlays, incidence angle 
dependence studies, and simple incidence angle correction using linear regression, 
provided further insight into the factors affecting SAR backscatter measurements. 

Preliminary investigations suggest that rainfall produces observable changes in Sentinel-1 
backscatter, particularly in the cross-polarized VH channel during heavy rain, and VV 
during moderate, although the relationship is influenced by factors such as incidence 
angle, ocean state, and the relatively coarse spatial resolution of IMERG. 

Nevertheless, the developed framework provides a useful foundation for further studies on rainfall 
characterization, larger multi-scene analyses, and the potential application of machine 
learning techniques for precipitation retrieval from SAR observations. 

Overall, this study demonstrates that combining high-resolution SAR observations with 
satellite-derived precipitation products is a promising approach for investigating rainfall 
signatures over marine environments and establishes a reusable framework that can be 
extended and refined for future research.
