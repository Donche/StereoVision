StereoVision V4

**Updated**

* OpenGL has been added to represent point cloud, but there's something wrong with it and it don't work very well. Need to check out what's going wrong with the post processing of the disparity map
* Add real-time parameter adjustment window

# TODOs
* Optimization of 3D reconstruction
* Enhanced robustness of BM matching 
* Enhance the number and accuracy of feature detection, extraction and matching

# DONE
* Calibration of binocular camera
* Calculate of disparity map of BM and SGBM
* feature detection, extraction and matching of SURF and ORB
    * use KNN match and ratio test to get the best result.
    * Symmetry verification
    * Calculate essential matrix and use this to derive matrix R and T

# Final Goal
* More stable stereo vision especially when the relative positions of the two cameras are unstable
