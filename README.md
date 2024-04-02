# RGBD Reconstruction

This repository explores various RGBD reconstruction and mapping techniques

1. Converting the RGB-D data into a point cloud based on the estimated camera pose and then stitching them into a global point cloud map composed of discrete points
2. Use the triangular mesh and the surface (surfel) to build the map to estimate the object’s surface
3. Create an occupancy map through voxels, to know the map’s obstacle information and navigate the map.
