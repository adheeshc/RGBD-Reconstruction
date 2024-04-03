#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/surfel_smoothing.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/impl/mls.hpp>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointXYZRGBNormal SurfelT;
typedef pcl::PointCloud<SurfelT> SurfelCloud;
typedef pcl::PointCloud<SurfelT>::Ptr SurfelCloudPtr;

void reconstructSurface(const PointCloudPtr& input, const SurfelCloudPtr& output, const float radius, const int polynomialOrder) {
    pcl::MovingLeastSquares<PointT, SurfelT> mls;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(radius);
    mls.setComputeNormals(true);
    mls.setSqrGaussParam(radius * radius);
    // mls.setPolynomialFit(polynomialOrder > 1);
    mls.setPolynomialOrder(polynomialOrder);
    mls.setInputCloud(input);
    mls.process(*output);
}

void triangulateMesh(const SurfelCloudPtr& surfels, pcl::PolygonMeshPtr triangles) {
    // Create search tree
    pcl::search::KdTree<SurfelT>::Ptr tree(new pcl::search::KdTree<SurfelT>);
    tree->setInputCloud(surfels);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<SurfelT> gp3;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius(0.05);

    // Set typical values for the parameters
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
    gp3.setMinimumAngle(M_PI / 18); // 10 degrees
    gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
    gp3.setNormalConsistency(true);

    // Get result
    gp3.setInputCloud(surfels);
    gp3.setSearchMethod(tree);
    gp3.reconstruct(*triangles);
}

int main() {

    PointCloudPtr inputCloud(new PointCloud);
    const std::string path = "../output/pcdMapping_final.ply";
    pcl::io::loadPLYFile(path, *inputCloud);

    if (inputCloud->points.size() == 0) {
        std::cout << "failed to load point cloud!" << std::endl;
        return -1;
    }
    //compute surface normals
    double mlsRadius = 0.05;
    double polynomialOrder = 2;
    SurfelCloudPtr surfelCloud(new SurfelCloud);
    pcl::PolygonMeshPtr triangleMesh(new pcl::PolygonMesh);
    reconstructSurface(inputCloud, surfelCloud, mlsRadius, polynomialOrder);
    triangulateMesh(surfelCloud, triangleMesh);

    // pcl::io::savePLYFile("../output/surfelCloud.ply", *surfelCloud);
    pcl::io::savePLYFile("../output/mesh.ply", *triangleMesh);

    std::cout << "Done!./" << std::endl;
    return 0;
}