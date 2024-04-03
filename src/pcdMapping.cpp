#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

// ------------------------------------------------------------------
// parameters
const double cx = 319.5;
const double cy = 239.5;
const double fx = 481.2;
const double fy = -480.0;
const double depthScale = 5000.0;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
// ------------------------------------------------------------------

bool loadData(const std::string& path, std::vector<cv::Mat>& colorImgs, std::vector<cv::Mat>& depthImgs, std::vector<Eigen::Isometry3d>& poses) {

    std::ifstream fin(path + "/pose.txt");
    if (!fin)
        return false;

    while (!fin.eof()) {
        double data[7];     //Data format: tx, ty, tz, qx, qy, qz, qw
        for (double& d : data) {
            fin >> d;
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.emplace_back(T);
        if (!fin.good())
            break;
    }

    fin.close();

    for (int i = 0; i < 5;i++) {
        std::string colorImgPath = path + "/color/" + std::to_string(i + 1) + ".png";
        std::string depthImgPath = path + "/depth/" + std::to_string(i + 1) + ".png";
        colorImgs.emplace_back(cv::imread(colorImgPath));
        depthImgs.emplace_back(cv::imread(depthImgPath, -1));
    }

    if (colorImgs[0].data == nullptr || depthImgs[0].data == nullptr) {
        return false;
    }

    if (colorImgs.size() != depthImgs.size()) {
        return false;
    }

    return true;
}

bool cleanPCD(PointCloud::Ptr pointCloud) {
    PointCloud::Ptr temp(new PointCloud);
    pcl::StatisticalOutlierRemoval<PointT> statisticalFilter;
    statisticalFilter.setMeanK(50);
    statisticalFilter.setStddevMulThresh(1.0);
    statisticalFilter.setInputCloud(pointCloud);
    statisticalFilter.filter(*temp);
    temp->swap(*pointCloud);
    return true;
}

bool applyVoxelDownsampling(PointCloud::Ptr pointCloud) {
    pointCloud->is_dense = false;
    PointCloud::Ptr temp(new PointCloud);
    pcl::VoxelGrid<PointT> voxelFilter;
    double resolution = 0.03;
    voxelFilter.setLeafSize(resolution, resolution, resolution);
    voxelFilter.setInputCloud(pointCloud);
    voxelFilter.filter(*temp);
    temp->swap(*pointCloud);
    return true;
}

bool alignColorDepth(std::vector<cv::Mat>& colorImgs, std::vector<cv::Mat>& depthImgs, std::vector<Eigen::Isometry3d>& poses, PointCloud::Ptr pointCloud) {
    for (int i = 0; i < colorImgs.size();i++) {
        PointCloud::Ptr current(new PointCloud);
        std::cout << "image: " << i + 1 << std::endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned short d = depth.ptr<unsigned short>(v)[u];
                if (d == 0)
                    continue;
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[1] = (v - cy) * point[2] / fy;
                point[0] = (u - cx) * point[2] / fx;

                Eigen::Vector3d pointWorld = T * point;

                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];
                pointCloud->points.emplace_back(p);
            }
    }
    return true;
}

int main() {
    std::vector<cv::Mat> colorImgs, depthImgs;
    std::vector<Eigen::Isometry3d> poses;
    const std::string path = "../data";
    PointCloud::Ptr pointCloud(new PointCloud);

    bool ret = loadData(path, colorImgs, depthImgs, poses);
    if (ret == false) {
        std::cout << "loading data failed, check paths!" << std::endl;
        return -1;
    }
    std::cout << "total images : " << colorImgs.size() << std::endl;

    ret = alignColorDepth(colorImgs, depthImgs, poses, pointCloud);
    if (ret == false) {
        std::cout << "alignColorDepth failed" << std::endl;
        return -1;
    }

    cleanPCD(pointCloud);
    applyVoxelDownsampling(pointCloud);

    pcl::io::savePLYFile("../output/final.ply", *pointCloud);

    std::cout << "Done!" << std::endl;

    return 0;
}