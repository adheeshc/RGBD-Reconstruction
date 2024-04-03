#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <octomap/octomap.h>

// ------------------------------------------------------------------
// parameters
const double cx = 319.5;
const double cy = 239.5;
const double fx = 481.2;
const double fy = -480.0;
const double depthScale = 5000.0;
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

void populateOctTree(std::vector<cv::Mat>& colorImgs, std::vector<cv::Mat>& depthImgs, std::vector<Eigen::Isometry3d>& poses, octomap::OcTree& tree, octomap::Pointcloud& cloud) {
    for (int i = 0; i < colorImgs.size();i++) {
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
                cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
            }
        tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3)));
    }
}

int main() {
    std::vector<cv::Mat> colorImgs, depthImgs;
    std::vector<Eigen::Isometry3d> poses;
    const std::string path = "../data";

    bool ret = loadData(path, colorImgs, depthImgs, poses);
    if (ret == false) {
        std::cout << "loading data failed, check paths!" << std::endl;
        return -1;
    }
    std::cout << "total images : " << colorImgs.size() << std::endl;

    octomap::OcTree tree(0.01); //resolution
    octomap::Pointcloud cloud;

    for (int i = 0; i < colorImgs.size();i++) {
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
                cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
            }
        tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3)));
    }

    // populateOctTree(colorImgs, depthImgs, poses, tree, cloud);

    tree.updateInnerOccupancy();
    tree.write("../output/octomap.ply");
    tree.writeBinary("../output/octomap.bt");


    std::cout << "Done!" << std::endl;
    return 0;
}