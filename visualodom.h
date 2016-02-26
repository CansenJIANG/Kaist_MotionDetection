#ifndef VISUALODOM_H
#define VISUALODOM_H

#include <iostream>
#include "stereocam.h"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <eigen3/Eigen/Dense>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

#define MIN_MOTION 1.5

class VisualOdom
{
public:
    // Variables
    vector<KeyPoint> keypoints_Prev, keypoints_Curr;
    Mat descriptors_Prev, descriptors_Curr;
    Mat Points3D; Mat Translation; Mat Rotation;
    StereoCam StereoRig, StereoRig_Prev;
    Mat Synthetic_Image;

    // 3d Point cloud
    Mat Pts_3D; // 3D point coordinates
    Mat Pts_Des;    // Corresponding Descriptors
    Mat Pose;
    Mat CarDim;

    Mat Pose_local;
    bool flag_add_points=true;

    // Stereo rigs
    VisualOdom();
    void OpenDim(string ParametersFile); // Open location of the camera on the car
    bool MotionEst(double size, double level, double min_disp, double y_thresh, double bi_radius, bool display);
    void ClearPointCloud(Mat PoseRear, double treshdist); // Remove teh points behind the rear car
    void ImageSynthesis(Mat PoseRear, Size RearImSize, Mat IntrinsicParam);
    Mat ROI_Estimation(Mat IntrinsicRear, Mat PoseRear, Size RearimgSize, Mat ImageDis, bool display); // Estimate the ROI using the car dimensions
    void MotionEstMatching (double thresh, bool display);
};

// FOR CERES

struct LeftReprojectionError {
LeftReprojectionError(double u, double v, double x, double y, double z, double f, double cx, double cy)
: u(u), v(v), x(x), y(y), z(z), f(f), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const camera,
              T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T point[3];
        point[0] = T(x);
        point[1] = T(y);
        point[2] = T(z);

        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        p[0] /= p[2];
        p[1] /= p[2];

        T up =  T(f)*p[0] + T(cx);
        T vp =  T(f)*p[1] + T(cy);

        // The error is the difference between the predicted and observed position.
        residuals[0] = up - T(u);
        residuals[1] = vp - T(v);

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double u,
                                       const double v,
                                       const double x,
                                       const double y,
                                       const double z,
                                       const double f,
                                       const double cx,
                                       const double cy) {
        return (new ceres::AutoDiffCostFunction<LeftReprojectionError, 2, 6>(new LeftReprojectionError(u, v, x, y, z, f, cx, cy)));
    }

    double u, v;
    double x, y, z;
    double f;
    double cx;
    double cy;
};

struct RightReprojectionError {
RightReprojectionError(double u, double v, double x, double y, double z, double f, double cx, double cy, double baseline)
: u(u), v(v), x(x), y(y), z(z), f(f), cx(cx), cy(cy), baseline(baseline) {}

    template <typename T>
    bool operator()(const T* const camera,
              T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T point[3];
        point[0] = T(x);
        point[1] = T(y);
        point[2] = T(z);

        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        p[0] -= T(baseline);

        p[0] /= p[2];
        p[1] /= p[2];

        T up =  T(f)*p[0] + T(cx);
        T vp =  T(f)*p[1] + T(cy);

        // The error is the difference between the predicted and observed position.
        residuals[0] = up - T(u);
        residuals[1] = vp - T(v);

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double u,
                                       const double v,
                                       const double x,
                                       const double y,
                                       const double z,
                                       const double f,
                                       const double cx,
                                       const double cy,
                                       const double baseline) {
        return (new ceres::AutoDiffCostFunction<RightReprojectionError, 2, 6>(new RightReprojectionError(u, v, x, y, z, f, cx, cy, baseline)));
    }

    double u, v;
    double x, y, z;
    double f;
    double cx;
    double cy;
    double baseline;
};

// Approach 2: optimize points and the pose
struct LeftReprojectionError2 {
LeftReprojectionError2(double u, double v, double f, double cx, double cy)
: u(u), v(v), f(f), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
              T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        p[0] /= p[2];
        p[1] /= p[2];

        T up =  T(f)*p[0] + T(cx);
        T vp =  T(f)*p[1] + T(cy);

        // The error is the difference between the predicted and observed position.
        residuals[0] = up - T(u);
        residuals[1] = vp - T(v);

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double u,
                                       const double v,
                                       const double f,
                                       const double cx,
                                       const double cy) {
        return (new ceres::AutoDiffCostFunction<LeftReprojectionError2, 2, 6, 3>(new LeftReprojectionError2(u, v, f, cx, cy)));
    }

    double u, v;
    double f;
    double cx;
    double cy;
};

struct RightReprojectionError2 {
RightReprojectionError2(double u, double v, double f, double cx, double cy, double baseline)
: u(u), v(v), f(f), cx(cx), cy(cy), baseline(baseline) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
              T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        p[0] -= T(baseline);

        p[0] /= p[2];
        p[1] /= p[2];

        T up =  T(f)*p[0] + T(cx);
        T vp =  T(f)*p[1] + T(cy);

        // The error is the difference between the predicted and observed position.
        residuals[0] = up - T(u);
        residuals[1] = vp - T(v);

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double u,
                                       const double v,
                                       const double f,
                                       const double cx,
                                       const double cy,
                                       const double baseline) {
        return (new ceres::AutoDiffCostFunction<RightReprojectionError2, 2, 6, 3>(new RightReprojectionError2(u, v, f, cx, cy, baseline)));
    }

    double u, v;
    double f;
    double cx;
    double cy;
    double baseline;
};

struct LeftReprojectionError3 {
LeftReprojectionError3(double u, double v, double f, double cx, double cy)
: u(u), v(v), f(f), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const point,
              T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        p[0] = point[0];
        p[1] = point[1];
        p[2] = point[2];

        p[0] /= p[2];
        p[1] /= p[2];

        T up =  T(f)*p[0] + T(cx);
        T vp =  T(f)*p[1] + T(cy);

        // The error is the difference between the predicted and observed position.
        residuals[0] = up - T(u);
        residuals[1] = vp - T(v);

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double u,
                                       const double v,
                                       const double f,
                                       const double cx,
                                       const double cy) {
        return (new ceres::AutoDiffCostFunction<LeftReprojectionError3, 2, 3>(new LeftReprojectionError3(u, v, f, cx, cy)));
    }

    double u, v;
    double f;
    double cx;
    double cy;
};

struct RightReprojectionError3 {
RightReprojectionError3(double u, double v, double f, double cx, double cy, double baseline)
: u(u), v(v), f(f), cx(cx), cy(cy), baseline(baseline) {}

    template <typename T>
    bool operator()(const T* const point,
              T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];

        p[0] = point[0];
        p[1] = point[1];
        p[2] = point[2];

        p[0] -= T(baseline);

        p[0] /= p[2];
        p[1] /= p[2];

        T up =  T(f)*p[0] + T(cx);
        T vp =  T(f)*p[1] + T(cy);

        // The error is the difference between the predicted and observed position.
        residuals[0] = up - T(u);
        residuals[1] = vp - T(v);

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double u,
                                       const double v,
                                       const double f,
                                       const double cx,
                                       const double cy,
                                       const double baseline) {
        return (new ceres::AutoDiffCostFunction<RightReprojectionError3, 2, 3>(new RightReprojectionError3(u, v, f, cx, cy, baseline)));
    }

    double u, v;
    double f;
    double cx;
    double cy;
    double baseline;
};


#endif // VISUALODOM_H
