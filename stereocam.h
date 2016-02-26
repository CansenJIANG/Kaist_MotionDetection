#ifndef STEREOCAM_H
#define STEREOCAM_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "elasFiles/elas.h"
#include "MiscTool.h"

#include "ceres/ceres.h"
#include "ceres/rotation.h"

using namespace std;
using namespace cv;

#define NUM_FEAT 1000
#define GRID_SIZE 10
#define MIN_DISP 1.0
#define THRESH_Y 0.5
#define THRESH_BI 0.2

#define DESC_PYR_LEVEL 8 //8
#define DESC_WIN_SIZE 31 //15
#define WTA_K 2

class StereoCam
{
    public:
    // Images
    Mat ImageL, ImageR, ImageL_Gray, ImageR_Gray;
    Mat ImageL_Rec, ImageR_Rec, ImageL_Gray_Rec, ImageR_Gray_Rec;
    Mat ImageL_Rec_Rescale, ImageR_Rec_Rescale;
    Mat Disparity;
    Mat DepthMap;
    Size SizeAfterRec;

    // Camera Parameters
    Mat IntrinsicsL, DistorsionL, IntrinsicsR, DistorsionR, Stereo_Rotation, Stereo_Translation;
    Size Stereo_imgSize;
    Mat Intrinsic_Rec_Rescale;

    // Rectified Parameters
    Mat Intrinsics_Rec;
    Mat rmap[2][2]; // Rectif LUT
    double baseline;

    // Stereo Matching
    vector<KeyPoint> keypointsL, keypointsR;
    Mat descriptorsL, descriptorsR;
    Mat Points3D; Mat PointsId;

    StereoCam(Mat ImageL, Mat ImageR); // Declare with images
    StereoCam(Mat ImageL, Mat ImageR, StereoCam Prev_Stereo); // Copy the previous informations
    StereoCam();

    // Functions:
    void OpenParams (string ParametersFile); // Open cameras' parameters
    void DisplayStereo(string Which); // Display images
    void RectifyStereo(); // Rectify stereo images
    //void MatchStereo(string DetectorName, string DescriptorName, double Thresh, bool display); // Match 2 stereo images
    void MatchStereo(bool display);
    void ComputeDisparity(double scale, bool display); // Compute the disparity, the scale of the images can be changed to speed up
    void Compute3DPts(); // Compute 3D points from the sparse 2D matching
    void ComputeDepthMap(); // Compute depth map from disparity
    void resizeIm(double scale); // Rescale the images and adjust the parameters
};

#endif // STEREOCAM_H
