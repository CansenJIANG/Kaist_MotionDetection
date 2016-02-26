#ifndef MOTIONDETECT_H
#define MOTIONDETECT_H
#include <Eigen/Core>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <boost/tuple/tuple.hpp>


/*
  1. Assume only LEFT camera is used, the corresponding intrinsic
     parameters K should be given according to the camera.

  2. Parameters definitions:
     -- featurePrev : tracked features in the previous frame;
     -- featureCurr : tracked features in the current frame;
     -- depthCurr: depth value of fea

*/
//template<typename T1, typename T2, typename T3>
//using triple = std::tuple<T1, T2, T3>;

class MotionDetect
{
public:
  MotionDetect();

  ////////////////////////////////////////////////////////////////////////
  ///                        Input parameters                          ///
  ////////////////////////////////////////////////////////////////////////
  std::vector<cv::Point2f> imagePointsLP; // Tracked features in previous frame;
  std::vector<cv::Point2f> imagePointsL;  // Tracked features in current frame;
  std::vector<cv::Point3f> scenePoints;   // Tracked features' 3D coordinate;

  Eigen::Matrix3f K;                      // Intrinsic parameters
  cv::Mat imgLP;                          // Left previous image
  cv::Mat imgL;                           // Left image

  ////////////////////////////////////////////////////////////////////////
  ///                        Output parameters                         ///
  ////////////////////////////////////////////////////////////////////////
  // flow speed of the features
  //  std::vector<boost::tuple<float, float, float> > flow;
  std::vector<cv::Point3f> flows;

  float medianSpeed;
  std::vector<float> flowLikelihood;



  ////////////////////////////////////////////////////////////////////////
  ///                        Functions                                 ///
  ////////////////////////////////////////////////////////////////////////
  void computeFlowSpeed(/*cv::Mat &imgL, cv::Mat &imgLP,*/
                        std::vector<cv::Point2f> &featureL,
                        std::vector<cv::Point2f> &featureLP,/*
                        std::vector<cv::Point3f> &feature3D,
                        Eigen::Matrix3f &K,
                        std::vector<boost::tuple<float, float, float> > &flows*/
                        std::vector<cv::Point3f> &flows);
};

#endif // MOTIONDETECT_H
