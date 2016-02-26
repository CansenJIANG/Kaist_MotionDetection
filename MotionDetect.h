#ifndef MOTIONDETECT_H
#define MOTIONDETECT_H
#include <Eigen/Core>
/*
  1. Assume only left or right camera is used, the corresponding intrinsic
     parameters K should be given according to the camera.

  2. Parameters definitions:
     -- featurePrev : tracked features in the previous frame;
     -- featureCurr : tracked features in the current frame;
     -- depthCurr: depth value of fea

*/
class MotionDetect
{
public:
  MotionDetect();
  std::vector<float> featurePrev; // Tracked features in current frame;
  std::vector<float> featureCurr; // Tracked features in current frame;
  std::vector<float> depthCurr; // Tracked features in current frame;
  std::vector<float> depthNext; // Tracked features in current frame;
  Eigen::Matrix3f K;
};

#endif // MOTIONDETECT_H
