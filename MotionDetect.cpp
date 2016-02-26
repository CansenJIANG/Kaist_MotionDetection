#include "MotionDetect.h"

MotionDetect::MotionDetect()
{


}

void MotionDetect::computeFlowSpeed(/*cv::Mat &imgL, cv::Mat &imgLP, */
                                    std::vector<cv::Point2f> &featureL,
                                    std::vector<cv::Point2f> &featureLP,/*
                                    std::vector<cv::Point3f> &feature3D,
                                    Eigen::Matrix3f &K,
                                    std::vector<boost::tuple<float, float, float> >&flows*/
                                    std::vector<cv::Point3f> &flows)
{
  std::vector<cv::Point2f>::iterator it_featureL  = featureL.begin();
  std::vector<cv::Point2f>::iterator it_featureLP = featureLP.begin();
  for(unsigned int i=0; i<featureL.size(); i++)
    {
      // flow value in x, y direction + magnitude of the flow
      cv::Point3f flow(.0, .0, .0);
      flow.x = (*it_featureLP).x - (*it_featureLP).x;
      flow.y = (*it_featureLP).y - (*it_featureLP).y;
      flow.z = cv::norm(cv::Point2f(flow.x, flow.y));//std::sqrt(flow.x*flow.x + flow.y*flow.y);
      flows.push_back(flow);
    }
}
