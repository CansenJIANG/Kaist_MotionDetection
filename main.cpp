#include <QCoreApplication>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <sstream> //Just to record frames
#include <opencv2/opencv.hpp>
#include "stereocam.h"
#include "visualodom.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    // Open the first images for initilisation
    Mat ImageL = imread("/home/jiang/CvDataset/KAIST_DataSet/Images/ImagesLC/camera_left_000000.png", CV_LOAD_IMAGE_COLOR);
    Mat ImageR = imread("/home/jiang/CvDataset/KAIST_DataSet/Images/ImagesRC/camera_right_000000.png", CV_LOAD_IMAGE_COLOR);

    // Read images sequence
    string argL = "/home/jiang/CvDataset/KAIST_DataSet/Images/ImagesLC/camera_left_%06d.png";
    VideoCapture sequenceL(argL);
    string argR = "/home/jiang/CvDataset/KAIST_DataSet/Images/ImagesRC/camera_right_%06d.png";
    VideoCapture sequenceR(argR);

    // Initialize VO
    string filename ="/home/jiang/CvDataset/KAIST_DataSet/Calib/StereoParams.yml";
    VisualOdom VO;
    VO.StereoRig.OpenParams(filename);
    VO.StereoRig.ImageL = ImageL; VO.StereoRig.ImageR = ImageR;
    VO.StereoRig.RectifyStereo(); VO.StereoRig.resizeIm(0.5);
    VO.StereoRig.MatchStereo(false);
    VO.StereoRig.Compute3DPts();

    int it = 0; int idx = 0;
    int failed=0;

    for(;;)
    {
        it++; idx++;
        // Load images
        sequenceL >> ImageL;
        sequenceR >> ImageR;
        if(ImageL.empty()) { cout << "End of Sequence" << endl; break;}

        if (it==3) {

            it = 0;

            // Reinitialize SLAM if failling
            if(failed>3) {
                failed=0;
                VO.StereoRig.keypointsL.clear();
                VO.StereoRig.keypointsR.clear();
                VO.StereoRig.descriptorsL.release();
                VO.StereoRig.Points3D.release();

                VO.StereoRig.RectifyStereo();
                VO.StereoRig.resizeIm(0.5);
                VO.StereoRig.MatchStereo(true);
                VO.StereoRig.Compute3DPts();
                continue;
            }

            // Update the images in the class
            StereoCam StereoTemp(ImageL, ImageR, VO.StereoRig);
            VO.StereoRig_Prev = VO.StereoRig;
            VO.StereoRig = StereoTemp;

            // Rectify Images
            VO.StereoRig.RectifyStereo();

            // Make a resized copy
            VO.StereoRig.resizeIm(0.5);

            // Compute displacement
            if( !VO.MotionEst(11, 3, 3, 0.5, 0.2, true) ) {
                VO.StereoRig=VO.StereoRig_Prev;
                failed++;
                continue;
            }

            VO.StereoRig.keypointsL.clear(); VO.StereoRig.keypointsR.clear();
            VO.StereoRig.descriptorsL.release(); VO.StereoRig.Points3D.release();

            // match 2 images
            VO.StereoRig.MatchStereo(false);
            //VO.StereoRig.MatchStereoAdd(4, 2, 0.5, 0.2, false);
            if(VO.StereoRig.keypointsL.size()<10) {
                VO.StereoRig=VO.StereoRig_Prev;
                failed++;
                continue;
            }


            //Compute 3D points from the sparse matching
            VO.StereoRig.Compute3DPts();

            cv::waitKey(1);



        }
    }

    return a.exec();
}
