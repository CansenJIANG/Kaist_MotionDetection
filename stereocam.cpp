#include "stereocam.h"
#include <time.h>

StereoCam::StereoCam()
{
}

// Declare with Images
StereoCam::StereoCam(Mat ImageLT, Mat ImageRT)
{
    ImageL = ImageLT; ImageR = ImageRT;
    /*GaussianBlur( ImageL,ImageL, Size( 3, 3 ), 0, 0 );
    GaussianBlur( ImageR,ImageR, Size( 3, 3 ), 0, 0 );*/
}

StereoCam::StereoCam(Mat ImageLT, Mat ImageRT, StereoCam Prev_StereoT)
{
    // load images
    ImageL = ImageLT; ImageR = ImageRT;
    /*GaussianBlur( ImageL,ImageL, Size( 3, 3 ), 0, 0 );
    GaussianBlur( ImageR,ImageR, Size( 3, 3 ), 0, 0 );*/

    // Load cameras parameters
    IntrinsicsL = Prev_StereoT.IntrinsicsL; IntrinsicsR = Prev_StereoT.IntrinsicsR;
    DistorsionL = Prev_StereoT.DistorsionL; DistorsionR = Prev_StereoT.DistorsionR;
    Stereo_Rotation = Prev_StereoT.Stereo_Rotation; Stereo_Translation = Prev_StereoT.Stereo_Translation;
    baseline = Prev_StereoT.baseline; Stereo_imgSize = Prev_StereoT.Stereo_imgSize;
    Intrinsics_Rec = Prev_StereoT.Intrinsics_Rec;
    rmap[0][0] = Prev_StereoT.rmap[0][0]; rmap[1][0] = Prev_StereoT.rmap[1][0];
    rmap[0][1] = Prev_StereoT.rmap[0][1]; rmap[1][1] = Prev_StereoT.rmap[1][1];
}

// Open Intrinsic, distorsion and Extrinsic parameters
// The Stereo Parameters are stored in a single yml file
void StereoCam::OpenParams(string ParametersFile) {

    //set up a FileStorage object to read camera params from file
    FileStorage fs;
    fs.open(ParametersFile, FileStorage::READ);

    // read cameras matrices, distortions and extrinsic coefficients from file
    fs["Camera_MatrixL"] >> IntrinsicsL;
    fs["Distortion_CoefficientsL"] >> DistorsionL;
    fs["Camera_MatrixR"] >> IntrinsicsR;
    fs["Distortion_CoefficientsR"] >> DistorsionR;
    fs["Stereo_Rotation"] >> Stereo_Rotation;
    fs["Stereo_Translation"] >> Stereo_Translation;
    fs["image_Width"] >> Stereo_imgSize.width;
    fs["image_Height"] >> Stereo_imgSize.height;
    fs["baseline"] >> baseline;
    // close the input file
    fs.release();
}


// Open Intrinsic, distorsion and Extrinsic parameters
// The Stereo Parameters are stored in a single yml file
void StereoCam::DisplayStereo(string Which) {

    if (Which == "LR" || Which == "RL" ) // Display both images
    {
        namedWindow("ImageL");
        imshow("ImageL",ImageL);
        namedWindow("ImageR");
        imshow("ImageR",ImageR);
        waitKey(1);
    }
    else if (Which == "R") // Display only right image
    {
        namedWindow("ImageR");
        imshow("ImageR",ImageR);
        waitKey(1);
    }
    else if (Which == "L") // Display only left image
    {
        namedWindow("ImageL");
        imshow("ImageL",ImageL);
        waitKey(1);
    }
    else if (Which == "RecL") // Display only left rectified image
    {
        namedWindow("ImageLRec");
        imshow("ImageLRec",ImageL_Rec);
        waitKey(1);
    }
    else if (Which == "RecR") // Display only rectified image
    {
        namedWindow("ImageRRec");
        imshow("ImageRRec",ImageR_Rec);
        waitKey(1);
    }
    else if (Which == "RecLR" || Which == "RecRL") // Display both rectified images
    {
        namedWindow("ImageRRec");
        imshow("ImageRRec",ImageR_Rec);
        namedWindow("ImageLRec");
        imshow("ImageLRec",ImageL_Rec);
        waitKey(1);
    }
}

// Rectify stereo Images and save new parameters
void StereoCam::RectifyStereo(){

    // Init Rectification (if LUT is empty)
    if (rmap[0][0].empty()) {
    Mat R1,R2,P1,P2,Q;
    SizeAfterRec.width = 1200; SizeAfterRec.height = 376;
    //SizeAfterRec.width = 600; SizeAfterRec.height = 200;

    // Rectification parameters
    stereoRectify(IntrinsicsL, DistorsionL, IntrinsicsR, DistorsionR, Stereo_imgSize, Stereo_Rotation, Stereo_Translation, R1, R2, P1, P2, Q,CV_CALIB_ZERO_DISPARITY,0, SizeAfterRec);

    //Left
    initUndistortRectifyMap(IntrinsicsL, DistorsionL, R1, P1, SizeAfterRec, CV_16SC2 , rmap[0][0], rmap[0][1]);
    //Right
    initUndistortRectifyMap(IntrinsicsR, DistorsionR, R2, P2, SizeAfterRec, CV_16SC2 , rmap[1][0], rmap[1][1]);

    // Save new parameters
    Mat TK(3,3,CV_64F);
    TK.at<double>(0,0) = P1.at<double>(0,0); TK.at<double>(0,1) = P1.at<double>(0,1); TK.at<double>(0,2) = P1.at<double>(0,2);
    TK.at<double>(1,0) = P1.at<double>(1,0); TK.at<double>(1,1) = P1.at<double>(1,1); TK.at<double>(1,2) = P1.at<double>(1,2);
    TK.at<double>(2,0) = P1.at<double>(2,0); TK.at<double>(2,1) = P1.at<double>(2,1); TK.at<double>(2,2) = P1.at<double>(2,2);
    Intrinsics_Rec = TK.clone();

    Mat TTK(3,1,CV_64F);
    TTK.at<double>(0,0) = P2.at<double>(0,3); TTK.at<double>(1,0) = P2.at<double>(1,3);TTK.at<double>(2,0) = P2.at<double>(2,3);
    Mat Temp = TK.inv()*TTK;
    baseline = abs(Temp.at<double>(0,0));
    Stereo_imgSize = SizeAfterRec;

    // Convert to Gray scale
    cvtColor(ImageL,ImageL_Gray,CV_BGR2GRAY);
    cvtColor(ImageR,ImageR_Gray,CV_BGR2GRAY);
    // Rectify images
    remap(ImageL_Gray, ImageL_Gray_Rec, rmap[0][0], rmap[0][1], INTER_LINEAR);
    remap(ImageR_Gray, ImageR_Gray_Rec, rmap[1][0], rmap[1][1], INTER_LINEAR);

    }
    else {

    // Convert to Gray scale
    cvtColor(ImageL,ImageL_Gray,CV_BGR2GRAY);
    cvtColor(ImageR,ImageR_Gray,CV_BGR2GRAY);

    // Rectify images
    remap(ImageL_Gray, ImageL_Gray_Rec, rmap[0][0], rmap[0][1], INTER_LINEAR);
    remap(ImageR_Gray, ImageR_Gray_Rec, rmap[1][0], rmap[1][1], INTER_LINEAR);
    }

}

// Match 2 stereo images
void StereoCam::MatchStereo(bool display){

    // vector of keypoints and descriptor
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1;

    // Detect points
    clock_t startTime = clock();
    Ptr<FeatureDetector> featdetector = new GridAdaptedFeatureDetector(FeatureDetector::create("FAST"), NUM_FEAT, GRID_SIZE, GRID_SIZE);
    //Ptr<FeatureDetector> featdetector = new GridAdaptedFeatureDetector(new DynamicAdaptedFeatureDetector (new FastAdjuster(20,true),12,15,2),800,7,4);
    //Ptr<FeatureDetector> featdetector = new GridAdaptedFeatureDetector(new FastAdjuster(), NUM_FEAT, GRID_SIZE, GRID_SIZE);
    featdetector->detect(ImageL_Rec_Rescale, keypoints1);
    //keypoints1 = ANMS(keypoints1, 9, 0.9, 500);

    ORB orbDesc(NUM_FEAT,1.2f,DESC_PYR_LEVEL,DESC_WIN_SIZE,0,WTA_K,ORB::HARRIS_SCORE,DESC_WIN_SIZE);
    orbDesc.compute(ImageL_Rec_Rescale, keypoints1, descriptors1);

    //OrbDescriptorExtractor orbDesc;
    //orbDesc.compute(ImageL_Gray_Rec, keypoints1, descriptors1);

    vector<Point2f> feature1,feature2,feature3;
    cv::KeyPoint::convert(keypoints1, feature1);

    vector<uchar> found1,found2;
    Mat err1, err2;

    calcOpticalFlowPyrLK(ImageL_Rec_Rescale,ImageR_Rec_Rescale,feature1,feature2,found1,err1,Size(11,11),3,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),0,1e-4);
    //calcOpticalFlowPyrLK(ImageR_Gray_Rec,ImageL_Gray_Rec,feature2,feature3,found2,err2,Size(11,11),3,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),0,1e-4);

    cout << "TIME Matching : " << double( clock() - startTime )*1000 / (double)CLOCKS_PER_SEC<< " milli seconds." << endl;

    int num_true_matches=0;
    for(int i=0; i<keypoints1.size(); i++)
    {
        //if( found1[i]==1 && found2[i]==1 && feature1[i].x-feature2[i].x>MIN_DISP && abs(feature1[i].y-feature2[i].y)<THRESH_Y && cv::norm(feature1[i]-feature3[i])<THRESH_BI)
        if( found1[i]==1 && feature1[i].x-feature2[i].x>MIN_DISP && abs(feature1[i].y-feature2[i].y)<THRESH_Y)
        {
            KeyPoint kp1, kp2;
            kp1=keypoints1[i]; kp1.pt.x=feature1[i].x; kp1.pt.y=feature1[i].y; kp1.class_id=0;
            kp2=keypoints1[i]; kp2.pt.x=feature2[i].x; kp2.pt.y=feature2[i].y; kp2.class_id=0;
            keypointsL.push_back(kp1);
            keypointsR.push_back(kp2);
            descriptorsL.push_back(descriptors1.row(i));
            num_true_matches++;
        }
    }

    if (display == true)
    {
        std::vector< cv::DMatch >good_matches;
        for(int i=0;i<keypointsL.size();i++) good_matches.push_back(cv::DMatch(i,i,0));

        cv::Mat img_matches;
        cv::drawMatches(ImageL_Rec_Rescale, keypointsL, ImageR_Rec_Rescale, keypointsR, \
            good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imshow( "Good Matches", img_matches );
        waitKey(1);
    }
    cout << "NUMBER OF STEREO MATCHES :" << num_true_matches << endl;
}

void StereoCam::ComputeDisparity(double scale, bool display){

    clock_t startTime = clock();
    // Prepare the data to be compatible with elasMP
    Mat lt,rt;
    if(ImageL_Gray_Rec.channels()==3){cvtColor(ImageL_Gray_Rec,lt,CV_BGR2GRAY);}
    else lt = ImageL_Gray_Rec.clone();
    if(ImageR_Gray_Rec.channels()==3)cvtColor(ImageR_Gray_Rec,rt,CV_BGR2GRAY);
    else rt = ImageR_Gray_Rec.clone();
    int bd =0; // border ??

    // Rescale
    int sx = lt.cols*scale; int sy = lt.rows*scale;
    Mat l(sy, sy, CV_8UC1 ); Mat r(sx, sy, CV_8UC1 );
    resize(lt, l, Size(sx,sy));
    resize(rt, r, Size(sx,sy));

    // border if needed
    Mat lb,rb;
    cv::copyMakeBorder(l,lb,0,0,bd,bd,cv::BORDER_REPLICATE);
    cv::copyMakeBorder(r,rb,0,0,bd,bd,cv::BORDER_REPLICATE);

    const cv::Size imsize = rb.size();
    const int32_t dims[3] = {imsize.width,imsize.height,imsize.width}; // bytes per line = width

    Elas::parameters param;
    param.postprocess_only_left = true;
    Elas elas(param);

    cv::Mat leftdpf = cv::Mat::zeros(imsize,CV_32F);
    cv::Mat rightdpf = cv::Mat::zeros(imsize,CV_32F);

    // disparity computation
    elas.process(lb.data,rb.data,leftdpf.ptr<float>(0),rightdpf.ptr<float>(0),dims);

    //Return values
    Mat disp, leftdisp, rightdisp;
    Mat(leftdpf(cv::Rect(bd,0,l.cols,l.rows))).copyTo(disp);
    disp.convertTo(leftdisp,CV_16S,16);

    // Rescale at original scale
    resize(disp*(1/scale), Disparity, Size(ImageL_Gray_Rec.cols,ImageL_Gray_Rec.rows));

    cout << "TIME disparity : " << double( clock() - startTime )*1000 / (double)CLOCKS_PER_SEC<< " milli seconds." << endl;

    if (display == true){
        // view
        Mat show;
        resize(leftdisp*(1/scale), leftdisp, Size(ImageL_Gray_Rec.cols,ImageL_Gray_Rec.rows));
        leftdisp.convertTo(show,CV_8U,1.0/8);
        imshow("disp",show);
        waitKey(1);
    }
}

// Compute sparse 3D points
void StereoCam::Compute3DPts(){
    double disp, depth; double f = Intrinsic_Rec_Rescale.at<double>(0,0);
    Mat Inv_K = Intrinsic_Rec_Rescale.inv(); Mat Pts3D(3,1,CV_64F); Mat Pts2D(3,1,CV_64F);
    Mat Temp;
    for (int i=0;i<keypointsL.size();i++)
    {
        disp = (keypointsL[i].pt.x - keypointsR[i].pt.x);
        depth = f*(baseline/disp);
        Pts2D.at<double>(0,0) = keypointsL[i].pt.x; Pts2D.at<double>(1,0) = keypointsL[i].pt.y; Pts2D.at<double>(2,0)=1;
        Pts3D = (Inv_K*Pts2D)*depth;
        Temp = Pts3D.t();
        Points3D.push_back(Temp);
    }
}

// Compute depth map from disparity
void StereoCam::ComputeDepthMap(){
    Disparity.convertTo(Disparity, CV_64F);
    DepthMap = Intrinsics_Rec.at<double>(0,0)*(baseline/Disparity);
    //DepthMap.convertTo(DepthMap, CV_32FC1);
}

void StereoCam::resizeIm(double scale){

    // Create a rescale image (Gray rescaled)
    int sx = ImageL_Gray_Rec.cols*scale; int sy = ImageL_Gray_Rec.rows*scale;
    Mat l(sy, sy, CV_8UC1 ); Mat r(sx, sy, CV_8UC1 );
    resize(ImageL_Gray_Rec, l, Size(sx,sy));
    resize(ImageR_Gray_Rec, r, Size(sx,sy));
    l.copyTo(ImageL_Rec_Rescale); r.copyTo(ImageR_Rec_Rescale);

    // and corersponding intrinsic parameters
    Mat Newparams(3,3,CV_64F);
    Newparams = Intrinsics_Rec*scale;
    Newparams.at<double>(2,2) = 1;
    Newparams.copyTo(Intrinsic_Rec_Rescale);

}
