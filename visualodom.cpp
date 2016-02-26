#include "visualodom.h"

VisualOdom::VisualOdom()
{
}

void VisualOdom::OpenDim(string ParametersFile) {

    //set up a FileStorage object to read camera params from file
    FileStorage fs;
    fs.open(ParametersFile, FileStorage::READ);

    double w1, w2, l1, l2, h1, h2;
    // read dimesion of the car
    fs["w1"] >> w1;
    fs["w2"] >> w2;
    fs["l1"] >> l1;
    fs["l2"] >> l2;
    fs["h1"] >> h1;
    fs["h2"] >> h2;

    Mat Dimtemp(8,4,CV_64F);
    Dimtemp.at<double>(0,0) = w1; Dimtemp.at<double>(0,1) = h1; Dimtemp.at<double>(0,2) = l1; Dimtemp.at<double>(0,3) = 1;
    Dimtemp.at<double>(1,0) = w2; Dimtemp.at<double>(1,1) = h1; Dimtemp.at<double>(1,2) = l1; Dimtemp.at<double>(1,3) = 1;
    Dimtemp.at<double>(2,0) = w2; Dimtemp.at<double>(2,1) = h1; Dimtemp.at<double>(2,2) = l2; Dimtemp.at<double>(2,3) = 1;
    Dimtemp.at<double>(3,0) = w1; Dimtemp.at<double>(3,1) = h1; Dimtemp.at<double>(3,2) = l2; Dimtemp.at<double>(3,3) = 1;
    Dimtemp.at<double>(4,0) = w1; Dimtemp.at<double>(4,1) = h2; Dimtemp.at<double>(4,2) = l1; Dimtemp.at<double>(4,3) = 1;
    Dimtemp.at<double>(5,0) = w2; Dimtemp.at<double>(5,1) = h2; Dimtemp.at<double>(5,2) = l1; Dimtemp.at<double>(5,3) = 1;
    Dimtemp.at<double>(6,0) = w2; Dimtemp.at<double>(6,1) = h2; Dimtemp.at<double>(6,2) = l2; Dimtemp.at<double>(6,3) = 1;
    Dimtemp.at<double>(7,0) = w1; Dimtemp.at<double>(7,1) = h2; Dimtemp.at<double>(7,2) = l2; Dimtemp.at<double>(7,3) = 1;
    CarDim = Dimtemp;

    // close the input file
    fs.release();
}

bool VisualOdom::MotionEst(double size, double level, double min_disp, double y_thresh, double bi_radius, bool display) {

    clock_t startTime = clock();
    // First Points in the cloud (initialisation first image)
    if (Pose.empty()){
        Pts_3D = StereoRig_Prev.Points3D.clone();
        Pts_Des = StereoRig_Prev.descriptorsL.clone();
        Mat Mt = Mat::eye(4, 4, CV_64F);
        Pose = Mt;
        Pose_local = Mat::eye(4,4,CV_64F);
    }
    if(Rotation.empty())
    {
        Rotation = Mat::zeros(3, 1, CV_64F);
        Translation = Mat::zeros(3, 1, CV_64F);
    }

    vector<Point2f> featureLP,featureL,featureRP,featureR,featureLR,featureRL;
    cv::KeyPoint::convert(StereoRig_Prev.keypointsL, featureLP);
    cv::KeyPoint::convert(StereoRig_Prev.keypointsR, featureRP);
    vector<uchar> found1,found2,found3,found4;
    Mat err1, err2, err3, err4;

    calcOpticalFlowPyrLK(StereoRig_Prev.ImageL_Rec_Rescale,StereoRig.ImageL_Rec_Rescale,featureLP,featureL,found1,err1,Size(size,size),level);
    calcOpticalFlowPyrLK(StereoRig_Prev.ImageR_Rec_Rescale,StereoRig.ImageR_Rec_Rescale,featureRP,featureR,found2,err2,Size(size,size),level);

    //calcOpticalFlowPyrLK(StereoRig.ImageL_Gray_Rec,StereoRig.ImageR_Gray_Rec,featureL,featureLR,found3,err3,Size(size,size),level);
    //calcOpticalFlowPyrLK(StereoRig.ImageR_Gray_Rec,StereoRig.ImageL_Gray_Rec,featureR,featureRL,found4,err4,Size(size,size),level);

    vector<Point2f> imagePointsL, imagePointsR, imagePointsLP, imagePointsRP;
    vector<Point3f> scenePoints;
    vector<int> inlier_idx;

    Mat rotVec=Rotation.clone();
    Mat transVec=Translation.clone();

    int num_true_matches=0;
    for( int i = 0; i < StereoRig_Prev.keypointsL.size(); i++ )
    {
        /*if( found1[i]==1 && found2[i]==1 && found3[i]==1 && found4[i]==1 &&
                featureL[i].x-featureR[i].x>min_disp &&
                abs(featureL[i].y-featureR[i].y)<y_thresh &&
                cv::norm(featureL[i]-featureRL[i])<bi_radius && cv::norm(featureR[i]-featureLR[i])<bi_radius )*/
        if( found1[i]==1 && found2[i]==1&&
                        featureL[i].x-featureR[i].x>min_disp &&
                        abs(featureL[i].y-featureR[i].y)<y_thresh)
        {
            imagePointsL.push_back(featureL[i]);
            imagePointsR.push_back(featureR[i]);
            imagePointsLP.push_back(featureLP[i]);
            imagePointsRP.push_back(featureRP[i]);
            Point3f tmp;
            tmp.x=StereoRig_Prev.Points3D.row(i).at<double>(0,0);
            tmp.y=StereoRig_Prev.Points3D.row(i).at<double>(0,1);
            tmp.z=StereoRig_Prev.Points3D.row(i).at<double>(0,2);
            scenePoints.push_back(tmp);
            inlier_idx.push_back(i);
            num_true_matches++;
        }
    }
    if (num_true_matches<3) return false;
    cout<<"Good match candidates "<<num_true_matches<<endl;

    // Ransac
    Mat Inliers;
    solvePnPRansac(scenePoints, imagePointsL, StereoRig_Prev.Intrinsic_Rec_Rescale, Mat::zeros(1, 5, CV_64F), rotVec, transVec, true, 1000, 3.0, 30, Inliers, CV_ITERATIVE);
    Rotation = rotVec.clone(); Translation = transVec.clone();
    cout<<"Motion Ransac Inliers "<<Inliers.rows<<endl;
    if (Inliers.rows<3) return false;

/*
    Mat rotMat=Mat(3,3,CV_64F);
    Rodrigues(rotVec,rotMat);

    // Update the Pose
    Mat Mt1, Mt2;
    hconcat(rotMat,transVec,Mt1);
    Mat LTemp(1, 4, CV_64F); LTemp.at<double>(0,0)=0; LTemp.at<double>(0,1)=0; LTemp.at<double>(0,2)=0; LTemp.at<double>(0,3)=1;
    vconcat(Mt1,LTemp,Mt2);
    Pose = Pose*Mt2.inv();

    Mat scenePointsMat=Mat(scenePoints);
    scenePointsMat.convertTo(scenePointsMat,CV_64F);

    StereoRig.keypointsL.clear();
    StereoRig.keypointsR.clear();
    StereoRig.descriptorsL.release();
    StereoRig.Points3D.release();

    for(int i=0; i<Inliers.rows; i++ ) {
        int id1=Inliers.at<int>(0,i);
        int id2=inlier_idx[id1];

        KeyPoint kp1, kp2;
        kp1=StereoRig_Prev.keypointsL[id2]; kp1.pt.x=imagePointsL[id1].x; kp1.pt.y=imagePointsL[id1].y; kp1.class_id++;
        kp2=StereoRig_Prev.keypointsL[id2]; kp2.pt.x=imagePointsR[id1].x; kp2.pt.y=imagePointsR[id1].y; kp2.class_id++;
        StereoRig.keypointsL.push_back(kp1);
        StereoRig.keypointsR.push_back(kp2);
        StereoRig.descriptorsL.push_back(StereoRig_Prev.descriptorsL.row(id2));
        StereoRig.Points3D.push_back(scenePointsMat.row(id1));
    }
*/
    vector<Point2f> imagePointsL2, imagePointsR2, imagePointsLP2, imagePointsRP2;
    vector<Point3f> scenePoints2;
    vector<int> inlier_idx2;

    num_true_matches=Inliers.rows;

    for(int i=0; i<num_true_matches; i++ ) {
        int id=Inliers.at<int>(0,i);
        imagePointsL2.push_back(imagePointsL[id]);
        imagePointsR2.push_back(imagePointsR[id]);
        imagePointsLP2.push_back(imagePointsLP[id]);
        imagePointsRP2.push_back(imagePointsRP[id]);
        scenePoints2.push_back(scenePoints[id]);
        //inlier_idx2.push_back(id);
        inlier_idx2.push_back(inlier_idx[id]);
    }

    Mat scenePoints2Mat=Mat(scenePoints2);
    scenePoints2Mat.convertTo(scenePoints2Mat,CV_64F);
    double* points=(double*)scenePoints2Mat.data;
    double camera[6]={rotVec.at<double>(0,0),rotVec.at<double>(1,0),rotVec.at<double>(2,0),
                          transVec.at<double>(0,0),transVec.at<double>(1,0),transVec.at<double>(2,0)};

    /*
    ceres::Problem problem;
    for( int i = 0; i < num_true_matches; i++ )
    {
        ceres::CostFunction* left_cost_function = LeftReprojectionError::Create(
                    imagePointsL2[i].x,imagePointsL2[i].y,
                    StereoRig.Intrinsics_Rec.at<double>(0,0),
                    StereoRig.Intrinsics_Rec.at<double>(0,2),
                    StereoRig.Intrinsics_Rec.at<double>(1,2));
        problem.AddResidualBlock(left_cost_function, new ceres::HuberLoss(1.0), camera);

        ceres::CostFunction* right_cost_function = RightReprojectionError::Create(
                    imagePointsR2[i].x,imagePointsR2[i].y,
                    StereoRig.Intrinsics_Rec.at<double>(0,0),
                    StereoRig.Intrinsics_Rec.at<double>(0,2),
                    StereoRig.Intrinsics_Rec.at<double>(1,2),
                    StereoRig.baseline);
        problem.AddResidualBlock(right_cost_function, new ceres::HuberLoss(1.0), camera);
    }
    */

    ceres::Problem problem;
    for( int i = 0; i < num_true_matches; i++ )
    {
        ceres::CostFunction* left_cost_function2 = LeftReprojectionError2::Create(
                    imagePointsL2[i].x,imagePointsL2[i].y,
                    StereoRig.Intrinsic_Rec_Rescale.at<double>(0,0),
                    StereoRig.Intrinsic_Rec_Rescale.at<double>(0,2),
                    StereoRig.Intrinsic_Rec_Rescale.at<double>(1,2));
        problem.AddResidualBlock(left_cost_function2, new ceres::HuberLoss(2.0), camera, points+(3*i));

        ceres::CostFunction* right_cost_function2 = RightReprojectionError2::Create(
                    imagePointsR2[i].x,imagePointsR2[i].y,
                    StereoRig.Intrinsic_Rec_Rescale.at<double>(0,0),
                    StereoRig.Intrinsic_Rec_Rescale.at<double>(0,2),
                    StereoRig.Intrinsic_Rec_Rescale.at<double>(1,2),
                    StereoRig.baseline);
        problem.AddResidualBlock(right_cost_function2, new ceres::HuberLoss(2.0), camera, points+(3*i));

        ceres::CostFunction* left_cost_function3 = LeftReprojectionError3::Create(
                    imagePointsLP2[i].x,imagePointsLP2[i].y,
                    StereoRig_Prev.Intrinsic_Rec_Rescale.at<double>(0,0),
                    StereoRig_Prev.Intrinsic_Rec_Rescale.at<double>(0,2),
                    StereoRig_Prev.Intrinsic_Rec_Rescale.at<double>(1,2));
        problem.AddResidualBlock(left_cost_function3, new ceres::HuberLoss(2.0), points+(3*i));

        ceres::CostFunction* right_cost_function3 = RightReprojectionError3::Create(
                    imagePointsRP2[i].x,imagePointsRP2[i].y,
                    StereoRig_Prev.Intrinsic_Rec_Rescale.at<double>(0,0),
                    StereoRig_Prev.Intrinsic_Rec_Rescale.at<double>(0,2),
                    StereoRig_Prev.Intrinsic_Rec_Rescale.at<double>(1,2),
                    StereoRig_Prev.baseline);
        problem.AddResidualBlock(right_cost_function3, new ceres::HuberLoss(2.0), points+(3*i));
    }

    /*
    Mat scenePointsMat=Mat(scenePoints);
    scenePointsMat.convertTo(scenePointsMat,CV_64F);
    double* points=(double*)scenePointsMat.data;
    double camera[6]={rotVec.at<double>(0,0),rotVec.at<double>(1,0),rotVec.at<double>(2,0),
                          transVec.at<double>(0,0),transVec.at<double>(1,0),transVec.at<double>(2,0)};

    ceres::Problem problem;
    for( int i = 0; i < num_true_matches; i++ )
    {
        ceres::CostFunction* left_cost_function2 = LeftReprojectionError2::Create(
                    imagePointsL[i].x,imagePointsL[i].y,
                    StereoRig.Intrinsics_Rec.at<double>(0,0),
                    StereoRig.Intrinsics_Rec.at<double>(0,2),
                    StereoRig.Intrinsics_Rec.at<double>(1,2));
        problem.AddResidualBlock(left_cost_function2,new ceres::HuberLoss(1.0),camera,points+(3*i));

        ceres::CostFunction* right_cost_function2 = RightReprojectionError2::Create(
                    imagePointsR[i].x,imagePointsR[i].y,
                    StereoRig.Intrinsics_Rec.at<double>(0,0),
                    StereoRig.Intrinsics_Rec.at<double>(0,2),
                    StereoRig.Intrinsics_Rec.at<double>(1,2),
                    StereoRig.baseline);
        problem.AddResidualBlock(right_cost_function2,new ceres::HuberLoss(1.0),camera,points+(3*i));

        ceres::CostFunction* left_cost_function3 = LeftReprojectionError3::Create(
                    imagePointsLP[i].x,imagePointsLP[i].y,
                    StereoRig_Prev.Intrinsics_Rec.at<double>(0,0),
                    StereoRig_Prev.Intrinsics_Rec.at<double>(0,2),
                    StereoRig_Prev.Intrinsics_Rec.at<double>(1,2));
        problem.AddResidualBlock(left_cost_function3,new ceres::HuberLoss(1.0),points+(3*i));

        ceres::CostFunction* right_cost_function3 = RightReprojectionError3::Create(
                    imagePointsRP[i].x,imagePointsRP[i].y,
                    StereoRig_Prev.Intrinsics_Rec.at<double>(0,0),
                    StereoRig_Prev.Intrinsics_Rec.at<double>(0,2),
                    StereoRig_Prev.Intrinsics_Rec.at<double>(1,2),
                    StereoRig_Prev.baseline);
        problem.AddResidualBlock(right_cost_function3,new ceres::HuberLoss(1.0),points+(3*i));
    }
    cout << "NUMBER OF MOTION MATCHES :" << num_true_matches << endl;
    */

    // Solve it
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    //scenePointsMat.copyTo(scenePoints);
    scenePoints2Mat.copyTo(scenePoints2);

    /*
    for(int i=0;i<scenePoints2.size();i++) {
        int id=inlier_idx2[i];
        scenePoints[id].x=scenePoints2[i].x;
        scenePoints[id].y=scenePoints2[i].y;
        scenePoints[id].z=scenePoints2[i].z;
    }*/

    Mat rotMat_ceres=Mat::eye(3,3,CV_64F);
    Mat rotVec_ceres=Mat(3,1,CV_64F,camera);
    Mat transVec_ceres=Mat(3,1,CV_64F,camera+3);

    Rodrigues(rotVec_ceres,rotMat_ceres);
    Rotation = rotVec_ceres.clone(); Translation = transVec_ceres.clone();

    // Update the Pose
    Mat Mt1, Mt2;
    hconcat(rotMat_ceres,transVec_ceres,Mt1);
    Mat LTemp(1, 4, CV_64F); LTemp.at<double>(0,0)=0; LTemp.at<double>(0,1)=0; LTemp.at<double>(0,2)=0; LTemp.at<double>(0,3)=1;
    vconcat(Mt1,LTemp,Mt2);
    Pose = Pose*Mt2.inv();
    Pose_local=Mt2*Pose_local;

    if(cv::norm(Pose_local(Range(0,3),Range(3,4)))>MIN_MOTION) {
        flag_add_points=true;
        Pose_local=Mat::eye(4,4,CV_64F);
        //cout<<"ADDDDDDDD"<<endl;
    }
    Mat P=Pose.inv();
    Mat rot=P(cv::Range(0, 3), cv::Range(0, 3));
    Mat tra=P(cv::Range(0, 3), cv::Range(3, 4));
    cout << "Cam " << -rot.t()*tra << endl;

    // filter outliers
    Mat O=Mat::zeros(3,1,CV_64F);
    Mat B=Mat::zeros(3,1,CV_64F);
    B.at<double>(0,0)=-StereoRig.baseline;
    vector<Point2f> projPointsLP, projPointsRP;
    vector<Point2f> projPointsL, projPointsR;

    //projectPoints(scenePoints, O, O, StereoRig_Prev.Intrinsics_Rec, Mat::zeros(1,5,CV_64F), projPointsLP);
    //projectPoints(scenePoints, O, B, StereoRig_Prev.Intrinsics_Rec, Mat::zeros(1,5,CV_64F), projPointsRP);
    projectPoints(scenePoints2, O, O, StereoRig_Prev.Intrinsic_Rec_Rescale, Mat::zeros(1,5,CV_64F), projPointsLP);
    projectPoints(scenePoints2, O, B, StereoRig_Prev.Intrinsic_Rec_Rescale, Mat::zeros(1,5,CV_64F), projPointsRP);

    //projectPoints(scenePoints, rotVec, transVec, StereoRig.Intrinsics_Rec, Mat::zeros(1,5,CV_64F), projPointsL);
    //projectPoints(scenePoints, rotVec, transVec+B, StereoRig.Intrinsics_Rec, Mat::zeros(1,5,CV_64F), projPointsR);
    //projectPoints(scenePoints, rotVec_ceres, transVec_ceres, StereoRig.Intrinsics_Rec, Mat::zeros(1,5,CV_64F), projPointsL);
    //projectPoints(scenePoints, rotVec_ceres, transVec_ceres+B, StereoRig.Intrinsics_Rec, Mat::zeros(1,5,CV_64F), projPointsR);
    projectPoints(scenePoints2, rotVec_ceres, transVec_ceres, StereoRig.Intrinsic_Rec_Rescale, Mat::zeros(1,5,CV_64F), projPointsL);
    projectPoints(scenePoints2, rotVec_ceres, transVec_ceres+B, StereoRig.Intrinsic_Rec_Rescale, Mat::zeros(1,5,CV_64F), projPointsR);

    StereoRig.keypointsL.clear();
    StereoRig.keypointsR.clear();
    StereoRig.descriptorsL.release();
    StereoRig.Points3D.release();

    /*
    Mat scenePointsMat=Mat(scenePoints);
    for(int i=0;i<scenePoints.size();i++) {
        double errLP=cv::norm(projPointsLP[i]-imagePointsLP[i]);
        double errRP=cv::norm(projPointsRP[i]-imagePointsRP[i]);
        double errL=cv::norm(projPointsL[i]-imagePointsL[i]);
        double errR=cv::norm(projPointsR[i]-imagePointsR[i]);

        int id=inlier_idx[i];
        if(errLP<1.0 && errRP<1.0 && errL<1.0 && errR<1.0) {
            KeyPoint kp1, kp2;
            kp1=StereoRig_Prev.keypointsL[id]; kp1.pt.x=imagePointsL[i].x; kp1.pt.y=imagePointsL[i].y; kp1.class_id++;
            kp2=StereoRig_Prev.keypointsL[id]; kp2.pt.x=imagePointsR[i].x; kp2.pt.y=imagePointsR[i].y; kp2.class_id++;
            StereoRig.keypointsL.push_back(kp1);
            StereoRig.keypointsR.push_back(kp2);
            StereoRig.descriptorsL.push_back(StereoRig_Prev.descriptorsL.row(id));
            StereoRig.Points3D.push_back(scenePointsMat.row(i));
        }
    }
    */

    for(int i=0;i<scenePoints2.size();i++) {
        double errLP=cv::norm(projPointsLP[i]-imagePointsLP2[i]);
        double errRP=cv::norm(projPointsRP[i]-imagePointsRP2[i]);
        double errL=cv::norm(projPointsL[i]-imagePointsL2[i]);
        double errR=cv::norm(projPointsR[i]-imagePointsR2[i]);

        int id=inlier_idx2[i];
        if(errLP<1.0 && errRP<1.0 && errL<1.0 && errR<1.0) {
            KeyPoint kp1, kp2;
            kp1=StereoRig_Prev.keypointsL[id]; kp1.pt.x=imagePointsL2[i].x; kp1.pt.y=imagePointsL2[i].y; kp1.class_id++;
            kp2=StereoRig_Prev.keypointsL[id]; kp2.pt.x=imagePointsR2[i].x; kp2.pt.y=imagePointsR2[i].y; kp2.class_id++;
            StereoRig.keypointsL.push_back(kp1);
            StereoRig.keypointsR.push_back(kp2);
            StereoRig.descriptorsL.push_back(StereoRig_Prev.descriptorsL.row(id));
            StereoRig.Points3D.push_back(scenePoints2Mat.row(i));
        }
    }

    cout << "Number of inliers in motion estimation: " << StereoRig.keypointsL.size() << endl;

    //Pts_3D.release();
    //Pts_Des.release();

    if(flag_add_points) {
        for( int i = 0; i < StereoRig.keypointsL.size(); i++ )
        {
            // Point 3D (1x4)
            Mat PtsTemp(4,1, CV_64F);
            PtsTemp.at<double>(0,0) = StereoRig.Points3D.row(i).at<double>(0,0);
            PtsTemp.at<double>(1,0) = StereoRig.Points3D.row(i).at<double>(0,1);
            PtsTemp.at<double>(2,0) = StereoRig.Points3D.row(i).at<double>(0,2);
            PtsTemp.at<double>(3,0) = 1;
            Mat PtsRef(4,1, CV_64F); PtsRef = Pose*PtsTemp;
            Mat PtsT(1,3, CV_64F);
            PtsT.at<double>(0,0) = PtsRef.at<double>(0,0);
            PtsT.at<double>(0,1) = PtsRef.at<double>(1,0);
            PtsT.at<double>(0,2) = PtsRef.at<double>(2,0);
            Pts_3D.push_back(PtsT);
            Pts_Des.push_back(StereoRig.descriptorsL.row(i));
        }
        flag_add_points=false;
        cout << "Point cloud ADDED"<< endl;
    }
    cout << Pts_3D.size() << endl;

    //cout << "rotation" << rotVec << endl;
    //cout << "translation" << transVec << endl;

    cout << "TIME Motion : " << double( clock() - startTime )*1000 / (double)CLOCKS_PER_SEC<< " milli seconds." << endl;


    if (display == true){
        Mat ImageDis; StereoRig.ImageL_Rec_Rescale.copyTo(ImageDis);
        for (int i = 0; i<imagePointsL2.size() ; i++){
            Point2f ptst; ptst = imagePointsL2[i];
            Point2f ptspt; ptspt = imagePointsLP2[i];
            line(ImageDis, ptspt, ptst, Scalar( 255, 255, 255 ), 1, 8);
        }

        // Display images
        namedWindow("Visual odometry");
        imshow("Visual odometry",ImageDis);
        cv::waitKey(1);

    }

    return true;
}

void VisualOdom::ClearPointCloud(Mat PoseRear, double treshdist){

    // Compute normal of the plane formed by the camera plan
    Mat ZAxe(4,1, CV_64F); ZAxe.at<double>(0,0)=0; ZAxe.at<double>(1,0)=0; ZAxe.at<double>(2,0)=1; ZAxe.at<double>(3,0)=1;
    Mat Plan = PoseRear*ZAxe; double d = sqrt(pow(Plan.at<double>(0,0),2)+pow(Plan.at<double>(1,0),2)+pow(Plan.at<double>(2,0),2));
    double a = Plan.at<double>(0,0)/d; double b = Plan.at<double>(1,0)/d; double c = Plan.at<double>(2,0)/d;
    double xp = Pose.at<double>(0,3); double yp = Pose.at<double>(1,3); double zp = Pose.at<double>(2,3);
    //cout << "a" << xp << "b" << yp << "c" << zp << "d" << d << endl;

    Mat New3DPts; Mat NewDes;
    Mat PtsTest(4,1, CV_64F);
    for( int i = 0; i < Pts_3D.rows; i++ ){

        // Test if the point is behind the plane
        double xx = Pts_3D.row(i).at<double>(0,0); double yy = Pts_3D.row(i).at<double>(0,1); double zz = Pts_3D.row(i).at<double>(0,2);
        PtsTest.at<double>(0,0)=xx; PtsTest.at<double>(1,0)=yy; PtsTest.at<double>(2,0)=zz; PtsTest.at<double>(3,0)=1;
        Mat TempM = PoseRear.inv()*PtsTest;
        double test = TempM.at<double>(2,0);
        //double test = xx*a + yy*b + zz*c + d;

        // Test at what distance the point is from the front car
        double Dist = sqrt(pow(xp-xx,2)+pow(yp-yy,2)+pow(zp-zz,2));

        // Only keep the points in front of the plane
        //cout << "Test" << test << endl;
        if (test>0 && Dist<treshdist)
        {
            New3DPts.push_back(Pts_3D.row(i));
            NewDes.push_back(Pts_Des.row(i));
        }
        else
        {
            //cout << "removed" << endl;
        }
    }

    // Return the new set of points
    Pts_3D.release(); Pts_Des.release();
    Pts_3D = New3DPts; Pts_Des = NewDes;
}


void VisualOdom::ImageSynthesis(Mat PoseRear, Size RearImSize, Mat IntrinsicParam){

    // Compute the inter-car transformation
        Mat InterCar =  Pose.inv()*PoseRear;
        Mat Trans = InterCar.inv();

        // Convert to Gray
        Mat imgGray;
        if(StereoRig.ImageL_Gray_Rec.channels()==3){cvtColor(StereoRig.ImageL_Gray_Rec,imgGray,CV_BGR2GRAY);}
        else {imgGray = StereoRig.ImageL_Gray_Rec;}
        double ox = IntrinsicParam.at<double>(0,2);
        double oy = IntrinsicParam.at<double>(1,2);
        double fx = IntrinsicParam.at<double>(0,0); double inv_fx = 1/fx;
        double fy = IntrinsicParam.at<double>(1,1); double inv_fy = 1/fy;
        double r11 = Trans.at<double>(0,0); double r12 = Trans.at<double>(0,1); double r13 = Trans.at<double>(0,2);
        double r21 = Trans.at<double>(1,0); double r22 = Trans.at<double>(1,1); double r23 = Trans.at<double>(1,2);
        double r31 = Trans.at<double>(2,0); double r32 = Trans.at<double>(2,1); double r33 = Trans.at<double>(2,2);
        double t1 = Trans.at<double>(0,3); double t2 = Trans.at<double>(1,3); double t3 = Trans.at<double>(2,3);
        Mat point3D(4,1, CV_64F); Mat transformedPoint3D(4,1, CV_64F);
        int transformed_r,transformed_c;
        double inv_fxFr = 1/StereoRig.Intrinsics_Rec.at<double>(0,0); double inv_fyFr = 1/StereoRig.Intrinsics_Rec.at<double>(1,1);
        double oxFr = StereoRig.Intrinsics_Rec.at<double>(0,2);
        double oyFr = StereoRig.Intrinsics_Rec.at<double>(1,2);

        //cout << "ox" << ox << endl ;
        Mat imgGray2(RearImSize.height, RearImSize.width, CV_8UC1, Scalar(0));
        Synthetic_Image = imgGray2.clone();

        // Generate
        unsigned char *input = (unsigned char*)(imgGray.data);
        unsigned char *output = (unsigned char*)(Synthetic_Image.data);
        double *DepthAdd = (double*)(StereoRig.DepthMap.data);
        double *TransPtsAdd = (double*)(transformedPoint3D.data);
        double *Pts3Dadd = (double*)(point3D.data);

        for(int r=0;r<imgGray.rows;r++)
            {
                for(int c=0;c<imgGray.cols;c++)
                {
                    if(StereoRig.DepthMap.at<double>(r,c)>0) //If has valid depth value
                    {
                        //Compute the local 3D coordinates of pixel(r,c) of frame 1
                        Pts3Dadd[2] = DepthAdd[StereoRig.DepthMap.cols * r + c]; //z
                        Pts3Dadd[0] = (c-oxFr) * Pts3Dadd[2] * inv_fxFr;	   //x
                        Pts3Dadd[1] = (r-oyFr) * Pts3Dadd[2] * inv_fyFr;	   //y
                        Pts3Dadd[3] = 1.0;			   //homogeneous coordinate

                        //Transform the 3D point using the transformation matrix Rt
                        //transformedPoint3D =  Trans * point3D; the opencv function is slow ...
                        TransPtsAdd[0] = t1 + r11*Pts3Dadd[0] + r12*Pts3Dadd[1] + r13*Pts3Dadd[2];
                        TransPtsAdd[1] = t2 + r21*Pts3Dadd[0] + r22*Pts3Dadd[1] + r23*Pts3Dadd[2];
                        TransPtsAdd[2] = t3 + r31*Pts3Dadd[0] + r32*Pts3Dadd[1] + r33*Pts3Dadd[2];

                        //Project the 3D point to the 2D plane
                        transformed_c = ((TransPtsAdd[0] * fx) / TransPtsAdd[2]) + ox; //transformed x (2D)
                        transformed_r = ((TransPtsAdd[1] * fy) / TransPtsAdd[2]) + oy; //transformed y (2D)

                        //Asign the intensity value to the warped image and compute the difference between the transformed
                        //pixel of frame 1 and the corresponding pixel of frame 2. Compute the error function
                        if((transformed_r>=0 && transformed_r < imgGray2.rows) &
                           (transformed_c>=0 && transformed_c < imgGray2.cols))
                        {
                            output[imgGray2.cols * transformed_r + transformed_c]=input[imgGray.cols * r + c];
                        }
                    }
                }
            }
}

Mat VisualOdom::ROI_Estimation(Mat IntrinsicRear, Mat PoseRear, Size RearimgSize, Mat ImageDis, bool display){

    // The rear image ImageDis is just necessary for test purposes to display the 3D box (need to make another function with less arguments)

    // Compute the inter-car transformation
    //Mat InterCar =  Pose.inv()*PoseRear;
    //Mat Trans = InterCar.inv();

    Mat InterCar = PoseRear.inv()*Pose;
    Mat Trans = InterCar;

    // 4x4 intrinsic
    Mat KK(4,4,CV_64F);
    KK.at<double>(0,0) = IntrinsicRear.at<double>(0,0); KK.at<double>(0,1) = IntrinsicRear.at<double>(0,1); KK.at<double>(0,2) = IntrinsicRear.at<double>(0,2); KK.at<double>(0,3) = 0;
    KK.at<double>(1,0) = IntrinsicRear.at<double>(1,0); KK.at<double>(1,1) = IntrinsicRear.at<double>(1,1); KK.at<double>(1,2) = IntrinsicRear.at<double>(1,2); KK.at<double>(1,3) = 0;
    KK.at<double>(2,0) = IntrinsicRear.at<double>(2,0); KK.at<double>(2,1) = IntrinsicRear.at<double>(2,1); KK.at<double>(2,2) = IntrinsicRear.at<double>(2,2); KK.at<double>(2,3) = 0;
    KK.at<double>(3,0) = 0; KK.at<double>(3,1) = 0; KK.at<double>(3,2) = 0; KK.at<double>(3,3) = 1;

    // Project in the rear camera image
    Mat NewPts3D = Trans*CarDim.t();
    Mat Pts2D = KK*NewPts3D;

    Mat NormPts(2,8,CV_64F);
    vector<Point2f> PtsLine;
    Point2f TempPt;

    // Normalize
    for(int c=0;c<8;c++)
    {
        NormPts.at<double>(0,c) = Pts2D.at<double>(0,c)/Pts2D.at<double>(2,c);
        NormPts.at<double>(1,c) = Pts2D.at<double>(1,c)/Pts2D.at<double>(2,c);
        TempPt.x = NormPts.at<double>(0,c); TempPt.y = NormPts.at<double>(1,c);
        PtsLine.push_back(TempPt);
    }

    // Find maximum
    double maxx, minx, maxy, miny;
    minMaxLoc(NormPts.row(0), &minx, &maxx);
    minMaxLoc(NormPts.row(1), &miny, &maxy);

    if (RearimgSize.height<maxy) { maxy = RearimgSize.height; }
    if (0>minx) { minx = 1; }
    if (RearimgSize.width<maxx) { maxx = RearimgSize.width; }
    if (0>miny) { miny = 1; }
    if (miny>maxy){miny = 1; maxy=RearimgSize.height; }
    if (minx>maxx){minx = 1; maxx=RearimgSize.width; }

    Mat Output(1,4,CV_64F);
    Output.at<double>(0,0) = minx; Output.at<double>(0,1) = miny;
    Output.at<double>(0,2) = maxx; Output.at<double>(0,3) = maxy;

    if (display == true){

    // Display 3D coordinates of the front camera
    Mat FrontCam3D(1,4,CV_64F);
    FrontCam3D.at<double>(0,0)=0; FrontCam3D.at<double>(0,1)=0;
    FrontCam3D.at<double>(0,2)=0; FrontCam3D.at<double>(0,3)=1;
    Mat NewPts3DCam = Trans*FrontCam3D.t();
    Mat Pts2DCam = KK*NewPts3DCam;
    Mat NormPtsCam(2,1,CV_64F);
    NormPtsCam.at<double>(0,0) = Pts2DCam.at<double>(0,0)/Pts2DCam.at<double>(2,0);
    NormPtsCam.at<double>(1,0) = Pts2DCam.at<double>(1,0)/Pts2DCam.at<double>(2,0);
    cv::Point2f ptc; ptc.x = NormPtsCam.at<double>(0,0); ptc.y = NormPtsCam.at<double>(1,0);
    circle(ImageDis,ptc,5.0, Scalar( 0, 0, 255 ), 1, 8 );

    // display axis X
    Mat Xax(1,4,CV_64F);
    Xax.at<double>(0,0)=1; Xax.at<double>(0,1)=0;
    Xax.at<double>(0,2)=0; Xax.at<double>(0,3)=1;
    Mat NewPts3DXax = Trans*Xax.t();
    Mat Pts2DXax = KK*NewPts3DXax;
    Mat NormPtsXax(2,1,CV_64F);
    NormPtsXax.at<double>(0,0) = Pts2DXax.at<double>(0,0)/Pts2DXax.at<double>(2,0);
    NormPtsXax.at<double>(1,0) = Pts2DXax.at<double>(1,0)/Pts2DXax.at<double>(2,0);
    cv::Point2f ptcax; ptcax.x = NormPtsXax.at<double>(0,0); ptcax.y = NormPtsXax.at<double>(1,0);
    line( ImageDis, ptc, ptcax, Scalar( 255, 0, 0 ),  2, 8 );

    // display axis Y
    Mat Yax(1,4,CV_64F);
    Yax.at<double>(0,0)=0; Yax.at<double>(0,1)=1;
    Yax.at<double>(0,2)=0; Yax.at<double>(0,3)=1;
    Mat NewPts3DYax = Trans*Yax.t();
    Mat Pts2DYax = KK*NewPts3DYax;
    Mat NormPtsYax(2,1,CV_64F);
    NormPtsYax.at<double>(0,0) = Pts2DYax.at<double>(0,0)/Pts2DYax.at<double>(2,0);
    NormPtsYax.at<double>(1,0) = Pts2DYax.at<double>(1,0)/Pts2DYax.at<double>(2,0);
    cv::Point2f ptcay; ptcay.x = NormPtsYax.at<double>(0,0); ptcay.y = NormPtsYax.at<double>(1,0);
    line( ImageDis, ptc, ptcay, Scalar( 0, 255, 0 ),  2, 8 );

    // display axis Z
    Mat Zax(1,4,CV_64F);
    Zax.at<double>(0,0)=0; Zax.at<double>(0,1)=0;
    Zax.at<double>(0,2)=1; Zax.at<double>(0,3)=1;
    Mat NewPts3DZax = Trans*Zax.t();
    Mat Pts2DZax = KK*NewPts3DZax;
    Mat NormPtsZax(2,1,CV_64F);
    NormPtsZax.at<double>(0,0) = Pts2DZax.at<double>(0,0)/Pts2DZax.at<double>(2,0);
    NormPtsZax.at<double>(1,0) = Pts2DZax.at<double>(1,0)/Pts2DZax.at<double>(2,0);
    cv::Point2f ptcaz; ptcaz.x = NormPtsZax.at<double>(0,0); ptcaz.y = NormPtsZax.at<double>(1,0);
    line( ImageDis, ptc, ptcaz, Scalar( 0, 0, 255 ),  2, 8 );


    // Display box
    line( ImageDis, PtsLine[0], PtsLine[1], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[1], PtsLine[2], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[2], PtsLine[3], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[3], PtsLine[0], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[4], PtsLine[5], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[5], PtsLine[6], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[6], PtsLine[7], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[7], PtsLine[4], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[0], PtsLine[4], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[1], PtsLine[5], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[2], PtsLine[6], Scalar( 110, 220, 0 ),  2, 8 );
    line( ImageDis, PtsLine[3], PtsLine[7], Scalar( 110, 220, 0 ),  2, 8 );
    namedWindow("PoseEst");
    imshow("PoseEst",ImageDis);
    }
    return Output;
}


void VisualOdom::MotionEstMatching (double thresh, bool display) {

    // First Points in the cloud (initialisation first image)
    if (Pts_3D.empty()){
        Pts_3D = StereoRig_Prev.Points3D;
        Pts_Des = StereoRig_Prev.descriptorsL;
        Mat Mt = Mat::eye(4, 4, CV_64F);
        Pose = Mt;
    }

    // Match left images
    BFMatcher matcher(NORM_HAMMING2);
    vector<DMatch> matches;
    matcher.match(StereoRig_Prev.descriptorsL, StereoRig.descriptorsL, matches);
    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distance between keypoints
    for( int i = 0; i < StereoRig_Prev.descriptorsL.rows; i++)
        { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
        }
    std::vector< cv::DMatch >good_matches;

    int inc=0;
    vector<Point3f> scenePoints;
    vector<Point2f> imagePoints;
    for( int i = 0; i < StereoRig_Prev.descriptorsL.rows; i++ )
        { if( matches[i].distance < (max_dist/1.2) && abs(StereoRig.keypointsL[matches[i].trainIdx].pt.y-StereoRig_Prev.keypointsL[matches[i].queryIdx].pt.y)<200 && abs(StereoRig.keypointsL[matches[i].trainIdx].pt.x-StereoRig_Prev.keypointsL[matches[i].queryIdx].pt.x)<200 )
            {
            good_matches.push_back(matches[i]);
            keypoints_Curr.push_back(StereoRig.keypointsL[matches[i].trainIdx]);
            keypoints_Prev.push_back(StereoRig_Prev.keypointsL[matches[i].queryIdx]);
            descriptors_Prev.push_back(StereoRig_Prev.descriptorsL.row(matches[i].queryIdx));
            descriptors_Curr.push_back(StereoRig.descriptorsL.row(matches[i].trainIdx));
            Points3D.push_back(StereoRig_Prev.Points3D.row(matches[i].queryIdx));
            imagePoints.push_back(StereoRig.keypointsL[matches[i].trainIdx].pt);
            Point3f Temp;
            Temp.x = StereoRig_Prev.Points3D.row(matches[i].queryIdx).at<double>(0,0);
            Temp.y = StereoRig_Prev.Points3D.row(matches[i].queryIdx).at<double>(0,1);
            Temp.z = StereoRig_Prev.Points3D.row(matches[i].queryIdx).at<double>(0,2);
            scenePoints.push_back(Temp);
            inc++;
            }
        }

    //- PnP
    if (keypoints_Prev.size()>0) {
    Mat distoRec = Mat::zeros(1, 5, CV_64F); Mat rotVec(1, 3, CV_64F); Mat transVec(1, 3, CV_64F); Mat Inliers;
    Points3D.convertTo(Points3D,CV_64F);
    solvePnPRansac(scenePoints, imagePoints, StereoRig_Prev.Intrinsic_Rec_Rescale, distoRec, rotVec, transVec, false, 10000, thresh,50,Inliers,CV_P3P);
    Rotation = rotVec.clone(); Translation = transVec.clone();

    cout << "PnP inliers FrontCar" << Inliers.size() << endl;

    // Update the Pose
    Mat Mt1,Mt2; Mat rotMat; Rodrigues(rotVec, rotMat); hconcat(rotMat,transVec,Mt1);
    Mat LTemp(1, 4, CV_64F); LTemp.at<double>(0,0)=0; LTemp.at<double>(0,1)=0; LTemp.at<double>(0,2)=0; LTemp.at<double>(0,3)=1;
    vconcat(Mt1,LTemp,Mt2);
    Pose = Pose*Mt2.inv();
    //cout << Pose << endl;

    // Save the Point in the global 3D point cloud
    Mat BinMat = Mat::zeros(1, StereoRig.descriptorsL.rows, CV_64F);
    for( int i = 0; i < Inliers.rows; i++ )
    {
        BinMat.at<double>(good_matches[Inliers.at<int>(0,i)].trainIdx) = 1;
    }

    for( int i = 0; i < StereoRig.descriptorsL.rows; i++ )
    {
        if (BinMat.at<double>(0,i) == 0)
        {
            // Point 3D (1x4)
            Mat PtsTemp(4,1, CV_64F);
            PtsTemp.at<double>(0,0) = StereoRig.Points3D.row(i).at<double>(0,0);
            PtsTemp.at<double>(1,0) = StereoRig.Points3D.row(i).at<double>(0,1);
            PtsTemp.at<double>(2,0) = StereoRig.Points3D.row(i).at<double>(0,2);
            PtsTemp.at<double>(3,0) = 1;
            Mat PtsRef(4,1, CV_64F); PtsRef = Pose*PtsTemp;
            Mat PtsT(1,3, CV_64F);
            PtsT.at<double>(0,0) = PtsRef.at<double>(0,0);
            PtsT.at<double>(0,1) = PtsRef.at<double>(1,0);
            PtsT.at<double>(0,2) = PtsRef.at<double>(2,0);
            Pts_3D.push_back(PtsT);
            Pts_Des.push_back(StereoRig.descriptorsL.row(i));
        }
    }

    cout << Pts_3D.size() << endl;
    //cout << "rotation" << rotVec << endl;
    //cout << "translation" << transVec << endl;

    }

    if (display == true){
    cv::Mat img_matches;
    cv::drawMatches(StereoRig_Prev.ImageL_Gray_Rec, StereoRig_Prev.keypointsL, StereoRig.ImageL_Gray_Rec, StereoRig.keypointsL, \
        good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow( "Good Matches", img_matches );}
}
