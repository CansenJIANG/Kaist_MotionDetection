#include "MiscTool.h"
vector<cv::KeyPoint> ANMS(vector<cv::KeyPoint> keypoints, double MaxRad, double crobust, int MaxPoints){
    // Adaptative non maxima suppression
    // crobust = 0.9; MaxRad = 8;

    // Init
    Mat Rad(1,keypoints.size(),CV_64F);
    double dist;
    for (int i = 0; i< keypoints.size(); i++)
    {
        double Val = keypoints[i].response;
        Rad.at<double>(i) = 10000;
            for (int j = 0; j< keypoints.size(); j++){

                if (Val < crobust*keypoints[j].response)
                {
                    dist = sqrt(pow(keypoints[j].pt.x - keypoints[i].pt.x,2)+pow(keypoints[j].pt.y - keypoints[i].pt.y,2));
                    if (dist < Rad.at<double>(i) ) //&& dist > MaxRad
                    {
                        Rad.at<double>(i) = dist;
                    }
                }
            }

    }

    // Sort by radius
    Mat Indx(1,keypoints.size(),CV_32S);
    sortIdx(Rad, Indx, CV_SORT_DESCENDING);
    vector<cv::KeyPoint> newkeypoints;
    int NbPt = MaxPoints;
    if (MaxPoints>keypoints.size())
    {NbPt = keypoints.size(); }
    for (int i=0;i<NbPt;i++)
    {
        if (Rad.at<double>(Indx.at<int>(i))>MaxRad){
        newkeypoints.push_back(keypoints[Indx.at<int>(i)]);
        }
    }
    //cout << Indx << endl;
    return newkeypoints;

    //sortIdx(vector<cv::KeyPoint>.response, B, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
}


void mat2quat(float *R, float *q) {
    float m00=R[0], m01=R[1], m02=R[2];
    float m10=R[3], m11=R[4], m12=R[5];
    float m20=R[6], m21=R[7], m22=R[8];

    float tr = m00 + m11 + m22;
    float qw, qx, qy, qz;

    if (tr > 0) {
      float S = sqrt(tr+1.0) * 2; // S=4*qw
      qw = 0.25 * S;
      qx = (m21 - m12) / S;
      qy = (m02 - m20) / S;
      qz = (m10 - m01) / S;
    }
    else if ((m00 > m11)&(m00 > m22)) {
      float S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx
      qw = (m21 - m12) / S;
      qx = 0.25 * S;
      qy = (m01 + m10) / S;
      qz = (m02 + m20) / S;
    }
    else if (m11 > m22) {
      float S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
      qw = (m02 - m20) / S;
      qx = (m01 + m10) / S;
      qy = 0.25 * S;
      qz = (m12 + m21) / S;
    }
    else {
      float S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
      qw = (m10 - m01) / S;
      qx = (m02 + m20) / S;
      qy = (m12 + m21) / S;
      qz = 0.25 * S;
    }

    q[0]=qw;
    q[1]=qx;
    q[2]=qy;
    q[3]=qz;
}

void quat2mat(float *q, float *R) {
    float qw=q[0];
    float qx=q[1];
    float qy=q[2];
    float qz=q[3];

    float sqw = qw*qw;
    float sqx = qx*qx;
    float sqy = qy*qy;
    float sqz = qz*qz;

    // invs (inverse square length) is only required if quaternion is not already normalised
    float invs = 1 / (sqx + sqy + sqz + sqw);
    float m00 = ( sqx - sqy - sqz + sqw)*invs ; // since sqw + sqx + sqy + sqz =1/invs*invs
    float m11 = (-sqx + sqy - sqz + sqw)*invs ;
    float m22 = (-sqx - sqy + sqz + sqw)*invs ;

    float tmp1 = qx*qy;
    float tmp2 = qz*qw;
    float m10 = 2.0 * (tmp1 + tmp2)*invs ;
    float m01 = 2.0 * (tmp1 - tmp2)*invs ;

    tmp1 = qx*qz;
    tmp2 = qy*qw;
    float m20 = 2.0 * (tmp1 - tmp2)*invs ;
    float m02 = 2.0 * (tmp1 + tmp2)*invs ;
    tmp1 = qy*qz;
    tmp2 = qx*qw;
    float m21 = 2.0 * (tmp1 + tmp2)*invs ;
    float m12 = 2.0 * (tmp1 - tmp2)*invs ;

    R[0]=m00; R[1]=m01; R[2]=m02;
    R[3]=m10; R[4]=m11; R[5]=m12;
    R[6]=m20; R[7]=m21; R[8]=m22;
}

Mat correctGamma( Mat& img, double gamma ) {
 double inverse_gamma = 1.0 / gamma;

 Mat lut_matrix(1, 256, CV_8UC1 );
 uchar * ptr = lut_matrix.ptr();
 for( int i = 0; i < 256; i++ )
   ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );

 Mat result;
 LUT( img, lut_matrix, result );

 return result;
}

void angleaxis2Mat(double *q, Mat R) {
    double theta2=q[0]*q[0]+q[1]*q[1]+q[2]*q[2];
    if (theta2 > std::numeric_limits<double>::epsilon()) {
        double theta = sqrt(theta2);
        double x=q[0]/theta, y=q[1]/theta, z=q[2]/theta;
        double x2=x*x, y2=y*y, z2=z*z;
        double xy=x*y, xz=x*z, yz=y*z;
        double s=sin(theta), c=cos(theta), c_=1-c;

        R.at<double>(0,0) = c+x2*c_;    R.at<double>(0,1)=xy*c_-z*s;    R.at<double>(0,2)=xz*c_+y*s;
        R.at<double>(1,0) = xy*c_+z*s;  R.at<double>(1,1)=c+y2*c_;      R.at<double>(1,2)=yz*c_-x*s;
        R.at<double>(2,0) = xz*c_-y*s;  R.at<double>(2,1)=yz*c_+x*s;    R.at<double>(2,2)=c+z2*c_;
    }
    else {
        R.at<double>(0,0) = 1; R.at<double>(0,1)=-q[2]; R.at<double>(0,2)=q[1];
        R.at<double>(1,0) = q[2]; R.at<double>(1,1)=1; R.at<double>(1,2)=-q[0];
        R.at<double>(2,0) = -q[1]; R.at<double>(2,1)=q[0]; R.at<double>(2,2)=1;
    }
}
