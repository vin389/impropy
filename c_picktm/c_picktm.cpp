#include <opencv2/opencv.hpp>
extern "C"
{
    __declspec(dllexport)
    int myCvMat(unsigned char* mdata, int m, int n)
    {
        cv::Mat img = cv::Mat(m, n, CV_8U, mdata);
        img.setTo(cv::Scalar(0));
        for (int i = 0; i < m; i++)
            if (i < n) img.at<uchar>(i, i) = 255;
        return 0;
    }
}
