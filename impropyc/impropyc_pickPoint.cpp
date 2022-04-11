#include <opencv2/opencv.hpp>

extern "C"
{
//
    __declspec(dllexport)
    int pickAPoint(unsigned char * mdata, int m, int n)
    {
        cv::Mat mat = cv::Mat(m, n, CV_8U, mdata);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                mat.at<uchar>(i, j) = i + j;
        return 0;
    }

//
}
