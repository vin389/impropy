#include <opencv2/opencv.hpp>

extern "C"
{
//
    // This is a demonstration function that shows how to call
    // a function (that uses OpenCV) from Python.
    // This function receives an m-by-n uint8 numpy array, the
    // m and n, and then modify the contents of the numpy array.
    __declspec(dllexport)
    int myCvMat(unsigned char * mdata, int m, int n)
    {
        cv::Mat mat = cv::Mat(m, n, CV_8U, mdata);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                mat.at<uchar>(i, j) = i + j;
        return 0;
    }
//
}
