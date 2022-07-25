#include <iostream>

#include "imgops.hpp"

using namespace cv;
using namespace std;

void normalizar(Mat src, Mat output) {
    Mat input = src.clone();
    cvtColor(src, input, COLOR_RGB2GRAY);
    output = input.clone();
    int max = 0, min = 255;
    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.cols; j++) {
            if (input.at<uchar>(i, j) > max) max = input.at<uchar>(i, j);
            if (input.at<uchar>(i, j) < min) min = input.at<uchar>(i, j);
        }
    }
    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.cols; j++) {
            output.at<uchar>(i, j) = round((input.at<uchar>(i, j) - min) / (max - min) * 255);
        }
    }
}

int main(int argc, char *argv[]) {
    Mat input, output;
    input = imread(argv[1]);
    output = input.clone();
    // namedWindow("Processamento Digital de Imagens", WINDOW_AUTOSIZE);
    normalizar(input, output);
    /* imshow("Processamento Digital de Imagens", output);
    while (getWindowProperty("Processamento Digital de Imagens", WND_PROP_VISIBLE))
        waitKey(50);
    destroyAllWindows(); */
    return 0;
}