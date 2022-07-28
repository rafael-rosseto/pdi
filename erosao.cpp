#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

void erosao(Mat input, Mat output);

int main(int argc, char *argv[]) {
    ImgOps imagem;
    Mat input = imread(argv[1]);
    Mat output = input.clone();
    erosao(input, output);
    imwrite("erosao.png", output);
    return 0;
}

void erosao(Mat input, Mat output) {
    int masc[3][3] = {
        {0, 1, 0},
        {1, 1, 1},
        {0, 1, 0}};
    int cor;
    bool remove;
    
    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.rows; j++) {
            output.at<Vec3b>(Point(i, j)) = Vec3b(0, 0, 0);
        }
    }

    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.rows; j++) {
            cor = input.at<Vec3b>(i, j)[1];
            if (cor > 0) {
                remove = false;
                for (int ii = -1; ii <= 1; ii++) {
                    for (int jj = -1; jj <= 1; jj++) {
                        if (masc[ii + 1][jj + 1] == 1 && input.at<Vec3b>(i + ii, j + jj)[1] == 0)
                            remove = true;
                        if (remove)
                            output.at<Vec3b>(Point(j, i)) = Vec3b(0, 0, 0);
                        else
                            output.at<Vec3b>(Point(j, i)) = Vec3b(255, 255, 255);
                    }
                }
            }
        }
    }
}