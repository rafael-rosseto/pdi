#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

void dilatacao(Mat input, Mat output);

int main(int argc, char *argv[]) {
    ImgOps imagem;
    Mat input = imread(argv[1]);
    Mat output = input.clone();
    dilatacao(input, output);
    imwrite("dilatacao.png", output);
    return 0;
}

void dilatacao(Mat input, Mat output) {
    int masc[3][3] = {
        {0, 1, 0},
        {1, 1, 1},
        {0, 1, 0}};
    int cor;
    
    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.rows; j++) {
            output.at<Vec3b>(Point(i, j)) = Vec3b(0, 0, 0);
        }
    }

    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.rows; j++) {
            cor = input.at<Vec3b>(i, j)[1];
            if (cor > 0) {
                for (int ii = -1; ii <= 1; ii++) {
                    for (int jj = -1; jj <= 1; jj++) {
                        if (masc[ii + 1][jj + 1] == 1)
                            output.at<Vec3b>(Point(j + jj, i + ii)) = Vec3b(255, 255, 255);
                    }
                }
            }
        }
    }
}