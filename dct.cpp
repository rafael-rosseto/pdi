#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#define pi 3.142857

using namespace cv;
using namespace std;

int w, h;
vector<vector<float>> dctdata;

void dctTransform(Mat input) {
    float ci, cj, dct1, sum;
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            if (i == 0)
                ci = 1 / sqrt(w);
            else
                ci = sqrt(2) / sqrt(w);
            if (j == 0)
                cj = 1 / sqrt(h);
            else
                cj = sqrt(2) / sqrt(h);
            sum = 0;
            for (int k = 0; k < w; k++) {
                for (int l = 0; l < h; l++) {
                    dct1 = input.at<uchar>(k, l) *
                           cos((2 * k + 1) * i * pi / (2 * w)) *
                           cos((2 * l + 1) * j * pi / (2 * h));
                    sum = sum + dct1;
                }
            }
            dctdata[i][j] = ci * cj * sum;
        }
    }
}

void idctTransform(Mat output) {
    float ci, cj, dct1, sum;
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            if (i == 0)
                ci = 1 / sqrt(w);
            else
                ci = sqrt(2) / sqrt(w);
            if (j == 0)
                cj = 1 / sqrt(h);
            else
                cj = sqrt(2) / sqrt(h);
            sum = 0;
            for (int k = 0; k < w; k++) {
                for (int l = 0; l < h; l++) {
                    dct1 = dctdata[k][l] * ci * cj *
                           cos((2 * k + 1) * i * pi / (2 * w)) *
                           cos((2 * l + 1) * j * pi / (2 * h));
                    sum = sum + dct1;
                }
            }
            dctdata[i][j] = sum;
            output.at<uchar>(i, j) = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    Mat input = imread(argv[1]);
    cvtColor(input, input, COLOR_RGB2GRAY);
    Mat output = input.clone();
    w = input.cols;
    h = input.rows;
    dctdata.resize(h, vector<float>(w));
    dctTransform(input);
    idctTransform(output);
    imwrite("idct.png", output);
    return 0;
}