#pragma once
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#define pi 3.142857

using namespace cv;
using namespace std;

class ImgOps {
   public:
    int w, h;
    vector<vector<float>> dctdata;

    void original(Mat input) {
        imshow("PDI", input);
        waitKey(0);
    }

    void inversao(Mat input, Mat output) {
        output = Scalar(255, 255, 255) - input;
        imshow("PDI", output);
        waitKey(0);
    }

    void cinza(Mat input, Mat output) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                output.at<Vec3b>(i, j)[0] = output.at<Vec3b>(i, j)[1] = output.at<Vec3b>(i, j)[2] =
                    (input.at<Vec3b>(i, j)[0] +
                     input.at<Vec3b>(i, j)[1] +
                     input.at<Vec3b>(i, j)[2]) /
                    3;
            }
        }
        imshow("PDI", output);
        waitKey(0);
    }

    void canalB(Mat input, Mat output) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                output.at<Vec3b>(i, j)[0] =
                    output.at<Vec3b>(i, j)[1] =
                        output.at<Vec3b>(i, j)[2] =
                            input.at<Vec3b>(i, j)[0];
            }
        }
        imshow("PDI", output);
        waitKey(0);
    }

    void canalG(Mat input, Mat output) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                output.at<Vec3b>(i, j)[0] =
                    output.at<Vec3b>(i, j)[1] =
                        output.at<Vec3b>(i, j)[2] =
                            input.at<Vec3b>(i, j)[1];
            }
        }
        imshow("PDI", output);
        waitKey(0);
    }

    void canalR(Mat input, Mat output) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                output.at<Vec3b>(i, j)[0] =
                    output.at<Vec3b>(i, j)[1] =
                        output.at<Vec3b>(i, j)[2] =
                            input.at<Vec3b>(i, j)[2];
            }
        }
        imshow("PDI", output);
        waitKey(0);
    }

    void media(Mat input, Mat output, int range) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                int pixel[3] = {0};
                int count = 0;
                for (int ii = i - range; ii <= i + range; ii++) {
                    for (int jj = j - range; jj <= j + range; jj++) {
                        if (ii < 0 || jj < 0 || ii >= input.rows || jj >= input.cols)
                            continue;
                        pixel[0] += input.at<Vec3b>(ii, jj)[0];
                        pixel[1] += input.at<Vec3b>(ii, jj)[1];
                        pixel[2] += input.at<Vec3b>(ii, jj)[2];
                        count++;
                    }
                }
                output.at<Vec3b>(i, j)[0] = pixel[0] * (1.0f / count);
                output.at<Vec3b>(i, j)[1] = pixel[1] * (1.0f / count);
                output.at<Vec3b>(i, j)[2] = pixel[2] * (1.0f / count);
            }
        }
        imshow("PDI", output);
        waitKey(0);
    }

    void binario(Mat input, Mat output, short alpha) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                int aux = input.at<Vec3b>(i, j)[0] +
                          input.at<Vec3b>(i, j)[1] +
                          input.at<Vec3b>(i, j)[2] / 3;
                if (aux > alpha)
                    output.at<Vec3b>(Point(i, j)) = Vec3b(255, 255, 255);
                else
                    output.at<Vec3b>(Point(i, j)) = Vec3b(0, 0, 0);
            }
        }
        imshow("PDI", output);
        waitKey(0);
    }

    void limiar(Mat input, Mat output, short alpha) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                int aux = input.at<Vec3b>(i, j)[0] +
                          input.at<Vec3b>(i, j)[1] +
                          input.at<Vec3b>(i, j)[2] / 3;
                if (aux > alpha)
                    output.at<Vec3b>(Point(i, j)) = input.at<Vec3b>(Point(i, j));
                else
                    output.at<Vec3b>(Point(i, j)) = Vec3b(0, 0, 0);
            }
        }
        imshow("PDI", output);
        waitKey(0);
    }

    void mediana(Mat input, Mat output, int range) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                vector<int> channelR, channelG, channelB;
                int count = 0;
                for (int ii = i - range; ii <= i + range; ii++) {
                    for (int jj = j - range; jj <= j + range; jj++) {
                        if (ii < 0 || jj < 0 || ii >= input.rows || jj >= input.cols)
                            continue;
                        channelR.push_back(input.at<Vec3b>(ii, jj)[2]);
                        channelG.push_back(input.at<Vec3b>(ii, jj)[1]);
                        channelB.push_back(input.at<Vec3b>(ii, jj)[0]);
                        count++;
                    }
                }
                sort(channelR.begin(), channelR.end());
                output.at<Vec3b>(i, j)[2] = channelR.at(count / 2 + 1);
                sort(channelG.begin(), channelG.end());
                output.at<Vec3b>(i, j)[1] = channelG.at(count / 2 + 1);
                sort(channelB.begin(), channelB.end());
                output.at<Vec3b>(i, j)[0] = channelB.at(count / 2 + 1);
            }
        }
        imshow("PDI", output);
        waitKey(0);
    }

    void normalizar(Mat input, Mat output) {
        int high = 0, low = 255;
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                if (input.at<uchar>(i, j) > high)
                    high = input.at<uchar>(i, j);
                if (input.at<uchar>(i, j) < low)
                    low = input.at<uchar>(i, j);
            }
        }
        for (int i = 0; i < input.cols; i++)
            for (int j = 0; j < input.rows; j++)
                output.at<uchar>(i, j) = input.at<uchar>(i, j) * (255 / high) - low;
    }

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
};