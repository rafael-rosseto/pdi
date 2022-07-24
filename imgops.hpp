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

    void inversao(Mat input, Mat output) {
        output = Scalar(255, 255, 255) - input;
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

    void filtraMax(Mat input, Mat output, int range) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {  // laços para percorrer a imagem
                // em cada pixel fazer:
                if (i < range || i > input.cols - range || j < range || j > input.rows - range) {
                    //se está no canto superior esquerdo
                    if (i < range && j < range) {
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = 0; m < i + range; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(max, max, max);
                    }
                    //se está no canto superior direito 
                    else if (i > (input.cols - range) && j < range) {
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(max, max, max);
                    }
                    //se está no canto inferior direito
                    else if (i > (input.cols - range) && j > (input.rows - range)) {
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(max, max, max);
                    }
                    //se está no canto inferior esquerdo
                    else if (i < range && j > (input.rows - range)) {
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = 0; m < i + range; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(max, max, max);
                    }
                    //se está na borda de cima
                    else if (j < range) {
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(max, max, max);
                    }
                    //se está na borda da direita
                    else if (i > input.cols - range) {
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(max, max, max);
                    }
                    //se está na borda de baixo
                    else if (j > input.rows - range) {
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(max, max, max);
                    }
                    //se está na borda da esquerda
                    else {
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = 0; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(max, max, max);
                    }
                }
                else {
                    int max = input.at<Vec3b>(i, j)[0];
                    for (int m = i - range; m < i + range; m++) {
                        for (int n = j - range; n < j + range; n++) {
                            if (max < input.at<Vec3b>(n, m)[0])
                                max = input.at<Vec3b>(n, m)[0];
                        }
                    }
                    output.at<Vec3b>(Point(i, j)) = Vec3b(max, max, max);
                }
            }
        }
    }

    void filtraMin(Mat input, Mat output, int range) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {  // laços para percorrer a imagem
                // em cada pixel fazer:
                if (i < range || i > input.cols - range || j < range || j > input.rows - range) {
                    //se está no canto superior esquerdo
                    if (i < range && j < range) {
                        int min = input.at<Vec3b>(i, j)[0];
                        for (int m = 0; m < i + range; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(min, min, min);
                    }
                    //se está no canto superior direito 
                    else if (i > (input.cols - range) && j < range) {
                        int min = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(min, min, min);
                    }
                    //se está no canto inferior direito
                    else if (i > (input.cols - range) && j > (input.rows - range)) {
                        int min = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(min, min, min);
                    }
                    //se está no canto inferior esquerdo
                    else if (i < range && j > (input.rows - range)) {
                        int min = input.at<Vec3b>(i, j)[0];
                        for (int m = 0; m < i + range; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(min, min, min);
                    }
                    //se está na borda de cima
                    else if (j < range) {
                        int min = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(min, min, min);
                    }
                    //se está na borda da direita
                    else if (i > input.cols - range) {
                        int min = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(min, min, min);
                    }
                    //se está na borda de baixo
                    else if (j > input.rows - range) {
                        int min = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(min, min, min);
                    }
                    //se está na borda da esquerda
                    else {
                        int min = input.at<Vec3b>(i, j)[0];
                        for (int m = 0; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(min, min, min);
                    }
                }
                else {
                    int min = input.at<Vec3b>(i, j)[0];
                    for (int m = i - range; m < i + range; m++) {
                        for (int n = j - range; n < j + range; n++) {
                            if (min > input.at<Vec3b>(n, m)[0])
                                min = input.at<Vec3b>(n, m)[0];
                        }
                    }
                    output.at<Vec3b>(Point(i, j)) = Vec3b(min, min, min);                  
                }
            }
        }
    }

    void filtraMedia(Mat input, Mat output, int range) {  //não utilizarei range por enquanto
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {  // laços para percorrer a imagem
                // em cada pixel fazer:
                if (i < range || i > input.cols - range || j < range || j > input.rows - range) {
                    //se está no canto superior esquerdo
                    if (i < range && j < range) {
                        int min = input.at<Vec3b>(i, j)[0];
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                                else if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        int media = int((max + min)/2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    //se está no canto superior direito 
                    else if (i > (input.cols - range) && j < range) {
                        int min = input.at<Vec3b>(i, j)[0];
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                                else if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        int media = int((max + min)/2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    //se está no canto inferior direito
                    else if (i > (input.cols - range) && j > (input.rows - range)) {
                        int min = input.at<Vec3b>(i, j)[0];
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                                else if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        int media = int((max + min)/2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    //se está no canto inferior esquerdo
                    else if (i < range && j > (input.rows - range)) {
                        int min = input.at<Vec3b>(i, j)[0];
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                                else if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        int media = int((max + min)/2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    //se está na borda de cima
                    else if (j < range) {
                        int min = input.at<Vec3b>(i, j)[0];
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                                else if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        int media = int((max + min)/2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    //se está na borda da direita
                    else if (i > input.cols - range) {
                        int min = input.at<Vec3b>(i, j)[0];
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                                else if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        int media = int((max + min)/2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    //se está na borda de baixo
                    else if (j > input.rows - range) {
                        int min = input.at<Vec3b>(i, j)[0];
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                                else if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        int media = int((max + min)/2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    //se está na borda da esquerda
                    else {
                        int min = input.at<Vec3b>(i, j)[0];
                        int max = input.at<Vec3b>(i, j)[0];
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<Vec3b>(n, m)[0])
                                    min = input.at<Vec3b>(n, m)[0];
                                else if (max < input.at<Vec3b>(n, m)[0])
                                    max = input.at<Vec3b>(n, m)[0];
                            }
                        }
                        int media = int((max + min)/2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                }
                else {
                    int min = input.at<Vec3b>(i, j)[0];
                    int max = input.at<Vec3b>(i, j)[0];
                    for (int m = i - range; m < i + range; m++) {
                        for (int n = j - range; n < j + range; n++) {
                            if (min > input.at<Vec3b>(n, m)[0])
                                min = input.at<Vec3b>(n, m)[0];
                            else if (max < input.at<Vec3b>(n, m)[0])
                                max = input.at<Vec3b>(n, m)[0];
                        }
                    }
                    int media = int((max + min)/2);
                    output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);                
                }
            }
        }
    }
   
    void otsu(Mat input, Mat output, int lim_min, int lim_max) {
        int menor_lim = lim_min;
        int menor_variancia;

        for (int lim = lim_min; lim <= lim_max; lim++) {
            int fundo = 0;
            int objeto = 0;
            float media_fundo = 0.0;
            float media_objeto = 0.0;
            float peso_fundo;
            float peso_objeto;
            float variancia_fundo = 0.0;
            float variancia_objeto = 0.0;
            float variancia_total = 0.0;

            //binarização e preparação para pesos e médias
            for (int i = 0; i < input.cols; i++) {
                for (int j = 0; j < input.rows; j++) {
                    if (input.at<Vec3b>(i, j)[0] > lim) {
                        fundo++;
                        media_fundo += input.at<Vec3b>(i, j)[0];
                    } else {
                        objeto++;
                        media_objeto += input.at<Vec3b>(i, j)[0];
                    }
                }
            }

            //pesos
            float tamanho = float(input.cols * input.rows);
            peso_fundo = fundo/tamanho;
            peso_objeto = objeto/tamanho;
            //médias
            media_fundo /= fundo;
            media_objeto /= objeto;

            for (int i = 0; i < input.cols; i++) {
                for (int j = 0; j < input.rows; j++) {
                    if (input.at<Vec3b>(i, j)[0] > lim)
                        variancia_fundo += ((input.at<Vec3b>(i, j)[0] - media_fundo) * (input.at<Vec3b>(i, j)[0] - media_fundo));
                    else
                        variancia_objeto += ((input.at<Vec3b>(i, j)[0] - media_objeto) * (input.at<Vec3b>(i, j)[0] - media_objeto));
                }
            }

            variancia_total = variancia_fundo * peso_fundo + variancia_objeto * peso_objeto;

            printf("lim %d var_total %f\n", lim, variancia_total);

            //verifica se tem menor variância
            if (lim == lim_min)
                menor_variancia = variancia_total;
            else {
                if (menor_variancia > variancia_total) {
                    menor_variancia = variancia_total;
                    menor_lim = lim;
                }
            }
        }

        //atribui o limiar com menor variância
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                if (input.at<Vec3b>(i, j)[0] > menor_lim)
                    output.at<Vec3b>(Point(j, i)) = Vec3b(255, 255, 255);
                else
                    output.at<Vec3b>(Point(j, i)) = Vec3b(0, 0, 0);
            }
        }

        printf("lim menor var %d menor var %f\n", menor_lim, menor_variancia);

    }
};