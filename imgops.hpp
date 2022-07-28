#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#define pi 3.142857

using namespace cv;
using namespace std;

class ImgOps {
   private:
    int w, h;
    vector<vector<float>> dctdata;

   public:
    void exibir(Mat imagem) {
        namedWindow("Processamento Digital de Imagens", WINDOW_AUTOSIZE);
        imshow("Processamento Digital de Imagens", imagem);
        while (getWindowProperty("Processamento Digital de Imagens", WND_PROP_VISIBLE)) {
            char c = waitKeyEx(50);
            if (c == ' ') break;
        }
        destroyAllWindows();
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

    void binario(Mat src, Mat output, short alpha) {
        Mat input;
        cvtColor(src, input, COLOR_RGB2GRAY);
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                int aux = input.at<uchar>(i, j);
                if (aux >= alpha)
                    output.at<uchar>(i, j) = 255;
                else
                    output.at<uchar>(i, j) = 0;
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

    void normalizarCinza(Mat input, Mat output) {
        int max = 0, min = 255;
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.cols; j++) {
                if (input.at<uchar>(i, j) > max) max = input.at<uchar>(i, j);
                if (input.at<uchar>(i, j) < min) min = input.at<uchar>(i, j);
            }
        }
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.cols; j++) {
                output.at<uchar>(i, j) = (input.at<uchar>(i, j) - min) * 255 / (max - min);
            }
        }
    }

    void normalizarColorido(Mat input, Mat output) {
        int max = 0, min = 255;
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.cols; j++) {
                if (input.at<Vec3b>(i, j)[1] > max) max = input.at<Vec3b>(i, j)[1];
                if (input.at<Vec3b>(i, j)[1] < min) min = input.at<Vec3b>(i, j)[1];
            }
        }
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.cols; j++) {
                output.at<Vec3b>(i, j)[1] = (input.at<Vec3b>(i, j)[1] - min) * 255 / (max - min);
            }
        }
    }

    void normalizarDct(Mat output) {
        float max = numeric_limits<float>::min();
        float min = numeric_limits<float>::max();
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                if (dctdata[i][j] > max) max = dctdata[i][j];
                if (dctdata[i][j] < min) min = dctdata[i][j];
            }
        }
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                output.at<uchar>(i, j) = (dctdata[i][j] - min) * 255 / (max - min);
            }
        }
    }

    void dctTransform(Mat input) {
        w = input.cols;
        h = input.rows;
        dctdata.resize(h, vector<float>(w));
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
                output.at<uchar>(i, j) = sum;
            }
        }
    }

    void filtraMax(Mat input, Mat output, int range) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {  // laços para percorrer a imagem
                // em cada pixel fazer:
                if (i < range || i > input.cols - range || j < range || j > input.rows - range) {
                    // se está no canto superior esquerdo
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
                    // se está no canto superior direito
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
                    // se está no canto inferior direito
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
                    // se está no canto inferior esquerdo
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
                    // se está na borda de cima
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
                    // se está na borda da direita
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
                    // se está na borda de baixo
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
                    // se está na borda da esquerda
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
                } else {
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
                    // se está no canto superior esquerdo
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
                    // se está no canto superior direito
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
                    // se está no canto inferior direito
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
                    // se está no canto inferior esquerdo
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
                    // se está na borda de cima
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
                    // se está na borda da direita
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
                    // se está na borda de baixo
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
                    // se está na borda da esquerda
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
                } else {
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

    void filtraMedia(Mat input, Mat output, int range) {  // não utilizarei range por enquanto
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {  // laços para percorrer a imagem
                // em cada pixel fazer:
                if (i < range || i > input.cols - range || j < range || j > input.rows - range) {
                    // se está no canto superior esquerdo
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
                        int media = int((max + min) / 2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    // se está no canto superior direito
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
                        int media = int((max + min) / 2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    // se está no canto inferior direito
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
                        int media = int((max + min) / 2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    // se está no canto inferior esquerdo
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
                        int media = int((max + min) / 2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    // se está na borda de cima
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
                        int media = int((max + min) / 2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    // se está na borda da direita
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
                        int media = int((max + min) / 2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    // se está na borda de baixo
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
                        int media = int((max + min) / 2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                    // se está na borda da esquerda
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
                        int media = int((max + min) / 2);
                        output.at<Vec3b>(Point(i, j)) = Vec3b(media, media, media);
                    }
                } else {
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
                    int media = int((max + min) / 2);
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

            // binarização e preparação para pesos e médias
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

            // pesos
            float tamanho = float(input.cols * input.rows);
            peso_fundo = fundo / tamanho;
            peso_objeto = objeto / tamanho;
            // médias
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

            // verifica se tem menor variância
            if (lim == lim_min)
                menor_variancia = variancia_total;
            else {
                if (menor_variancia > variancia_total) {
                    menor_variancia = variancia_total;
                    menor_lim = lim;
                }
            }
        }

        // atribui o limiar com menor variância
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                if (input.at<Vec3b>(i, j)[0] > menor_lim)
                    output.at<Vec3b>(Point(j, i)) = Vec3b(255, 255, 255);
                else
                    output.at<Vec3b>(Point(j, i)) = Vec3b(0, 0, 0);
            }
        }
    }

    void pseudocores(Mat input, Mat output) {
        //definir intervalos
        int cor;
        int borda_sup = int(input.rows/4);
        int borda_inf = input.rows - borda_sup;

        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {

                if (j < borda_sup || j > borda_inf) { //para as bordas

                    cor = input.at<Vec3b>(j, i)[1];
                    if (cor >= 100) //no 130 deixa branco antártica e sibéria
                        output.at<Vec3b>(Point(i, j)) = Vec3b(170, 230, 230);
                    else if (cor <= 15) { // oceâno
                        cor *=  12; //12 de intervalo fica bom pra oceâno
                        if (cor <= 255)                        //(b, g, r)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(cor, 0, 0); //aumenta azul
                        else if (cor <= 510)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(255, cor - 255, 0); //aumenta verde
                        else if (cor <= 765)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(255 - (cor - 510), 255, 0); //diminui azul
                        else if (cor <= 1020)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(0, 255, cor - 765); //aumenta vermelho
                        else if (cor <= 1275)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(0, 255 - (cor - 1020), 255); //diminui azul
                        else output.at<Vec3b>(Point(i, j)) = Vec3b(0, 30, 0); //aumenta verde e azul
                    }
                    else {
                        cor *= 8;
                        if (cor <= 500)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(0, int(cor/5), 0);
                        else
                            output.at<Vec3b>(Point(i, j)) = Vec3b(0, int(cor/4), 0);
                    }
                }
                else { //para o centro
                    cor = input.at<Vec3b>(j, i)[1];
                    if (cor <= 15) { // oceâno
                        cor *=  12; //12 de intervalo fica bom pra oceâno
                        if (cor <= 255)                        //(x, y, z)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(cor, 0, 0); //aumenta azul
                        else if (cor <= 510)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(255, cor - 255, 0); //aumenta verde
                        else if (cor <= 765)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(255 - (cor - 510), 255, 0); //diminui azul
                        else if (cor <= 1020)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(0, 255, cor - 765); //aumenta vermelho
                        else if (cor <= 1275)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(0, 255 - (cor - 1020), 255); //diminui azul
                        else output.at<Vec3b>(Point(i, j)) = Vec3b(255, 255, 255); //aumenta verde e azul
                    }
                    else { //continentes do verde escuro ao verde-amarelado claro
                        cor *= 3;
                        if (cor <= 40)                        //(b, g, r)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(0, int(cor/2), 0); //parte escura
                        else if (cor <= 330)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(0, int(cor/1.5), 0); //parte clara
                        else if (cor <= 500)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(38, 200, 195); //parte amarelada
                        else if (cor <= 600)
                            output.at<Vec3b>(Point(i, j)) = Vec3b(60, 210, 210); //transição
                        else output.at<Vec3b>(Point(i, j)) = Vec3b(140, 230, 230); //neve
                    }
                }
            }
        }
    }

    
};