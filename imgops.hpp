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
                        int max = input.at<uchar>(i, j);
                        for (int m = 0; m < i + range; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        output.at<Vec3b>(Point(i, j)) = Vec3b(max, max, max);
                    }
                    // se está no canto superior direito
                    else if (i > (input.cols - range) && j < range) {
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = max;
                    }
                    // se está no canto inferior direito
                    else if (i > (input.cols - range) && j > (input.rows - range)) {
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = max;
                    }
                    // se está no canto inferior esquerdo
                    else if (i < range && j > (input.rows - range)) {
                        int max = input.at<uchar>(i, j);
                        for (int m = 0; m < i + range; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = max;
                    }
                    // se está na borda de cima
                    else if (j < range) {
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = max;
                    }
                    // se está na borda da direita
                    else if (i > input.cols - range) {
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = max;
                    }
                    // se está na borda de baixo
                    else if (j > input.rows - range) {
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = max;
                    }
                    // se está na borda da esquerda
                    else {
                        int max = input.at<uchar>(i, j);
                        for (int m = 0; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = max;
                    }
                } else {
                    int max = input.at<uchar>(i, j);
                    for (int m = i - range; m < i + range; m++) {
                        for (int n = j - range; n < j + range; n++) {
                            if (max < input.at<uchar>(m, n))
                                max = input.at<uchar>(m, n);
                        }
                    }
                    output.at<uchar>(i, j) = max;
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
                        int min = input.at<uchar>(i, j);
                        for (int m = 0; m < i + range; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = min;
                    }
                    // se está no canto superior direito
                    else if (i > (input.cols - range) && j < range) {
                        int min = input.at<uchar>(i, j);
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = min;
                    }
                    // se está no canto inferior direito
                    else if (i > (input.cols - range) && j > (input.rows - range)) {
                        int min = input.at<uchar>(i, j);
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = min;
                    }
                    // se está no canto inferior esquerdo
                    else if (i < range && j > (input.rows - range)) {
                        int min = input.at<uchar>(i, j);
                        for (int m = 0; m < i + range; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = min;
                    }
                    // se está na borda de cima
                    else if (j < range) {
                        int min = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = 0; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = min;
                    }
                    // se está na borda da direita
                    else if (i > input.cols - range) {
                        int min = input.at<uchar>(i, j);
                        for (int m = i - range; m < input.cols; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = min;
                    }
                    // se está na borda de baixo
                    else if (j > input.rows - range) {
                        int min = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < input.rows; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = min;
                    }
                    // se está na borda da esquerda
                    else {
                        int min = input.at<uchar>(i, j);
                        for (int m = 0; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                            }
                        }
                        output.at<uchar>(i, j) = min;
                    }
                } else {
                    int min = input.at<uchar>(i, j);
                    for (int m = i - range; m < i + range; m++) {
                        for (int n = j - range; n < j + range; n++) {
                            if (min > input.at<uchar>(m, n))
                                min = input.at<uchar>(m, n);
                        }
                    }
                    output.at<uchar>(i, j) = min;
                }
            }
        }
    }

    void filtraMedia(Mat input, Mat output, int range) {
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {  // laços para percorrer a imagem
                // em cada pixel fazer:
                if (i < range || i > input.cols - range || j < range || j > input.rows - range) {
                    // se está no canto superior esquerdo
                    if (i < range && j < range) {
                        int min = input.at<uchar>(i, j);
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                                else if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        int media = int((max + min) / 2);
                        output.at<uchar>(i, j) = media;
                    }
                    // se está no canto superior direito
                    else if (i > (input.cols - range) && j < range) {
                        int min = input.at<uchar>(i, j);
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                                else if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        int media = int((max + min) / 2);
                        output.at<uchar>(i, j) = media;
                    }
                    // se está no canto inferior direito
                    else if (i > (input.cols - range) && j > (input.rows - range)) {
                        int min = input.at<uchar>(i, j);
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                                else if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        int media = int((max + min) / 2);
                        output.at<uchar>(i, j) = media;
                    }
                    // se está no canto inferior esquerdo
                    else if (i < range && j > (input.rows - range)) {
                        int min = input.at<uchar>(i, j);
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                                else if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        int media = int((max + min) / 2);
                        output.at<uchar>(i, j) = media;
                    }
                    // se está na borda de cima
                    else if (j < range) {
                        int min = input.at<uchar>(i, j);
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                                else if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        int media = int((max + min) / 2);
                        output.at<uchar>(i, j) = media;
                    }
                    // se está na borda da direita
                    else if (i > input.cols - range) {
                        int min = input.at<uchar>(i, j);
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                                else if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        int media = int((max + min) / 2);
                        output.at<uchar>(i, j) = media;
                    }
                    // se está na borda de baixo
                    else if (j > input.rows - range) {
                        int min = input.at<uchar>(i, j);
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                                else if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        int media = int((max + min) / 2);
                        output.at<uchar>(i, j) = media;
                    }
                    // se está na borda da esquerda
                    else {
                        int min = input.at<uchar>(i, j);
                        int max = input.at<uchar>(i, j);
                        for (int m = i - range; m < i + range; m++) {
                            for (int n = j - range; n < j + range; n++) {
                                if (min > input.at<uchar>(m, n))
                                    min = input.at<uchar>(m, n);
                                else if (max < input.at<uchar>(m, n))
                                    max = input.at<uchar>(m, n);
                            }
                        }
                        int media = int((max + min) / 2);
                        output.at<uchar>(i, j) = media;
                    }
                } else {
                    int min = input.at<uchar>(i, j);
                    int max = input.at<uchar>(i, j);
                    for (int m = i - range; m < i + range; m++) {
                        for (int n = j - range; n < j + range; n++) {
                            if (min > input.at<uchar>(m, n))
                                min = input.at<uchar>(m, n);
                            else if (max < input.at<uchar>(m, n))
                                max = input.at<uchar>(m, n);
                        }
                    }
                    int media = int((max + min) / 2);
                    output.at<uchar>(i, j) = media;
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
                    if (input.at<uchar>(i, j) > lim) {
                        fundo++;
                        media_fundo += input.at<uchar>(i, j);
                    } else {
                        objeto++;
                        media_objeto += input.at<uchar>(i, j);
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
                    if (input.at<uchar>(i, j) > lim)
                        variancia_fundo += ((input.at<uchar>(i, j) - media_fundo) * (input.at<uchar>(i, j) - media_fundo));
                    else
                        variancia_objeto += ((input.at<uchar>(i, j) - media_objeto) * (input.at<uchar>(i, j) - media_objeto));
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
                if (input.at<uchar>(i, j) > menor_lim)
                    output.at<uchar>(i, j) = 255;
                else
                    output.at<uchar>(i, j) = 0;
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

    void dilatacao(Mat input, Mat output) {
        int masc[3][3] = {
            {0, 1, 0},
            {1, 1, 1},
            {0, 1, 0}};
        int cor;

        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                cor = input.at<uchar>(i, j);
                if (cor > 0) {
                    for (int ii = -1; ii <= 1; ii++) {
                        for (int jj = -1; jj <= 1; jj++) {
                            if (masc[ii + 1][jj + 1] == 1)
                                output.at<uchar>(i + ii, j + jj) = 255;
                        }
                    }
                }
            }
        }
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
                cor = input.at<uchar>(i, j);
                if (cor > 0) {
                    remove = false;
                    for (int ii = -1; ii <= 1; ii++) {
                        for (int jj = -1; jj <= 1; jj++) {
                            if (masc[ii + 1][jj + 1] == 1 && input.at<uchar>(i + ii, j + jj) == 0)
                                remove = true;
                            if (remove)
                                output.at<uchar>(i, j) = 0;
                            else
                                output.at<uchar>(i, j) = 255;
                        }
                    }
                }
            }
        }
    }

    void laplacGaussiano(Mat input, Mat output) {
        int masc[25] = {0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 1,
            2, -16, 2, 1, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0};
        int pos_masc = 0;
        int cont_masc = 0;

        for (int i = 2; i < input.cols - 2; i++) {
            for (int j = 2; j < input.rows - 2; j++) {
                cont_masc = 0;
                for (int ii = -2; ii <= 2; ii++) {
                    for (int jj = -2; jj <= 2; jj++, cont_masc++) {
                        pos_masc += input.at<uchar>(i + ii, j + jj) * masc[cont_masc];
                    }
                }
                pos_masc /= 25;
                output.at<uchar>(i, j) = pos_masc;
            }
        }
    }

    void laplaciano(Mat input, Mat output) {
        int masc[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
        int pos_masc = 0;
        int cont_masc = 0;

        for (int i = 1; i < input.cols - 1; i++) {
            for (int j = 1; j < input.rows - 1; j++) {
                cont_masc = 0;
                for (int ii = -1; ii <= 1; ii++) {
                    for (int jj = -1; jj <= 1; jj++, cont_masc++) {
                        pos_masc += input.at<uchar>(i + ii, j + jj) * masc[cont_masc];
                    }
                }
                pos_masc /= 9;
                output.at<uchar>(i, j) = pos_masc;
            }
        }
    }
};
