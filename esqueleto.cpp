#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "imgops.hpp"

#define P2 ((int)input.at<uchar>(i, j-1))
#define P3 ((int)input.at<uchar>(i+1, j-1))
#define P4 ((int)input.at<uchar>(i+1, j))
#define P5 ((int)input.at<uchar>(i+1, j+1))
#define P6 ((int)input.at<uchar>(i, j+1))
#define P7 ((int)input.at<uchar>(i-1, j+1))
#define P8 ((int)input.at<uchar>(i-1, j))
#define P9 ((int)input.at<uchar>(i-1, j-1))

using namespace cv;
using namespace std;

ImgOps imagem;

vector<Vec2i> remover;

void zhangsuen(Mat input, Mat output) {
    do {
        remover.clear();
        remover.resize(0);
        for (int i = 0; i < input.cols; i++) {
            for (int j = 0; j < input.rows; j++) {
                if (input.at<uchar>(i, j) == 255) {
                    bool a = false, b = false, c = false, d = false;
                    // Condição A
                    int sum = 0;
                    for (int ii = i - 1; ii <= i + 1; ii++) {
                        for (int jj = j - 1; jj <= j + 1; jj++) {
                            if (ii == i && jj == j) continue;
                            if (input.at<uchar>(ii, jj) == 255) sum++;
                        }
                    }
                    if (2 <= sum && sum <= 6) a = true;
                    // Condição B
                    sum = 0;
                    if (P2 == 0 && P3 == 255 && P4 == 0 && P5 == 255) sum++;
                    if (P3 == 0 && P4 == 255 && P5 == 0 && P6 == 255) sum++;
                    if (P4 == 0 && P5 == 255 && P6 == 0 && P7 == 255) sum++;
                    if (P5 == 0 && P6 == 255 && P7 == 0 && P8 == 255) sum++;
                    if (P6 == 0 && P7 == 255 && P8 == 0 && P9 == 255) sum++;
                    if (P7 == 0 && P8 == 255 && P9 == 0 && P2 == 255) sum++;
                    if (P8 == 0 && P9 == 255 && P2 == 0 && P3 == 255) sum++;
                    if (P9 == 0 && P2 == 255 && P3 == 0 && P4 == 255) sum++;
                    if (sum == 1) b = true;
                    // Condição C
                    if (P2 * P4 * P6 == 0) c = true;
                    // Condição D
                    if (P4 * P6 * P8 == 0) c = true;
                    // Marcar pra remover
                    if (a && b && c && d) remover.push_back(Vec2i(i, j));
                }
            }
        }
        // Remover pixels
        for (int i = 0; i < remover.size(); i++) {
            output.at<uchar>(remover[i][0], remover[i][1]) = 0;
        }
        input = output.clone();
    } while (remover.size() != 0);
}

int main(int argc, char* argv[]) {
    Mat input = imread(argv[1]);
    cvtColor(input, input, COLOR_BGR2GRAY);
    Mat output = input.clone();
    zhangsuen(input, output);
    imagem.exibir(output);
    return 0;
}
