#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "imgops.hpp"

using namespace cv;
using namespace std;

ImgOps imagem;

vector<Vec2i> coord, remover;

void borda(Mat input, Mat output) {
    coord.clear();
    coord.resize(0);
    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.rows; j++) {
            if (input.at<uchar>(i, j) == 255) {
                bool aux = false;
                for (int ii = i - 1; ii <= i + 1; ii++) {
                    for (int jj = j - 1; jj <= j + 1; jj++) {
                        if (ii < 0 || jj < 0 || ii >= input.rows || jj >= input.cols)
                            continue;
                        if (input.at<uchar>(ii, jj) == 0) {
                            coord.push_back(Vec2i(i, j));
                            aux = true;
                        }
                        if (aux) break;
                    }
                    if (aux) break;
                }
            }
        }
    }
}

void zhangsuen(Mat input, Mat output) {
    remover.clear();
    remover.resize(0);
    int a = 0, b = 0, c = 0, d = 0;
    for (int x = 0; x < coord.size(); x++) {
        int sum = 0;
        for (int i = coord[x][0] - 1; i <= coord[x][0] + 1; i++) {
            for (int j = coord[x][1] - 1; j <= coord[x][1] + 1; j++) {
                if (i == coord[x][0] && j == coord[x][1]) continue;
                if (input.at<uchar>(i, j) == 255) sum++;
            }
        }
        if (2 <= sum && sum <= 6) a++;

        sum = 0;
        if ((input.at<uchar>((coord[x][0]), (coord[x][1] - 1)) == 0) &&
            (input.at<uchar>((coord[x][0] + 1), (coord[x][1] - 1)) == 1)) {
            sum++;
        }
        if ((input.at<uchar>((coord[x][0] + 1), (coord[x][1] - 1)) == 0) &&
            (input.at<uchar>((coord[x][0] + 1), (coord[x][1])) == 1)) {
            sum++;
        }
        if ((input.at<uchar>((coord[x][0] + 1), (coord[x][1])) == 0) &&
            (input.at<uchar>((coord[x][0] + 1), (coord[x][1] + 1)) == 1)) {
            sum++;
        }
        if ((input.at<uchar>((coord[x][0] + 1), (coord[x][1] + 1)) == 0) &&
            (input.at<uchar>((coord[x][0]), (coord[x][1] + 1)) == 1)) {
            sum++;
        }
        if ((input.at<uchar>((coord[x][0]), (coord[x][1] + 1)) == 0) &&
            (input.at<uchar>((coord[x][0] - 1), (coord[x][1] + 1)) == 1)) {
            sum++;
        }
        if ((input.at<uchar>((coord[x][0] - 1), (coord[x][1] + 1)) == 0) &&
            (input.at<uchar>((coord[x][0]), (coord[x][1] - 1)) == 1)) {
            sum++;
        }
        if ((input.at<uchar>((coord[x][0]), (coord[x][1] - 1)) == 0) &&
            (input.at<uchar>((coord[x][0] - 1), (coord[x][1] - 1)) == 1)) {
            sum++;
        }
        if ((input.at<uchar>((coord[x][0] - 1), (coord[x][1] - 1)) == 0) &&
            (input.at<uchar>((coord[x][0]), (coord[x][1] - 1)) == 1)) {
            sum++;
        }
        if (sum == 1) b++;

        if (input.at<uchar>((coord[x][0]), (coord[x][1] - 1)) *
                input.at<uchar>((coord[x][0] + 1), (coord[x][1])) *
                input.at<uchar>((coord[x][0]), (coord[x][1] + 1)) ==
            0) c++;

        if (input.at<uchar>((coord[x][0] + 1), (coord[x][1])) *
                input.at<uchar>((coord[x][0]), (coord[x][1] + 1)) *
                input.at<uchar>((coord[x][0] - 1), (coord[x][1])) ==
            0) d++;

        if (a && b && c && d) remover.push_back(coord[x]);
    }
    if (c) cout << (int)a << " " << (int)b << " " << (int)c << " " << (int)d << endl;
    for (int x = 0; x < remover.size(); x++) {
        output.at<uchar>((remover[x][0]), remover[x][1]) = 0;
    }
    input = output.clone();
}

int main(int argc, char* argv[]) {
    Mat input = imread(argv[1]);
    cvtColor(input, input, COLOR_BGR2GRAY);
    Mat output = input.clone();
    borda(input, output);
    // cout << coord.size() << endl;
    zhangsuen(input, output);
    // cout << remover.size() << endl;
    imagem.exibir(output);
    return 0;
}