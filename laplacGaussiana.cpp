#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

void laplacGaussiano(Mat input, Mat output);

int main(int argc, char *argv[]) {
    ImgOps imagem;
    Mat input = imread(argv[1]);
    Mat output = input.clone();
    laplacGaussiano(input, output);
    imwrite("laplacGaussiano.png", output);
    return 0;
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
                    pos_masc += input.at<Vec3b>(i + ii, j + jj)[1] * masc[cont_masc];
                }
            }
            pos_masc /= 25;
            output.at<Vec3b>(Point(j, i)) = Vec3b(pos_masc, pos_masc, pos_masc);
        }
    }
}