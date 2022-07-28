#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

void laplaciano(Mat input, Mat output);

int main(int argc, char *argv[]) {
    ImgOps imagem;
    Mat input = imread(argv[1]);
    Mat output = input.clone();
    laplaciano(input, output);
    imwrite("laplaciano.png", output);
    return 0;
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
                    pos_masc += input.at<Vec3b>(i + ii, j + jj)[1] * masc[cont_masc];
                }
            }
            pos_masc /= 9;
            output.at<Vec3b>(Point(j, i)) = Vec3b(pos_masc, pos_masc, pos_masc);
        }
    }
}