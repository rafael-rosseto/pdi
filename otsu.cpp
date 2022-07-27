#include <iostream>
#include <opencv2/opencv.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

ImgOps imagem;

int main(int argc, char *argv[]) {
    Mat input = imread(argv[1]);
    Mat output = input.clone();
    int lim_min = stoi(argv[2]);
    int lim_max = stoi(argv[3]);
    if (input.at<Vec3b>(1, 1)[0] != input.at<Vec3b>(1, 1)[1] || input.at<Vec3b>(1, 1)[0] != input.at<Vec3b>(1, 1)[2]) {
        cvtColor(input, output, COLOR_RGB2GRAY);
        input = output.clone();
    }
    imagem.otsu(input, output, lim_min, lim_max);
    imagem.exibir(output);
    return 0;
}