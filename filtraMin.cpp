#include <iostream>
#include <opencv2/opencv.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

ImgOps imagem;

int main(int argc, char *argv[]) {
    Mat input = imread(argv[1]);
    Mat output = input.clone();
    int range = stoi(argv[2]);
    if (input.at<Vec3b>(1, 1)[0] != input.at<Vec3b>(1, 1)[1] || input.at<Vec3b>(1, 1)[0] != input.at<Vec3b>(1, 1)[2]) {
        cvtColor(input, output, COLOR_RGB2GRAY);
        input = output.clone();
    }
    imagem.filtraMin(input, output, range);
    imagem.exibir(output);
    return 0;
}