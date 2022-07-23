#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    ImgOps imagem;
    Mat input = imread(argv[1]);
    Mat output = input.clone();
    int range = stoi(argv[2]);
    if(input.at<Vec3b>(1,1)[0] != input.at<Vec3b>(1,1)[1] || input.at<Vec3b>(1,1)[0] != input.at<Vec3b>(1,1)[2]) {
        cvtColor(input, output, COLOR_RGB2GRAY);
        input = output.clone();
    }
    imagem.filtraMax(input, output, range);
    imwrite("filtraMax.png", output);
    return 0;
}