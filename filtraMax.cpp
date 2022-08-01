#include <iostream>
#include <opencv2/opencv.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

ImgOps imagem;

int main(int argc, char *argv[]) {
    Mat input = imread(argv[1]);
    int range = 1;
    cvtColor(input, input, COLOR_BGR2GRAY);
    Mat output = input.clone();
    imagem.filtraMax(input, output, range);
    imagem.exibir(output);
    return 0;
}
