#include <iostream>
#include <opencv2/opencv.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

ImgOps imagem;

int main(int argc, char *argv[]) {
    Mat input = imread(argv[1]), output;
    cvtColor(input, input, COLOR_BGR2GRAY);
    output = input.clone();
    imagem.normalizarCinza(input, output);
    imagem.exibir(output);
    return 0;
}