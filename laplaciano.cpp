#include <iostream>
#include <opencv2/opencv.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    ImgOps imagem;
    Mat input = imread(argv[1]);
    cvtColor(input, input, COLOR_BGR2GRAY);
    Mat output = input.clone();
    imagem.laplaciano(input, output);
    imagem.exibir(output);
    return 0;
}
