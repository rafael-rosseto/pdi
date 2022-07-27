#include <iostream>
#include <opencv2/opencv.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

ImgOps imagem;

int main(int argc, char *argv[]) {
    Mat input = imread(argv[1]);
    imagem.exibir(input);
    cvtColor(input, input, COLOR_BGR2HLS);
    Mat output = input.clone();
    imagem.normalizarColorido(input, output);
    cvtColor(output, output, COLOR_HLS2BGR);
    imagem.exibir(output);
    return 0;
}