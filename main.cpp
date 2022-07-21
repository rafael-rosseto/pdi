#include <iostream>
#include <opencv2/opencv.hpp>

#include "imgops.hpp"

using namespace cv;

int main(int argc, char* argv[]) {
    ImgOps imagem;
    Mat input = imread(argv[1]);
    cvtColor(input, input, COLOR_RGB2GRAY);
    Mat output = input.clone();
    imagem.normalizar(input, output);
    namedWindow("Processamento Digital de Imagens", WINDOW_AUTOSIZE);
    imshow("Processamento Digital de Imagens", input);
    waitKey(0);
    imshow("Processamento Digital de Imagens", output);
    waitKey(0);
    destroyAllWindows();
    return 0;
}