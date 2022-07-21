#include <iostream>

#include "imgops.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    ImgOps imagem;
    Mat input = imread(argv[1]);
    Mat output = input.clone();
    int range = stoi(argv[2]);
    imagem.media(input, output, range);
    imwrite("media.png", output);
    return 0;
}