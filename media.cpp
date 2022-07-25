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
    namedWindow("Media");
    imshow("Media", output);
    while (getWindowProperty("Media", WND_PROP_VISIBLE))
        waitKey(50);
    destroyAllWindows();
    return 0;
}