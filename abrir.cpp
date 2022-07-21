#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    try {
        Mat input = imread(argv[1]);
    } catch (...) {
        cout << "Erro de sintaxe" << endl;
    }
    Mat input = imread(argv[1]);
    namedWindow("Processamento Digital de Imagens", WINDOW_AUTOSIZE);
    imshow("Processamento Digital de Imagens", input);
    waitKey(0);
    destroyWindow("Processamento Digital de Imagens");
    return 0;
}