#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    Mat input = imread(argv[1]);
    Mat output = input.clone();
    cvtColor(input, output, COLOR_RGB2GRAY);
    imwrite("cinza.png", output);
    return 0;
}