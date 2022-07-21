#include <iostream>
#include <opencv2/opencv.hpp>

#include "imgops.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    // Instanciar objeto ImgOps
    ImgOps imagem;

    // Variavel Mat armazena imagem de input passada como parametro no argv[1]
    Mat input = imread(argv[1]);  // recomendo jogar as imagens dentro da pasta build
                                  // e executar ./main lenna.png 2

    // Copia do input, vai ser usado pra armazenar output
    Mat output = input.clone();

    // Instanciar janela do OpenCV HighGUI
    namedWindow("Processamento Digital de Imagens", WINDOW_AUTOSIZE);

    // Mostra imagem de input e aguarda infinitamente o usuario apertar uma tecla
    imshow("Processamento Digital de Imagens", input);
    waitKey(0);

    // Converte o input do argv[2] pra inteiro
    int range = stoi(argv[2]);

    // Metodo pra calcular media, implementado no imgops.hpp
    imagem.media(input, output, range);

    // Grava imagem em arquivo
    imwrite("lenna-media.png", output);

    // Mostra output
    imshow("Processamento Digital de Imagens", output);
    waitKey(0);

    // Mata as janelas se alguma ainda estiver aberta
    destroyAllWindows();

    return 0;
}

/**
 * Acessar pixel em imagem RGB:
 * int azul = input.at<Vec3b>(x, y)[0];
 * int verde = input.at<Vec3b>(x, y)[1];
 * int vermelho = input.at<Vec3b>(x, y)[2];
 *
 * Acessar pixel em imagem preto e branco:
 * int cinza = input.at<uchar>(x, y);
 *
 * Converter RGB para BW:
 * cvtColor(input, output, COLOR_RGB2GRAY);
 *
 * Variaveis do tipo Mat, Vec3b e uchar não podem
 * assumir valor menor que 0 ou maior que 255
 */