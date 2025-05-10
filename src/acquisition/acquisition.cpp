#include "opencv4/opencv2/opencv.hpp"
#include "../preprocessing/preprocessing.h"
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {
    //TODO centraliser les datas qu'on aura à tester directement et la manière dont on les acquiret ==> éventuellement avec un fichier de config ou des méthodes pour prendre en compte soit
    //caméra ou image et la métrique se fera par la suite comme ça 
    string image_path = "../../data/route_low.jpg";
    Mat image = imread(image_path);
    if (image.empty()) {
        cerr << "Erreur : Impossible de charger l'image" << endl;
        return -1;
    }

    imshow("Original", image);

    unsigned char* gray_vec = new unsigned char[image.rows * image.cols];
    unsigned char* grad_vec = new unsigned char[image.rows * image.cols];
    unsigned char* bin_vec  = new unsigned char[image.rows * image.cols];

    // Conversion en niveaux de gris
    grayscale_filter(&image, gray_vec);

    // Gradient Sobel
    gradient_filter(gray_vec, grad_vec, image.rows, image.cols);

    // Binarisation
    binarize(grad_vec, bin_vec, image.rows, image.cols);

    // Hough
    hough_filter(bin_vec, &image, image.rows, image.cols);
    cout << "Transformation de Hough terminée." << endl;

    // Afficher le résultat
    imshow("Résultat Hough", image);
    waitKey(0);

    // Libérer la mémoire
    delete[] gray_vec;
    delete[] grad_vec;
    delete[] bin_vec;

    return 0;
}