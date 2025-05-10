#include "preprocessing.h"
#include "opencv4/opencv2/opencv.hpp"
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

// Fonction 1 : Conversion en niveaux de gris
void grayscale_filter(Mat* input_image, unsigned char* gray_vec) {
    for (int i = 0; i < input_image->rows; i++) {
        for (int j = 0; j < input_image->cols; j++) {
            Vec3b pixel = input_image->at<Vec3b>(i, j);
            gray_vec[i * input_image->cols + j] = static_cast<unsigned char>(
                0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
        }
    }
}


// Fonction 2 : Filtre de gradient (Sobel)
void gradient_filter(unsigned char* gray_vec, unsigned char* gradient_vec, int rows, int cols) {
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int sobel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int gx = 0, gy = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int pixel = gray_vec[(i + k) * cols + (j + l)];
                    gx += sobel_x[k + 1][l + 1] * pixel;
                    gy += sobel_y[k + 1][l + 1] * pixel;
                }
            }
            gradient_vec[i * cols + j] = static_cast<unsigned char>(
                sqrt(gx * gx + gy * gy) / 4);
        }
    }
}

// Fonction 3 : Binarisation
void binarize(unsigned char* gradient_vec, unsigned char* bin_vec, int rows, int cols) {
    const unsigned char threshold = 128;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            bin_vec[i * cols + j] = (gradient_vec[i * cols + j] > threshold) ? 255 : 0;
        }
    }
}

// Fonction 4 : Transformation de Hough
void hough_filter(unsigned char* bin_vec, Mat* output_image, int rows, int cols) {
    int max_dist = sqrt(rows * rows + cols * cols);
    int accumulator[180][max_dist] = {0};

    // Accumulateur pour la transformation de Hough
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (bin_vec[i * cols + j] == 255) {
                for (int theta = 0; theta < 180; theta++) {
                    double rad = theta * CV_PI / 180.0;
                    int r = static_cast<int>(i * cos(rad) + j * sin(rad));
                    if (r >= 0 && r < max_dist) {
                        accumulator[theta][r]++;
                    }
                }
            }
        }
    }

    // Dessiner les lignes détectées
    for (int theta = 0; theta < 180; theta++) {
        for (int r = 0; r < max_dist; r++) {
            if (accumulator[theta][r] > 100) { // Seuil arbitraire
                double rad = theta * CV_PI / 180.0;
                Point pt1, pt2;
                if (sin(rad) != 0) {
                    pt1 = Point(0, r / sin(rad));
                    pt2 = Point(cols, (r - cols * cos(rad)) / sin(rad));
                } else {
                    pt1 = Point(r / cos(rad), 0);
                    pt2 = Point(r / cos(rad), rows);
                }
                line(*output_image, pt1, pt2, Scalar(0, 0, 255), 1);
            }
        }
    }
}