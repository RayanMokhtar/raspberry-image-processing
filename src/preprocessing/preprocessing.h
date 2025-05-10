#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

// Déclaration des fonctions
void grayscale_filter(cv::Mat* input_image, unsigned char* gray_vec);
void gradient_filter(unsigned char* gray_vec, unsigned char* gradient_vec, int rows, int cols);
void binarize(unsigned char* gradient_vec, unsigned char* bin_vec, int rows, int cols);
void hough_filter(unsigned char* bin_vec, cv::Mat* output_image, int rows, int cols);

#endif // IMAGE_PROCESSING_H