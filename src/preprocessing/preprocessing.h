#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>

// Structure pour les résultats du détecteur de lignes
struct LineDetectionResult {
    std::vector<cv::Vec2f> lines; // Lignes détectées en format rho, theta
    std::vector<cv::Vec4i> lineSegments; // Segments de ligne en format x1, y1, x2, y2
    double processingTime;  // Temps de traitement en ms
};

// Méthodes de grayscale
void grayscaleFilter(const cv::Mat& inputImage, cv::Mat& outputImage, bool useMultithread = false);
void grayscaleFilterThreaded(const cv::Mat& inputImage, cv::Mat& outputImage, int numThreads);

// Méthodes de détection de contours
void sobelFilter(const cv::Mat& inputImage, cv::Mat& outputImage, bool useMultithread = false);
void sobelFilterThreaded(const cv::Mat& inputImage, cv::Mat& outputImage, int numThreads);
void cannyFilter(const cv::Mat& inputImage, cv::Mat& outputImage, double threshold1 = 50, double threshold2 = 150, bool useMultithread = false);

// Méthodes de binarisation
void binarize(const cv::Mat& inputImage, cv::Mat& outputImage, int threshold = 128);
void adaptiveBinarize(const cv::Mat& inputImage, cv::Mat& outputImage, int blockSize = 11, int C = 2);

// Transformées de Hough
LineDetectionResult houghTransform(const cv::Mat& inputImage, cv::Mat& outputImage, int threshold = 100, bool useBinarization = true, bool useMultithread = false);
LineDetectionResult houghTransformNoBinarization(const cv::Mat& inputImage, cv::Mat& outputImage, int threshold = 50, bool useMultithread = false);
LineDetectionResult probabilisticHoughTransform(const cv::Mat& inputImage, cv::Mat& outputImage, int threshold = 50, bool useMultithread = false);

// Méthodes utilitaires
void drawLines(cv::Mat& image, const std::vector<cv::Vec2f>& lines, const cv::Scalar& color = cv::Scalar(0, 0, 255));
void drawLineSegments(cv::Mat& image, const std::vector<cv::Vec4i>& lineSegments, const cv::Scalar& color = cv::Scalar(0, 0, 255));

#endif // PREPROCESSING_H