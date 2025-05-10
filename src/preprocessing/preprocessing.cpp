#include "preprocessing.h"
#include <chrono>
#include <algorithm>
#include <future>

// Fonction de conversion en niveaux de gris
void grayscaleFilter(const cv::Mat& inputImage, cv::Mat& outputImage, bool useMultithread) {
    if (inputImage.channels() == 1) {
        inputImage.copyTo(outputImage);
        return;
    }
    
    if (useMultithread) {
        int numThreads = std::thread::hardware_concurrency();
        grayscaleFilterThreaded(inputImage, outputImage, numThreads);
    } else {
        cv::cvtColor(inputImage, outputImage, cv::COLOR_BGR2GRAY);
    }
}

// Implémentation multi-thread de grayscale
void grayscaleFilterThreaded(const cv::Mat& inputImage, cv::Mat& outputImage, int numThreads) {
    outputImage = cv::Mat(inputImage.size(), CV_8UC1);
    
    auto processRows = [&](int startRow, int endRow) {
        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < inputImage.cols; j++) {
                cv::Vec3b pixel = inputImage.at<cv::Vec3b>(i, j);
                outputImage.at<uchar>(i, j) = static_cast<uchar>(
                    0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            }
        }
    };
    
    std::vector<std::thread> threads;
    int rowsPerThread = inputImage.rows / numThreads;
    
    for (int t = 0; t < numThreads; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? inputImage.rows : (t + 1) * rowsPerThread;
        threads.push_back(std::thread(processRows, startRow, endRow));
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

// Filtre de Sobel
void sobelFilter(const cv::Mat& inputImage, cv::Mat& outputImage, bool useMultithread) {
    cv::Mat gradientX, gradientY, absGradientX, absGradientY;
    
    if (useMultithread) {
        auto sobelX = std::async(std::launch::async, [&]() {
            cv::Sobel(inputImage, gradientX, CV_16S, 1, 0, 3);
            cv::convertScaleAbs(gradientX, absGradientX);
        });
        
        auto sobelY = std::async(std::launch::async, [&]() {
            cv::Sobel(inputImage, gradientY, CV_16S, 0, 1, 3);
            cv::convertScaleAbs(gradientY, absGradientY);
        });
        
        sobelX.wait();
        sobelY.wait();
    } else {
        cv::Sobel(inputImage, gradientX, CV_16S, 1, 0, 3);
        cv::convertScaleAbs(gradientX, absGradientX);
        
        cv::Sobel(inputImage, gradientY, CV_16S, 0, 1, 3);
        cv::convertScaleAbs(gradientY, absGradientY);
    }
    
    cv::addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0, outputImage);
}

// Version threadée de Sobel pour des images plus grandes
void sobelFilterThreaded(const cv::Mat& inputImage, cv::Mat& outputImage, int numThreads) {
    cv::Mat grayImage;
    if (inputImage.channels() > 1) {
        grayscaleFilter(inputImage, grayImage, true);
    } else {
        grayImage = inputImage;
    }
    
    outputImage = cv::Mat(grayImage.size(), CV_8UC1);
    
    auto processSobel = [&](int startRow, int endRow) {
        for (int i = startRow; i < endRow; i++) {
            for (int j = 1; j < grayImage.cols - 1; j++) {
                if (i == 0 || i == grayImage.rows - 1) {
                    outputImage.at<uchar>(i, j) = 0;
                    continue;
                }
                
                // Calcul du gradient de Sobel manuellement
                int gx = -grayImage.at<uchar>(i-1, j-1) - 2*grayImage.at<uchar>(i, j-1) - grayImage.at<uchar>(i+1, j-1) +
                          grayImage.at<uchar>(i-1, j+1) + 2*grayImage.at<uchar>(i, j+1) + grayImage.at<uchar>(i+1, j+1);
                
                int gy = -grayImage.at<uchar>(i-1, j-1) - 2*grayImage.at<uchar>(i-1, j) - grayImage.at<uchar>(i-1, j+1) +
                          grayImage.at<uchar>(i+1, j-1) + 2*grayImage.at<uchar>(i+1, j) + grayImage.at<uchar>(i+1, j+1);
                
                int magnitude = std::sqrt(gx * gx + gy * gy);
                outputImage.at<uchar>(i, j) = magnitude > 255 ? 255 : static_cast<uchar>(magnitude);
            }
        }
    };
    
    std::vector<std::thread> threads;
    int rowsPerThread = grayImage.rows / numThreads;
    
    for (int t = 0; t < numThreads; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? grayImage.rows : (t + 1) * rowsPerThread;
        threads.push_back(std::thread(processSobel, startRow, endRow));
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

// Filtre de Canny
void cannyFilter(const cv::Mat& inputImage, cv::Mat& outputImage, double threshold1, double threshold2, bool useMultithread) {
    cv::Mat grayImage;
    
    if (inputImage.channels() > 1) {
        grayscaleFilter(inputImage, grayImage, useMultithread);
    } else {
        grayImage = inputImage;
    }
    
    // Pour Canny, on utilise l'implémentation OpenCV car elle est déjà optimisée
    cv::Canny(grayImage, outputImage, threshold1, threshold2);
}

// Binarisation simple
void binarize(const cv::Mat& inputImage, cv::Mat& outputImage, int threshold) {
    cv::threshold(inputImage, outputImage, threshold, 255, cv::THRESH_BINARY);
}

// Binarisation adaptative
void adaptiveBinarize(const cv::Mat& inputImage, cv::Mat& outputImage, int blockSize, int C) {
    cv::adaptiveThreshold(inputImage, outputImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                          cv::THRESH_BINARY, blockSize, C);
}

// Transformée de Hough standard avec binarisation
LineDetectionResult houghTransform(const cv::Mat& inputImage, cv::Mat& outputImage, int threshold, bool useBinarization, bool useMultithread) {
    auto startTime = std::chrono::high_resolution_clock::now();
    LineDetectionResult result;
    
    cv::Mat edges = inputImage.clone();
    if (inputImage.channels() > 1) {
        grayscaleFilter(inputImage, edges, useMultithread);
    }
    
    // Appliquer Sobel pour détection des contours
    sobelFilter(edges, edges, useMultithread);
    
    // Appliquer la binarisation si demandé
    if (useBinarization) {
        binarize(edges, edges, 100);
    }
    
    // Détection des lignes avec Hough
    cv::HoughLines(edges, result.lines, 1, CV_PI/180, threshold);
    
    // Copier l'image d'entrée vers la sortie si nécessaire
    if (inputImage.data != outputImage.data) {
        inputImage.copyTo(outputImage);
    }
    
    // Dessiner les lignes détectées
    drawLines(outputImage, result.lines);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    return result;
}

// Transformée de Hough sans binarisation
LineDetectionResult houghTransformNoBinarization(const cv::Mat& inputImage, cv::Mat& outputImage, int threshold, bool useMultithread) {
    return houghTransform(inputImage, outputImage, threshold, false, useMultithread);
}

// Transformée de Hough probabiliste
LineDetectionResult probabilisticHoughTransform(const cv::Mat& inputImage, cv::Mat& outputImage, int threshold, bool useMultithread) {
    auto startTime = std::chrono::high_resolution_clock::now();
    LineDetectionResult result;
    
    cv::Mat edges = inputImage.clone();
    if (inputImage.channels() > 1) {
        grayscaleFilter(inputImage, edges, useMultithread);
    }
    
    // Appliquer Canny pour détection des contours
    cannyFilter(edges, edges, 50, 150, useMultithread);
    
    // Détection des segments de lignes avec Hough probabiliste
    cv::HoughLinesP(edges, result.lineSegments, 1, CV_PI/180, threshold, 50, 10);
    
    // Copier l'image d'entrée vers la sortie si nécessaire
    if (inputImage.data != outputImage.data) {
        inputImage.copyTo(outputImage);
    }
    
    // Dessiner les segments de lignes détectés
    drawLineSegments(outputImage, result.lineSegments);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    return result;
}

// Dessiner les lignes détectées par HoughLines
void drawLines(cv::Mat& image, const std::vector<cv::Vec2f>& lines, const cv::Scalar& color) {
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        cv::line(image, pt1, pt2, color, 2, cv::LINE_AA);
    }
}

// Dessiner les segments de lignes détectés par HoughLinesP
void drawLineSegments(cv::Mat& image, const std::vector<cv::Vec4i>& lineSegments, const cv::Scalar& color) {
    for (size_t i = 0; i < lineSegments.size(); i++) {
        cv::line(image, cv::Point(lineSegments[i][0], lineSegments[i][1]),
                cv::Point(lineSegments[i][2], lineSegments[i][3]), color, 2, cv::LINE_AA);
    }
}