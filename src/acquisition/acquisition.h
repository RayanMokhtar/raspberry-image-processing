#ifndef ACQUISITION_H
#define ACQUISITION_H

#include <opencv2/opencv.hpp>
#include <string>

class ImageAcquisition {
public:
    enum class SourceType {
        IMAGE_FILE,
        CAMERA
    };

    ImageAcquisition();
    ~ImageAcquisition();

    // Initialise la source (fichier image ou caméra)
    bool init(SourceType type, const std::string& source = "");
    
    // Capture une frame (à partir de l'image ou de la caméra)
    bool getFrame(cv::Mat& frame);
    
    // Libère les ressources
    void release();

private:
    SourceType sourceType;
    cv::Mat staticImage;
    cv::VideoCapture camera;
    std::string imagePath;
    bool isInitialized;
};

#endif // ACQUISITION_H