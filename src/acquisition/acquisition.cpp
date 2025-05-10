#include "acquisition.h"
#include <iostream>

ImageAcquisition::ImageAcquisition() : isInitialized(false) {
}

ImageAcquisition::~ImageAcquisition() {
    release();
}

bool ImageAcquisition::init(SourceType type, const std::string& source) {
    sourceType = type;
    
    if (type == SourceType::IMAGE_FILE) {
        if (source.empty()) {
            std::cerr << "Error: Image path is empty" << std::endl;
            return false;
        }
        
        staticImage = cv::imread(source);
        if (staticImage.empty()) {
            std::cerr << "Error: Could not load image from " << source << std::endl;
            return false;
        }
        
        imagePath = source;
        isInitialized = true;
        return true;
    } 
    else if (type == SourceType::CAMERA) {
        int cameraId = 0;
        if (!source.empty()) {
            try {
                cameraId = std::stoi(source);
            } catch (const std::exception& e) {
                std::cerr << "Error converting camera ID: " << e.what() << std::endl;
                return false;
            }
        }
        
        if (!camera.open(cameraId)) {
            std::cerr << "Error: Could not open camera " << cameraId << std::endl;
            return false;
        }
        
        // Vérifier si la caméra est ouverte correctement
        cv::Mat testFrame;
        if (!camera.read(testFrame) || testFrame.empty()) {
            std::cerr << "Error: Camera opened but cannot read frames" << std::endl;
            camera.release();
            return false;
        }
        
        isInitialized = true;
        return true;
    }
    
    return false;
}

bool ImageAcquisition::getFrame(cv::Mat& frame) {
    if (!isInitialized) {
        std::cerr << "Error: Acquisition not initialized" << std::endl;
        return false;
    }
    
    if (sourceType == SourceType::IMAGE_FILE) {
        staticImage.copyTo(frame);
        return !frame.empty();
    } 
    else if (sourceType == SourceType::CAMERA) {
        if (!camera.isOpened()) {
            std::cerr << "Error: Camera not opened" << std::endl;
            return false;
        }
        
        if (!camera.read(frame) || frame.empty()) {
            std::cerr << "Error: Could not read frame from camera" << std::endl;
            return false;
        }
        
        return true;
    }
    
    return false;
}

void ImageAcquisition::release() {
    if (sourceType == SourceType::CAMERA && camera.isOpened()) {
        camera.release();
    }
    
    if (!staticImage.empty()) {
        staticImage.release();
    }
    
    isInitialized = false;
}