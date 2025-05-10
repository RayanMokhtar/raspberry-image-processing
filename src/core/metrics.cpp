#include "metrics.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>

PerformanceMetrics::PerformanceMetrics() {
    // Initialisation des métriques pour tous les types
    metrics.resize(static_cast<int>(MetricType::DISPLAY) + 1);
}

PerformanceMetrics::~PerformanceMetrics() {
}

void PerformanceMetrics::startMeasurement(MetricType type) {
    std::lock_guard<std::mutex> lock(metricMutex);
    metrics[static_cast<int>(type)].startTime = std::chrono::high_resolution_clock::now();
}

double PerformanceMetrics::endMeasurement(MetricType type) {
    auto endTime = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(metricMutex);
    auto& metric = metrics[static_cast<int>(type)];
    double duration = std::chrono::duration<double, std::milli>(endTime - metric.startTime).count();
    metric.measurements.push_back(duration);
    return duration;
}

double PerformanceMetrics::getLastMeasurement(MetricType type) const {
    std::lock_guard<std::mutex> lock(metricMutex);
    const auto& measurements = metrics[static_cast<int>(type)].measurements;
    return measurements.empty() ? 0.0 : measurements.back();
}

double PerformanceMetrics::getAverageMeasurement(MetricType type) const {
    std::lock_guard<std::mutex> lock(metricMutex);
    const auto& measurements = metrics[static_cast<int>(type)].measurements;
    
    if (measurements.empty()) return 0.0;
    
    return std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
}

void PerformanceMetrics::getStatistics(MetricType type, double& min, double& max, double& avg, double& stdDev) const {
    std::lock_guard<std::mutex> lock(metricMutex);
    const auto& measurements = metrics[static_cast<int>(type)].measurements;
    
    if (measurements.empty()) {
        min = max = avg = stdDev = 0.0;
        return;
    }
    
    min = *std::min_element(measurements.begin(), measurements.end());
    max = *std::max_element(measurements.begin(), measurements.end());
    avg = std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
    
    double variance = 0.0;
    for (const auto& measurement : measurements) {
        variance += std::pow(measurement - avg, 2);
    }
    variance /= measurements.size();
    stdDev = std::sqrt(variance);
}

double PerformanceMetrics::calculateFPS() const {
    double avgProcessingTime = getAverageMeasurement(MetricType::TOTAL_PROCESSING);
    if (avgProcessingTime <= 0.0) return 0.0;
    
    return 1000.0 / avgProcessingTime; // 1000ms / temps de traitement moyen
}

void PerformanceMetrics::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    file << "Metric,Count,Min,Max,Average,StdDev,FPS\n";
    
    std::lock_guard<std::mutex> lock(metricMutex);
    
    for (int i = 0; i <= static_cast<int>(MetricType::DISPLAY); i++) {
        const auto& measurements = metrics[i].measurements;
        if (measurements.empty()) continue;
        
        double min, max, avg, stdDev;
        getStatistics(static_cast<MetricType>(i), min, max, avg, stdDev);
        
        std::string metricName;
        switch (static_cast<MetricType>(i)) {
            case MetricType::TOTAL_PROCESSING: metricName = "Total"; break;
            case MetricType::ACQUISITION: metricName = "Acquisition"; break;
            case MetricType::GRAYSCALE: metricName = "Grayscale"; break;
            case MetricType::EDGE_DETECTION: metricName = "Edge Detection"; break;
            case MetricType::BINARIZATION: metricName = "Binarization"; break;
            case MetricType::HOUGH_TRANSFORM: metricName = "Hough Transform"; break;
            case MetricType::DISPLAY: metricName = "Display"; break;
        }
        
        double fps = (i == static_cast<int>(MetricType::TOTAL_PROCESSING)) ? calculateFPS() : 0.0;
        
        file << metricName << ","
             << measurements.size() << ","
             << min << ","
             << max << ","
             << avg << ","
             << stdDev << ","
             << fps << "\n";
    }
    
    file.close();
}

void PerformanceMetrics::printReport() const {
    std::cout << "\n=== Performance Report ===\n";
    std::cout << std::fixed << std::setprecision(2);
    
    for (int i = 0; i <= static_cast<int>(MetricType::DISPLAY); i++) {
        const auto& measurements = metrics[i].measurements;
        if (measurements.empty()) continue;
        
        double min, max, avg, stdDev;
        getStatistics(static_cast<MetricType>(i), min, max, avg, stdDev);
        
        std::string metricName;
        switch (static_cast<MetricType>(i)) {
            case MetricType::TOTAL_PROCESSING: metricName = "Total Processing"; break;
            case MetricType::ACQUISITION: metricName = "Acquisition"; break;
            case MetricType::GRAYSCALE: metricName = "Grayscale"; break;
            case MetricType::EDGE_DETECTION: metricName = "Edge Detection"; break;
            case MetricType::BINARIZATION: metricName = "Binarization"; break;
            case MetricType::HOUGH_TRANSFORM: metricName = "Hough Transform"; break;
            case MetricType::DISPLAY: metricName = "Display"; break;
        }
        
        std::cout << metricName << ":\n";
        std::cout << "  Count:  " << measurements.size() << "\n";
        std::cout << "  Min:    " << min << " ms\n";
        std::cout << "  Max:    " << max << " ms\n";
        std::cout << "  Avg:    " << avg << " ms\n";
        std::cout << "  StdDev: " << stdDev << " ms\n";
        
        if (i == static_cast<int>(MetricType::TOTAL_PROCESSING)) {
            std::cout << "  FPS:    " << calculateFPS() << "\n";
        }
        
        std::cout << std::endl;
    }
}

void PerformanceMetrics::reset() {
    std::lock_guard<std::mutex> lock(metricMutex);
    for (auto& metric : metrics) {
        metric.measurements.clear();
    }
}

// Implémentation de ScopedTimer
ScopedTimer::ScopedTimer(PerformanceMetrics& metrics, MetricType type)
    : metrics(metrics), metricType(type) {
    metrics.startMeasurement(metricType);
}

ScopedTimer::~ScopedTimer() {
    metrics.endMeasurement(metricType);
}