#ifndef METRICS_H
#define METRICS_H

#include <chrono>
#include <string>
#include <vector>
#include <mutex>

// Enumération pour les différents types de métriques
enum class MetricType {
    TOTAL_PROCESSING,
    ACQUISITION,
    GRAYSCALE,
    EDGE_DETECTION,
    BINARIZATION,
    HOUGH_TRANSFORM,
    DISPLAY
};

// Classe pour la mesure et l'analyse des performances
class PerformanceMetrics {
public:
    PerformanceMetrics();
    ~PerformanceMetrics();
    
    // Démarrer une mesure
    void startMeasurement(MetricType type);
    
    // Terminer une mesure et enregistrer le temps écoulé
    double endMeasurement(MetricType type);
    
    // Obtenir la dernière mesure pour un type donné
    double getLastMeasurement(MetricType type) const;
    
    // Obtenir la moyenne des mesures pour un type donné
    double getAverageMeasurement(MetricType type) const;
    
    // Obtenir les statistiques complètes (min, max, moyenne, écart-type)
    void getStatistics(MetricType type, double& min, double& max, double& avg, double& stdDev) const;
    
    // Calculer les FPS basés sur la durée totale de traitement
    double calculateFPS() const;
    
    // Enregistrer les mesures dans un fichier
    void saveToFile(const std::string& filename) const;
    
    // Afficher un rapport de performance dans la console
    void printReport() const;
    
    // Remise à zéro des mesures
    void reset();
    
private:
    struct MetricData {
        std::chrono::high_resolution_clock::time_point startTime;
        std::vector<double> measurements; // durées en millisecondes
    };
    
    std::vector<MetricData> metrics;
    mutable std::mutex metricMutex;
};

// Classe utilitaire pour mesurer automatiquement la durée d'un bloc de code
class ScopedTimer {
public:
    ScopedTimer(PerformanceMetrics& metrics, MetricType type);
    ~ScopedTimer();
    
private:
    PerformanceMetrics& metrics;
    MetricType metricType;
};

#endif // METRICS_H