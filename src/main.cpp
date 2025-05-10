#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "acquisition/acquisition.h"
#include "preprocessing/preprocessing.h"
#include "core/metrics.h"

// Définition des paramètres configurables
#define USE_CAMERA false           // true = utiliser la caméra, false = utiliser une image
#define IMAGE_PATH "../../data/route_low.jpg"  // Chemin de l'image à utiliser
#define CAMERA_ID 0                // ID de la caméra (généralement 0 pour la caméra par défaut)

#define USE_BINARIZATION true      // true = utiliser la binarisation, false = non
#define HOUGH_THRESHOLD 100        // Seuil pour la transformée de Hough (50-150 typiquement)
#define USE_PROBABILISTIC_HOUGH false // true = Hough probabiliste, false = Hough standard

#define USE_MULTITHREADING true    // true = utiliser multithreading pour les traitements
#define THREAD_COUNT 4             // Nombre de threads à utiliser

#define SHOW_DEBUG_WINDOWS true    // Afficher les fenêtres intermédiaires (grayscale, edges)
#define RUN_TIME_SECONDS 0         // Durée d'exécution en secondes (0 = infinie)

int main(int argc, char** argv) {
    std::cout << "Démarrage du détecteur de lignes..." << std::endl;
    
    // Initialisation des métriques
    PerformanceMetrics metrics;
    
    // Initialisation de l'acquisition
    ImageAcquisition acquisition;
    bool initSuccess = false;
    
    if (USE_CAMERA) {
        std::cout << "Initialisation de la caméra " << CAMERA_ID << "..." << std::endl;
        initSuccess = acquisition.init(ImageAcquisition::SourceType::CAMERA, std::to_string(CAMERA_ID));
    } else {
        std::cout << "Chargement de l'image " << IMAGE_PATH << "..." << std::endl;
        initSuccess = acquisition.init(ImageAcquisition::SourceType::IMAGE_FILE, IMAGE_PATH);
    }
    
    if (!initSuccess) {
        std::cerr << "Échec de l'initialisation de la source d'image." << std::endl;
        return 1;
    }
    
    cv::Mat frame, grayscale, edges, result;
    bool running = true;
    int frameCount = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "Traitement en cours..." << std::endl;
    
    while (running) {
        // Mesurer le temps total
        metrics.startMeasurement(MetricType::TOTAL_PROCESSING);
        
        // Acquisition d'image
        {
            ScopedTimer timer(metrics, MetricType::ACQUISITION);
            if (!acquisition.getFrame(frame)) {
                std::cerr << "Échec de l'acquisition d'image" << std::endl;
                break;
            }
            result = frame.clone(); // Copie pour afficher les résultats
        }
        
        // Conversion en niveaux de gris
        {
            ScopedTimer timer(metrics, MetricType::GRAYSCALE);
            grayscaleFilter(frame, grayscale, USE_MULTITHREADING);
            // Application d'un flou gaussien pour réduire le bruit
            cv::GaussianBlur(grayscale, grayscale, cv::Size(5, 5), 1.5, 1.5);
        }
        
        // Détection de contours
        {
            ScopedTimer timer(metrics, MetricType::EDGE_DETECTION);
            sobelFilter(grayscale, edges, USE_MULTITHREADING);
        }
        
        // Binarisation (si activée)
        if (USE_BINARIZATION) {
            ScopedTimer timer(metrics, MetricType::BINARIZATION);
            binarize(edges, edges, 50); // Seuil de binarisation à 50
        }
        
        // Transformée de Hough
        LineDetectionResult houghResult;
        {
            ScopedTimer timer(metrics, MetricType::HOUGH_TRANSFORM);
            if (USE_PROBABILISTIC_HOUGH) {
                houghResult = probabilisticHoughTransform(edges, result, HOUGH_THRESHOLD, USE_MULTITHREADING);
            } else {
                houghResult = houghTransform(edges, result, HOUGH_THRESHOLD, USE_BINARIZATION, USE_MULTITHREADING);
            }
        }
        
        // Affichage
        {
            ScopedTimer timer(metrics, MetricType::DISPLAY);
            
            // Afficher les FPS sur l'image
            double fps = metrics.calculateFPS();
            std::string fpsText = "FPS: " + std::to_string(fps);
            cv::putText(result, fpsText, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            std::string timeText = "Temps: " + std::to_string(metrics.getLastMeasurement(MetricType::TOTAL_PROCESSING)) + " ms";
            cv::putText(result, timeText, cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            // Affichage des fenêtres de debug
            if (SHOW_DEBUG_WINDOWS) {
                cv::imshow("Grayscale", grayscale);
                cv::imshow("Edges", edges);
            }
            
            cv::imshow("Résultat", result);
            
            int key = cv::waitKey(1);
            if (key == 27) { // Échap pour quitter
                running = false;
            }
        }
        
        // Finaliser la mesure du temps total
        metrics.endMeasurement(MetricType::TOTAL_PROCESSING);
        
        // Incrémenter le compteur de frames
        frameCount++;
        
        // Vérifier si la durée d'exécution est écoulée (si spécifiée)
        if (RUN_TIME_SECONDS > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            if (duration >= RUN_TIME_SECONDS) {
                running = false;
            }
        }
        
        // Si on traite une seule image (pas en mode caméra), sortir après une frame
        if (!USE_CAMERA) {
            cv::waitKey(0);  // Attendre qu'une touche soit pressée avant de quitter
            running = false;
        }
    }
    
    // Afficher les résultats de performance
    auto endTime = std::chrono::high_resolution_clock::now();
    double totalRunTime = std::chrono::duration<double>(endTime - startTime).count();
    
    std::cout << "\n=== Résumé des performances ===" << std::endl;
    std::cout << "Images traitées: " << frameCount << std::endl;
    std::cout << "Durée totale: " << totalRunTime << " s" << std::endl;
    std::cout << "FPS moyen: " << frameCount / totalRunTime << std::endl;
    
    metrics.printReport();
    
    // Sauvegarder les métriques dans un fichier
    std::string metricsFile = "performance_metrics.csv";
    metrics.saveToFile(metricsFile);
    std::cout << "Métriques sauvegardées dans " << metricsFile << std::endl;
    
    // Libérer les ressources
    acquisition.release();
    cv::destroyAllWindows();
    
    return 0;
}