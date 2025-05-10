#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

#include "acquisition/acquisition.h"
#include "preprocessing/preprocessing.h"
#include "core/metrics.h"
#include "core/thread_pool.h"

// Paramètres du programme
struct ProgramSettings {
    std::string inputSource;
    bool useCamera;
    bool useBinarization;
    bool useMultithreading;
    int numThreads;
    int houghThreshold;
    bool useHoughProbabilistic;
    bool showDebugWindows;
    bool testPerformance;
    int stressLevel; // Niveau de stress pour les tests (0 = aucun)
    int runTime;     // Durée d'exécution en secondes (0 = infinie)
};

// Configuration par défaut
ProgramSettings defaultSettings() {
    return {
        "../../data/route_low.jpg", // Source d'entrée
        false,                      // Utiliser la caméra
        true,                       // Utiliser la binarisation
        true,                       // Utiliser le multithreading
        std::thread::hardware_concurrency(), // Nombre de threads
        100,                        // Seuil Hough
        false,                      // Utiliser Hough probabiliste
        true,                       // Afficher fenêtres de débogage
        false,                      // Mode test de performance
        0,                          // Niveau de stress
        0                           // Durée d'exécution (0 = infinie)
    };
}

// Parses command line arguments and updates settings
void parseArgs(int argc, char** argv, ProgramSettings& settings) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--camera" || arg == "-c") {
            settings.useCamera = true;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                settings.inputSource = argv[++i];
            } else {
                settings.inputSource = "0"; // Default camera
            }
        }
        else if (arg == "--image" || arg == "-i") {
            settings.useCamera = false;
            if (i + 1 < argc) {
                settings.inputSource = argv[++i];
            }
        }
        else if (arg == "--no-binarization") {
            settings.useBinarization = false;
        }
        else if (arg == "--no-multithread") {
            settings.useMultithreading = false;
        }
        else if (arg == "--threads") {
            if (i + 1 < argc) {
                settings.numThreads = std::stoi(argv[++i]);
            }
        }
        else if (arg == "--threshold") {
            if (i + 1 < argc) {
                settings.houghThreshold = std::stoi(argv[++i]);
            }
        }
        else if (arg == "--probabilistic") {
            settings.useHoughProbabilistic = true;
        }
        else if (arg == "--no-debug") {
            settings.showDebugWindows = false;
        }
        else if (arg == "--test") {
            settings.testPerformance = true;
        }
        else if (arg == "--stress") {
            if (i + 1 < argc) {
                settings.stressLevel = std::stoi(argv[++i]);
            } else {
                settings.stressLevel = 5; // Default stress level
            }
        }
        else if (arg == "--time") {
            if (i + 1 < argc) {
                settings.runTime = std::stoi(argv[++i]);
            }
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -c, --camera [id]     Use camera (default: 0)\n"
                      << "  -i, --image <path>    Use image file\n"
                      << "  --no-binarization     Disable binarization\n"
                      << "  --no-multithread      Disable multithreading\n"
                      << "  --threads <n>         Set number of threads\n"
                      << "  --threshold <n>       Set Hough threshold\n"
                      << "  --probabilistic       Use probabilistic Hough\n"
                      << "  --no-debug            Hide debug windows\n"
                      << "  --test                Run performance tests\n"
                      << "  --stress <level>      Add CPU stress (0-10)\n"
                      << "  --time <seconds>      Run duration in seconds\n"
                      << "  -h, --help            Show this help\n";
            exit(0);
        }
    }
}

// Fonction pour simuler une charge CPU (pour les tests de stress)
void stressCPU(int level) {
    if (level <= 0) return;
    
    // Créer des threads pour stresser le CPU
    int numThreads = level;
    std::vector<std::thread> stressThreads;
    
    for (int i = 0; i < numThreads; i++) {
        stressThreads.emplace_back([i]() {
            auto start = std::chrono::high_resolution_clock::now();
            while (std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - start).count() < 600) {
                // Calculer des racines carrées pour stresser le CPU
                for (volatile int j = 0; j < 100000; j++) {
                    volatile double x = std::sqrt(j * 123.456);
                    (void)x;
                }
            }
        });
    }
    
    // Détacher les threads pour qu'ils s'exécutent en arrière-plan
    for (auto& t : stressThreads) {
        t.detach();
    }
}

int main(int argc, char** argv) {
    // Configuration du programme
    ProgramSettings settings = defaultSettings();
    parseArgs(argc, argv, settings);
    
    // Initialisation des métriques
    PerformanceMetrics metrics;
    
    // Initialisation de l'acquisition
    ImageAcquisition acquisition;
    if (!acquisition.init(
        settings.useCamera ? ImageAcquisition::SourceType::CAMERA : ImageAcquisition::SourceType::IMAGE_FILE,
        settings.inputSource)) {
        std::cerr << "Failed to initialize acquisition from source: " << settings.inputSource << std::endl;
        return 1;
    }
    
    // Appliquer la charge de stress si demandée
    if (settings.stressLevel > 0) {
        std::cout << "Applying CPU stress level " << settings.stressLevel << std::endl;
        stressCPU(settings.stressLevel);
    }
    
    cv::Mat frame, grayscale, edges, result;
    bool running = true;
    int frameCount = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (running) {
        // Mesurer le temps total
        metrics.startMeasurement(MetricType::TOTAL_PROCESSING);
        
        // Acquisition d'image
        {
            ScopedTimer timer(metrics, MetricType::ACQUISITION);
            if (!acquisition.getFrame(frame)) {
                std::cerr << "Failed to acquire frame" << std::endl;
                break;
            }
        }
        
        // Prétraitement (grayscale)
        {
            ScopedTimer timer(metrics, MetricType::GRAYSCALE);
            grayscaleFilter(frame, grayscale, settings.useMultithreading);
        }
        
        // Détection de contours
        {
            ScopedTimer timer(metrics, MetricType::EDGE_DETECTION);
            sobelFilter(grayscale, edges, settings.useMultithreading);
        }
        
        // Binarisation (si activée)
        if (settings.useBinarization) {
            ScopedTimer timer(metrics, MetricType::BINARIZATION);
            binarize(edges, edges, 100);
        }
        
        // Transformée de Hough
        LineDetectionResult houghResult;
        {
            ScopedTimer timer(metrics, MetricType::HOUGH_TRANSFORM);
            if (settings.useHoughProbabilistic) {
                houghResult = probabilisticHoughTransform(edges, frame, settings.houghThreshold, settings.useMultithreading);
            } else {
                houghResult = houghTransform(edges, frame, settings.houghThreshold, settings.useBinarization, settings.useMultithreading);
            }
        }
        
        // Affichage
        {
            ScopedTimer timer(metrics, MetricType::DISPLAY);
            if (settings.showDebugWindows) {
                cv::imshow("Original", frame);
                cv::imshow("Grayscale", grayscale);
                cv::imshow("Edges", edges);
            }
            
            // Afficher les FPS sur l'image
            double fps = metrics.calculateFPS();
            std::string fpsText = "FPS: " + std::to_string(fps);
            cv::putText(frame, fpsText, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            std::string timeText = "Time: " + std::to_string(metrics.getLastMeasurement(MetricType::TOTAL_PROCESSING)) + " ms";
            cv::putText(frame, timeText, cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("Result", frame);
            
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
        if (settings.runTime > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            if (duration >= settings.runTime) {
                running = false;
            }
        }
        
        // Si on traite une seule image (pas en mode caméra), sortir après une frame
        if (!settings.useCamera && !settings.testPerformance) {
            running = false;
        }
    }
    
    // Afficher les résultats de performance
    auto endTime = std::chrono::high_resolution_clock::now();
    double totalRunTime = std::chrono::duration<double>(endTime - startTime).count();
    
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Total frames: " << frameCount << std::endl;
    std::cout << "Total run time: " << totalRunTime << " s" << std::endl;
    std::cout << "Average FPS: " << frameCount / totalRunTime << std::endl;
    
    metrics.printReport();
    
    // Sauvegarder les métriques dans un fichier
    std::string metricsFile = "performance_metrics_" + 
                             std::to_string(settings.useBinarization) + "_" +
                             std::to_string(settings.useMultithreading) + "_" +
                             std::to_string(settings.stressLevel) + ".csv";
    metrics.saveToFile(metricsFile);
    std::cout << "Metrics saved to " << metricsFile << std::endl;
    
    // Libérer les ressources
    acquisition.release();
    cv::destroyAllWindows();
    
    return 0;
}