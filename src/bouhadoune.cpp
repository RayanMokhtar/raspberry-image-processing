#include <stdio.h>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <opencv4/opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#include <sys/time.h>

#define PI 3.14159265
#define thetaStep 6      // Augmenté pour réduire le nombre d'orientations à tester
#define NUM_THREADS 4    // Nombre de threads pour la parallélisation
#define ROI_FACTOR 0.6   // Facteur pour la région d'intérêt (60% inférieure de l'image)
#define USE_IMAGE 1      // Utiliser une image au lieu de la caméra
#define BENCHMARK_RUNS 50 // Nombre d'exécutions pour le benchmark

// Mutex pour l'accès concurrent à l'accumulateur Hough
std::mutex acc_mutex;

// Structure pour passer les paramètres aux threads
struct ThreadParams {
    Mat* frame;
    int* filterGX;
    int* filterGY;
    int size;
    Mat* out;
    int limit;
    int startRow;
    int endRow;
};

// Version threadée et optimisée de l'algorithme de Sobel
void SobelThread(ThreadParams params) {
    int step = std::floor(params.size/2);
    float sumX, sumY;
    const int cols = params.frame->cols;
    const int rows = params.frame->rows;
    
    // Précalcul des indices pour éviter les calculs répétitifs
    for(int x = params.startRow; x < params.endRow; x++) {
        if (x <= 0 || x >= rows - 1) continue;
        
        uchar* outRow = params.out->ptr<uchar>(x);
        
        for(int y = 1; y < cols - 1; y++) {
            // Version optimisée du calcul de Sobel avec accès direct aux pixels
            uchar* pixel = params.frame->ptr<uchar>(x) + y;
            uchar* pixelAbove = params.frame->ptr<uchar>(x-1) + y;
            uchar* pixelBelow = params.frame->ptr<uchar>(x+1) + y;
            
            // Calcul simplifié avec accès direct aux pixels
            sumX = *(pixelAbove + 1) - *(pixelAbove - 1) +
                  2 * (*(pixel + 1) - *(pixel - 1)) +
                  *(pixelBelow + 1) - *(pixelBelow - 1);
                  
            sumY = *(pixelBelow - 1) - *(pixelAbove - 1) +
                  2 * (*pixelBelow - *pixelAbove) +
                  *(pixelBelow + 1) - *(pixelAbove + 1);
            
            float gradient = sqrt(sumX*sumX + sumY*sumY) / 4;
            outRow[y] = gradient < params.limit ? 0 : gradient;
        }
    }
}

// Version multi-threadée améliorée du filtre Sobel
void SobelMultiThread(Mat& frame, int* filterGX, int* filterGY, int size, Mat& out, int limit) {
    std::vector<std::thread> threads;
    std::vector<ThreadParams> params(NUM_THREADS);
    
    int rowsPerThread = frame.rows / NUM_THREADS;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        params[i] = {
            &frame,
            filterGX,
            filterGY,
            size,
            &out,
            limit,
            i * rowsPerThread,
            (i == NUM_THREADS - 1) ? frame.rows : (i + 1) * rowsPerThread
        };
        
        threads.push_back(std::thread(SobelThread, params[i]));
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

// Version optimisée de la transformée de Hough pour chaque thread
void HoughThread(const Mat& frame, Mat& acc, int startRow, int endRow) {
    const int cols = frame.cols;
    const int step = frame.step;
    const double angleStep = thetaStep * PI / 180.0;
    
    // Précalcul des sinus et cosinus pour toutes les orientations
    const int angleCount = 180 / thetaStep;
    vector<double> cosTable(angleCount);
    vector<double> sinTable(angleCount);
    
    for (int t = 0; t < angleCount; t++) {
        double theta = t * angleStep;
        cosTable[t] = cos(theta);
        sinTable[t] = sin(theta);
    }
    
    for (int i = startRow; i < endRow; i++) {
        const uchar* row = frame.ptr<uchar>(i);
        
        for (int j = 0; j < cols; j++) {
            if (row[j] > 0) {  // Pixel est un bord
                for (int t = 0; t < angleCount; t++) {
                    double rho = j * cosTable[t] + i * sinTable[t];
                    if (rho > 0) {
                        int rhoIndex = cvRound(rho);
                        
                        // Protection de l'accès concurrent à l'accumulateur
                        acc_mutex.lock();
                        acc.at<ushort>(t, rhoIndex) += 1;
                        acc_mutex.unlock();
                    }
                }
            }
        }
    }
}

// Transformée de Hough multi-threadée optimisée avec région d'intérêt
void HoughMultiThread(const Mat& frame, Mat& acc, Mat& f) {
    std::vector<std::thread> threads;
    
    // Calcul de la région d'intérêt (ROI) - partie inférieure de l'image
    int roiStartRow = frame.rows * (1 - ROI_FACTOR);
    int rowsPerThread = (frame.rows - roiStartRow) / NUM_THREADS;
    
    // Lancement des threads pour la transformée de Hough
    for (int i = 0; i < NUM_THREADS; i++) {
        int startRow = roiStartRow + i * rowsPerThread;
        int endRow = (i == NUM_THREADS - 1) ? frame.rows : roiStartRow + (i + 1) * rowsPerThread;
        
        threads.push_back(std::thread(HoughThread, std::ref(frame), std::ref(acc), startRow, endRow));
    }
    
    // Attente de la fin de tous les threads
    for (auto& t : threads) {
        t.join();
    }
    
    // Extraction des lignes à partir de l'accumulateur
    vector<Vec4i> lines;
    
    // Trouver les maxima locaux dans l'accumulateur
    for (int i = 0; i < 8; i++) {  // Limité à 8 lignes pour plus de rapidité
        cv::Point min_loc, max_loc;
        double min, max;
        cv::minMaxLoc(acc, &min, &max, &min_loc, &max_loc);
        
        if (max < 20) break;  // Seuil minimum pour considérer un pic comme une ligne
        
        // Convertir les coordonnées polaires en cartésiennes
        int theta = max_loc.y * thetaStep;
        double a = cos(theta * PI / 180);
        double b = sin(theta * PI / 180);
        double x0 = a * max_loc.x;
        double y0 = b * max_loc.x;
        
        // Calculer les points de la ligne
        Point pt1, pt2;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        
        // Enregistrer la ligne
        lines.push_back(Vec4i(pt1.x, pt1.y, pt2.x, pt2.y));
        
        // Effacer le maximum pour en trouver un nouveau
        acc.at<ushort>(max_loc) = 0;
        
        // Supprimer les pics proches pour éviter les lignes redondantes
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                Point neighbour = max_loc + Point(dx, dy);
                if (neighbour.x >= 0 && neighbour.x < acc.cols && 
                    neighbour.y >= 0 && neighbour.y < acc.rows) {
                    acc.at<ushort>(neighbour) = 0;
                }
            }
        }
    }
    
    // Dessiner les lignes
    for (const auto& line : lines) {
        cv::line(f, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 0, 255), 2);
    }
}

// Fonction optimisée de calcul d'angle pour la conduite autonome
double calculateSteeringAngle(const Mat& acc, const Mat& frame) {
    cv::Point min_loc, max_loc;
    double min, max;
    cv::minMaxLoc(acc, &min, &max, &min_loc, &max_loc);
    
    if (max < 10) return 0.0;  // Pas assez de votes pour une ligne fiable
    
    // Calcul de l'angle à partir de la ligne la plus forte
    double theta = (double)max_loc.y * thetaStep;
    
    // Normalisation de l'angle entre -90 et 90 degrés
    double steeringAngle = theta - 90;
    if (steeringAngle < -90) steeringAngle += 180;
    if (steeringAngle > 90) steeringAngle -= 180;
    
    return steeringAngle;
}

// Calcul optimisé de la différence de temps en millisecondes
int diff_ms(timeval t1, timeval t2) {
    return (((t1.tv_sec - t2.tv_sec) * 1000000) + (t1.tv_usec - t2.tv_usec)) / 1000;
}

// Conversion RGB vers niveaux de gris optimisée
void RGBtoGrayScale(const Mat& rgb, Mat& grayscale) {
    // Utilisation directe de la fonction optimisée d'OpenCV
    cvtColor(rgb, grayscale, COLOR_BGR2GRAY);
}

// Version optimisée du filtre de détection de bords pour une meilleure performance
void FastEdgeDetection(const Mat& src, Mat& dst, int threshold) {
    // Créer une copie pour accès rapide
    Mat gray = src.clone();
    dst = Mat::zeros(src.size(), CV_8UC1);
    
    const int rows = src.rows;
    const int cols = src.cols;
    
    #pragma omp parallel for
    for (int i = 1; i < rows - 1; i++) {
        const uchar* prev = gray.ptr<uchar>(i - 1);
        const uchar* curr = gray.ptr<uchar>(i);
        const uchar* next = gray.ptr<uchar>(i + 1);
        uchar* out = dst.ptr<uchar>(i);
        
        for (int j = 1; j < cols - 1; j++) {
            // Opérateur Sobel 3x3 simplifié avec accès direct aux pixels
            int gx = (prev[j+1] - prev[j-1]) + 
                     2 * (curr[j+1] - curr[j-1]) + 
                     (next[j+1] - next[j-1]);
                     
            int gy = (next[j-1] - prev[j-1]) + 
                     2 * (next[j] - prev[j]) + 
                     (next[j+1] - prev[j+1]);
            
            int sum = abs(gx) + abs(gy);  // Manhattan distance pour plus de rapidité
            out[j] = sum > threshold ? 255 : 0;
        }
    }
}

int main(int argc, char** argv) {
    printf("Démarrage du système optimisé de détection de lignes\n");
    
    // Vecteurs pour stocker les résultats de benchmark
    vector<double> processing_times;
    vector<double> fps_values;
    
    // Charger l'image d'entrée
    Mat frame = imread("route_low.jpg", IMREAD_COLOR);
    if (frame.empty()) {
        printf("Erreur: Impossible de charger l'image 'route_low.jpg'\n");
        return -1;
    }
    
    // Réduire la taille de l'image pour améliorer les performances
    resize(frame, frame, Size(), 0.5, 0.5, INTER_AREA);
    
    printf("Configuration du système\n");
    printf("Dimensions de l'image: %d lignes, %d colonnes\n", frame.rows, frame.cols);
    
    // Préparation des matrices
    int accWidth = cvRound(sqrt(pow(frame.cols, 2) + pow(frame.rows, 2)));
    Mat acc = Mat::zeros(180/thetaStep, accWidth, CV_16UC1);
    Mat grayscale, sobel;
    
    // Filtres Sobel précalculés
    int filterGX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int filterGY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int convSize = 3;
    
    // Création des fenêtres d'affichage
    namedWindow("Image avec lignes détectées", WINDOW_NORMAL);
    namedWindow("Contours détectés", WINDOW_NORMAL);
    namedWindow("Accumulateur Hough", WINDOW_NORMAL);
    
    // Faire le benchmark sur plusieurs exécutions
    for (int run = 0; run < BENCHMARK_RUNS; run++) {
        Mat frameCopy = frame.clone();
        acc = Mat::zeros(180/thetaStep, accWidth, CV_16UC1);
        
        // Mesurer le temps d'exécution
        auto start = chrono::high_resolution_clock::now();
        
        // Étape 1: Conversion en niveaux de gris
        RGBtoGrayScale(frameCopy, grayscale);
        
        // Étape 2: Flou gaussien avec noyau optimisé
        GaussianBlur(grayscale, grayscale, Size(5, 5), 1.5);
        
        // Étape 3: Détection de contours optimisée
        sobel = Mat::zeros(grayscale.size(), CV_8UC1);
        FastEdgeDetection(grayscale, sobel, 30);
        
        // Étape 4: Transformée de Hough multi-threadée
        HoughMultiThread(sobel, acc, frameCopy);
        
        // Étape 5: Calcul de l'angle de direction
        double steeringAngle = calculateSteeringAngle(acc, frameCopy);
        
        auto end = chrono::high_resolution_clock::now();
        double processing_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        double fps = 1000.0 / processing_time;
        
        processing_times.push_back(processing_time);
        fps_values.push_back(fps);
        
        // Afficher les statistiques
        char text[100];
        sprintf(text, "FPS: %.2f | Angle: %.2f deg | Run: %d/%d", 
                fps, steeringAngle, run+1, BENCHMARK_RUNS);
        putText(frameCopy, text, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        
        // Afficher une flèche indiquant la direction
        Point center(frameCopy.cols / 2, frameCopy.rows - 30);
        Point direction(center.x + sin(steeringAngle * PI / 180) * 50, 
                       center.y - cos(steeringAngle * PI / 180) * 50);
        arrowedLine(frameCopy, center, direction, Scalar(0, 255, 0), 2);
        
        // Afficher les résultats
        imshow("Image avec lignes détectées", frameCopy);
        imshow("Contours détectés", sobel);
        
        // Normaliser l'accumulateur pour l'affichage
        Mat acc_display;
        normalize(acc, acc_display, 0, 255, NORM_MINMAX, CV_8UC1);
        imshow("Accumulateur Hough", acc_display);
        
        waitKey(100); // Petit délai pour voir l'évolution
    }
    
    // Calculer les statistiques
    double avg_time = 0, min_time = 1e9, max_time = 0;
    double avg_fps = 0, min_fps = 1e9, max_fps = 0;
    double variance = 0;
    
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        avg_time += processing_times[i];
        avg_fps += fps_values[i];
        
        if (processing_times[i] < min_time) min_time = processing_times[i];
        if (processing_times[i] > max_time) max_time = processing_times[i];
        
        if (fps_values[i] < min_fps) min_fps = fps_values[i];
        if (fps_values[i] > max_fps) max_fps = fps_values[i];
    }
    
    avg_time /= BENCHMARK_RUNS;
    avg_fps /= BENCHMARK_RUNS;
    
    // Calculer l'écart-type pour évaluer la stabilité
    for (double fps : fps_values) {
        variance += pow(fps - avg_fps, 2);
    }
    variance /= fps_values.size();
    double stdDev = sqrt(variance);
    double cv = (stdDev / avg_fps) * 100; // Coefficient de variation
    
    // Afficher le rapport final de certification
    printf("\n=== CERTIFICATION DE PERFORMANCE ===\n");
    printf("Nombre d'exécutions: %d\n", BENCHMARK_RUNS);
    printf("Temps moyen de traitement: %.2f ms\n", avg_time);
    printf("Temps minimum: %.2f ms\n", min_time);
    printf("Temps maximum: %.2f ms\n", max_time);
    printf("FPS moyen: %.2f\n", avg_fps);
    printf("FPS minimum: %.2f\n", min_fps);
    printf("FPS maximum: %.2f\n", max_fps);
    printf("Écart-type des FPS: %.2f\n", stdDev);
    printf("Coefficient de variation: %.2f%%\n", cv);
    printf("Certification temps réel: %s\n", (cv < 10.0) ? "STABLE" : "INSTABLE");
    
    // Attendre une touche avant de terminer
    printf("\nAppuyez sur une touche pour quitter...\n");
    waitKey(0);
    
    return 0;
}