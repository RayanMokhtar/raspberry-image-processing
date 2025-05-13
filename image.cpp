#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <unistd.h>
#include <string>
#include <sys/sysinfo.h>
#include <atomic>
#include <condition_variable>

using namespace cv;
using namespace std;
using namespace chrono;

// Constantes globales
const int NUM_THREADS = 4;
const double ROI_FACTOR = 0.5;
const int THETA_STEP = 2;
const double PI = 3.14159265;

/* ===========================================================
 *  Section 0 – CpuMonitor helpers (lecture /proc/stat & temp)
 * =========================================================*/
struct CpuSnap { uint64_t user, nice, sys, idle; };

static CpuSnap readCpuLine(const string& key) {
    ifstream stat("/proc/stat");
    string tag; CpuSnap s{0,0,0,0};
    while (stat >> tag) {
        if (tag == key) {
            stat >> s.user >> s.nice >> s.sys >> s.idle;
            break;
        }
        getline(stat, tag); // skip rest of line
    }
    return s;
}

static double deltaPct(const CpuSnap& a, const CpuSnap& b, bool idle=true) {
    uint64_t dUser = b.user - a.user;
    uint64_t dNice = b.nice - a.nice;
    uint64_t dSys  = b.sys  - a.sys;
    uint64_t dIdle = b.idle - a.idle;
    uint64_t total = dUser + dNice + dSys + dIdle;
    if (total == 0) return 0.0;
    return 100.0 * (idle ? dIdle : (total - dIdle)) / total;
}

// % idle global sur 100 ms (modifiable)
static double getCPUIdlePercentage(int sampleMs = 100) {
    CpuSnap a = readCpuLine("cpu");
    this_thread::sleep_for(chrono::milliseconds(sampleMs));
    CpuSnap b = readCpuLine("cpu");
    return deltaPct(a, b, true);
}

// % usage (non-idle) d'un cœur (core>=0) sur 100 ms
static double getCpuUsage(int core, int sampleMs = 100) {
    string key = "cpu" + to_string(core);
    CpuSnap a = readCpuLine(key);
    this_thread::sleep_for(chrono::milliseconds(sampleMs));
    CpuSnap b = readCpuLine(key);
    return deltaPct(a, b, false);
}

// % idle d'un cœur spécifique
static double getCpuIdle(int core, int sampleMs = 100) {
    string key = "cpu" + to_string(core);
    CpuSnap a = readCpuLine(key);
    this_thread::sleep_for(chrono::milliseconds(sampleMs));
    CpuSnap b = readCpuLine(key);
    return deltaPct(a, b, true);
}

// °C ; renvoie -1 si thermal_zone0 absent (ex : WSL2)
static double getTemperature() {
    ifstream fs("/sys/class/thermal/thermal_zone0/temp");
    double milli;
    if(!(fs >> milli)) return -1.0;
    return milli / 1000.0;
}

// Récupère l'usage RAM en MB (utilisé, total)
static pair<double, double> getRamUsageMB() {
    struct sysinfo info;
    if (sysinfo(&info) != 0) {
        return make_pair(0.0, 0.0);
    }
    
    double total = (double)info.totalram * info.mem_unit / (1024 * 1024);
    double free = (double)info.freeram * info.mem_unit / (1024 * 1024);
    return make_pair(total - free, total);
}

// Structure pour les métriques système à un instant T
struct SystemMetrics {
    double cpuIdle;               // % CPU idle global
    double cpuUsagePerCore[NUM_THREADS];  // % CPU utilisé par cœur
    double cpuIdlePerCore[NUM_THREADS];   // % CPU idle par cœur
    double temperature;           // Température en °C
    double ramUsageMB;            // Utilisation RAM (MB)
    double ramTotalMB;            // RAM totale (MB)
    double ramUsagePercent;       // % d'utilisation RAM
    high_resolution_clock::time_point timestamp; // Horodatage de la mesure
    
    SystemMetrics() {
        cpuIdle = 0.0;
        temperature = 0.0;
        ramUsageMB = 0.0;
        ramTotalMB = 0.0;
        ramUsagePercent = 0.0;
        timestamp = high_resolution_clock::now();
        
        for (int i = 0; i < NUM_THREADS; i++) {
            cpuUsagePerCore[i] = 0.0;
            cpuIdlePerCore[i] = 0.0;
        }
    }
};

// Version optimisée pour mesures rapides
static SystemMetrics collectSystemMetricsQuick() {
    SystemMetrics metrics;
    metrics.timestamp = high_resolution_clock::now();
    
    // Lecture CPU sans délai (utiliser les données précédemment échantillonnées)
    metrics.cpuIdle = 0.0;
    for (int i = 0; i < NUM_THREADS; i++) {
        metrics.cpuUsagePerCore[i] = 0.0;
        metrics.cpuIdlePerCore[i] = 0.0;
    }
    
    // Température
    metrics.temperature = getTemperature();
    if (metrics.temperature < 0) {
        metrics.temperature = 45.0;
    }
    
    // RAM
    auto ramInfo = getRamUsageMB();
    metrics.ramUsageMB = ramInfo.first;
    metrics.ramTotalMB = ramInfo.second;
    metrics.ramUsagePercent = (metrics.ramTotalMB > 0) ? 
                            (100.0 * metrics.ramUsageMB / metrics.ramTotalMB) : 0.0;
                              
    return metrics;
}

// Version avec délai pour mesures précises
static SystemMetrics collectSystemMetrics(bool withDelay = false) {
    SystemMetrics metrics;
    metrics.timestamp = high_resolution_clock::now();
    
    // Utiliser un délai minimum pour obtenir des lectures CPU significatives
    const int minDelay = 50; // 50ms minimum pour avoir des mesures fiables
    
    // CPU idle global avec délai minimum
    metrics.cpuIdle = getCPUIdlePercentage(withDelay ? max(50, minDelay) : minDelay);
    
    // CPU par cœur avec délai significatif mais pas excessif
    for (int i = 0; i < NUM_THREADS; i++) {
        // Utiliser le même échantillon pour tous les cœurs pour éviter d'attendre trop longtemps
        metrics.cpuUsagePerCore[i] = getCpuUsage(i, 20);
        metrics.cpuIdlePerCore[i] = 100.0 - metrics.cpuUsagePerCore[i];
    }
    
    // Température - gérer le cas où elle n'est pas disponible
    metrics.temperature = getTemperature();
    if (metrics.temperature < 0) {
        // Si la température n'est pas disponible (WSL, VM, etc.), utiliser une valeur par défaut
        metrics.temperature = 45.0; // Valeur fictive pour les tests
    }
    
    // RAM - code inchangé
    auto ramInfo = getRamUsageMB();
    metrics.ramUsageMB = ramInfo.first;
    metrics.ramTotalMB = ramInfo.second;
    metrics.ramUsagePercent = (metrics.ramTotalMB > 0) ? 
                            (100.0 * metrics.ramUsageMB / metrics.ramTotalMB) : 0.0;
                              
    return metrics;
}

// Structure regroupant toutes les métriques
struct Metrics {
    // Timestamps pour mesures de performances
    high_resolution_clock::time_point startTotal;
    high_resolution_clock::time_point endTotal;
    
    // Métriques temporelles (en ms)
    double readImageTime;
    double grayscaleTime;
    double gaussianTime;
    double hsvMaskTime;
    double sobelTime;
    double houghTime;
    double totalTime;
    
    // Métriques algorithmiques
    int numLinesDetected;
    double averageLineLength;
    int verticalLines;
    int centralLines;
    bool isRoadStraight;
    double processingFPS;
    
    // Métriques système par étape
    SystemMetrics initialMetrics;     // Au démarrage
    SystemMetrics readImageMetrics;   // Après lecture image
    SystemMetrics grayscaleMetrics;   // Après conversion niveaux de gris
    SystemMetrics gaussianMetrics;    // Après flou gaussien
    SystemMetrics maskingMetrics;     // Après masque HSV
    SystemMetrics sobelMetrics;       // Après filtre Sobel
    SystemMetrics houghMetrics;       // Après transformée de Hough
    SystemMetrics finalMetrics;       // À la fin du traitement
    
    // Constructeur avec initialisation
    Metrics() {
        readImageTime = 0;
        grayscaleTime = 0;
        gaussianTime = 0;
        hsvMaskTime = 0;
        sobelTime = 0;
        houghTime = 0;
        totalTime = 0;
        numLinesDetected = 0;
        averageLineLength = 0;
        verticalLines = 0;
        centralLines = 0;
        isRoadStraight = false;
        processingFPS = 0;
    }
};

// Classe pour monitoring continu
class SystemMonitor {
private:
    thread monitoringThread;
    atomic<bool> stopMonitoring{false};
    string outputFile;
    ofstream monitoringFile;
    int samplingIntervalMs;
    high_resolution_clock::time_point startTime;
    string currentOperation;
    mutex opMutex;
    
public:
    SystemMonitor(const string& filename = "continuous_monitoring.csv", int intervalMs = 10) 
        : outputFile(filename), samplingIntervalMs(intervalMs), currentOperation("Initialisation") {
        
        // Initialiser le fichier CSV
        monitoringFile.open(outputFile);
        if (!monitoringFile.is_open()) {
            cerr << "Erreur: Impossible de créer le fichier de monitoring" << endl;
            return;
        }
        
        // Écrire les en-têtes
        monitoringFile << "TimestampMs,Operation,CPU_Idle_Pct";
        for (int i = 0; i < NUM_THREADS; i++) {
            monitoringFile << ",Core" << i << "_Usage_Pct,Core" << i << "_Idle_Pct";
        }
        monitoringFile << ",Temperature_C,RAM_Usage_MB,RAM_Total_MB,RAM_Pct" << endl;
        
        startTime = high_resolution_clock::now();
        
        // Échantillonner initial pour des mesures de référence
        CpuSnap globalInitial = readCpuLine("cpu");
        vector<CpuSnap> coreInitials;
        for (int i = 0; i < NUM_THREADS; i++) {
            coreInitials.push_back(readCpuLine("cpu" + to_string(i)));
        }
        
        // Démarrer le thread de monitoring
        monitoringThread = thread([this, globalInitial, coreInitials]() {
            this->monitoringLoop(globalInitial, coreInitials);
        });
    }
    
    ~SystemMonitor() {
        stop();
    }
    
    void setOperation(const string& operation) {
        lock_guard<mutex> lock(opMutex);
        currentOperation = operation;
    }
    
    void stop() {
        stopMonitoring = true;
        if (monitoringThread.joinable()) {
            monitoringThread.join();
        }
        if (monitoringFile.is_open()) {
            monitoringFile.close();
            cout << "Monitoring terminé, données enregistrées dans " << outputFile << endl;
        }
    }
    
private:
    void monitoringLoop(CpuSnap initialGlobal, vector<CpuSnap> initialCores) {
        CpuSnap prevGlobal = initialGlobal;
        vector<CpuSnap> prevCores = initialCores;
        
        while (!stopMonitoring) {
            auto now = high_resolution_clock::now();
            auto elapsedMs = duration_cast<milliseconds>(now - startTime).count();
            
            // Lire les données CPU
            CpuSnap currentGlobal = readCpuLine("cpu");
            double cpuIdle = deltaPct(prevGlobal, currentGlobal, true);
            prevGlobal = currentGlobal;
            
            vector<double> coreUsage(NUM_THREADS, 0.0);
            vector<double> coreIdle(NUM_THREADS, 0.0);
            
            for (int i = 0; i < NUM_THREADS; i++) {
                CpuSnap currentCore = readCpuLine("cpu" + to_string(i));
                coreUsage[i] = deltaPct(prevCores[i], currentCore, false);
                coreIdle[i] = 100.0 - coreUsage[i];
                prevCores[i] = currentCore;
            }
            
            // Lire température et RAM
            double temperature = getTemperature();
            if (temperature < 0) temperature = 45.0;
            
            auto ramInfo = getRamUsageMB();
            
            // Récupérer l'opération actuelle
            string operation;
            {
                lock_guard<mutex> lock(opMutex);
                operation = currentOperation;
            }
            
            // Écrire l'enregistrement dans le CSV
            monitoringFile << elapsedMs << "," << operation << "," << cpuIdle;
            
            for (int i = 0; i < NUM_THREADS; i++) {
                monitoringFile << "," << coreUsage[i] << "," << coreIdle[i];
            }
            
            monitoringFile << "," << temperature
                           << "," << ramInfo.first
                           << "," << ramInfo.second
                           << "," << (100.0 * ramInfo.first / ramInfo.second) << endl;
            
            // Attendre l'intervalle d'échantillonnage
            this_thread::sleep_for(chrono::milliseconds(samplingIntervalMs));
        }
    }
};

// Afficher les métriques système
void printSystemMetrics(const SystemMetrics& metrics, const string& operation) {
    cout << "\n--- Métriques Système pendant " << operation << " ---" << endl;
    cout << "CPU global: " << fixed << setprecision(2) << (100.0 - metrics.cpuIdle) << "% utilisé ("
         << metrics.cpuIdle << "% idle)" << endl;
    
    cout << "Utilisation par cœur: " << endl;
    for (int i = 0; i < NUM_THREADS; i++) {
        cout << "  Cœur " << i << ": " << metrics.cpuUsagePerCore[i] << "% utilisé ("
             << metrics.cpuIdlePerCore[i] << "% idle)" << endl;
    }
    
    cout << "Température CPU: " << metrics.temperature << " °C" << endl;
    cout << "RAM: " << metrics.ramUsageMB << "/" << metrics.ramTotalMB 
         << " MB (" << metrics.ramUsagePercent << "%)" << endl;
    cout << "-------------------------------------" << endl;
}

// Mutex pour l'accès concurrent
mutex acc_mutex;

// Structure pour les paramètres threads Sobel
struct ThreadParams {
    Mat* input;
    Mat* output;
    int startRow;
    int endRow;
    int threshold;
};

// Thread worker pour Sobel
void sobelThread(ThreadParams params) {
    for(int y = params.startRow; y < params.endRow; y++) {
        for(int x = 1; x < params.input->cols - 1; x++) {
            float gx = -params.input->at<uchar>(y-1, x-1) - 2*params.input->at<uchar>(y, x-1) - params.input->at<uchar>(y+1, x-1) +
                      params.input->at<uchar>(y-1, x+1) + 2*params.input->at<uchar>(y, x+1) + params.input->at<uchar>(y+1, x+1);
            
            float gy = -params.input->at<uchar>(y-1, x-1) - 2*params.input->at<uchar>(y-1, x) - params.input->at<uchar>(y-1, x+1) +
                      params.input->at<uchar>(y+1, x-1) + 2*params.input->at<uchar>(y+1, x) + params.input->at<uchar>(y+1, x+1);
            
            float magnitude = sqrt(gx*gx + gy*gy) / 4.0;
            params.output->at<uchar>(y, x) = magnitude > params.threshold ? magnitude : 0;
        }
    }
}

// Thread worker pour Hough
void houghThread(Mat& edges, Mat& accumulator, int startRow, int endRow) {
    for(int y = startRow; y < endRow; y++) {
        for(int x = 0; x < edges.cols; x++) {
            if(edges.at<uchar>(y, x) > 0) {
                for(int theta = 0; theta < 180; theta += THETA_STEP) {
                    double rho = x * cos(theta * PI / 180.0) + y * sin(theta * PI / 180.0);
                    int rhoIndex = cvRound(rho);
                    
                    if(rhoIndex >= 0 && rhoIndex < accumulator.rows) {
                        lock_guard<mutex> lock(acc_mutex);
                        accumulator.at<ushort>(rhoIndex, theta/THETA_STEP)++;
                    }
                }
            }
        }
    }
}

// Classe principale pour la détection de lignes de route
class RoadDetector {
private:
    Metrics metrics;
    vector<thread> threads;
    shared_ptr<SystemMonitor> monitor;
    
    // Fonction optimisée pour appliquer Sobel (multithreading)
    Mat applySobel(Mat& input, int threshold = 50) {
        if (monitor) monitor->setOperation("Sobel");
        
        Mat output = Mat::zeros(input.size(), CV_8UC1);
        int rowsPerThread = input.rows / NUM_THREADS;
        
        threads.clear();
        for(int i = 0; i < NUM_THREADS; i++) {
            ThreadParams params = {
                &input,
                &output,
                i * rowsPerThread,
                (i == NUM_THREADS-1) ? input.rows : (i+1) * rowsPerThread,
                threshold
            };
            threads.emplace_back(sobelThread, params);
        }
        
        for(auto& t : threads) {
            t.join();
        }
        
        return output;
    }
    
    // Fonction pour appliquer HoughLinesP avec détection adaptative
    int applyHoughLinesP_adaptatif(Mat& edges, Mat& result, int roi_y, int roi_mode, bool& is_straight, bool permissif=false) {
        if (monitor) monitor->setOperation("Hough");
        
        vector<Vec4i> linesP;
        // Paramètres adaptatifs
        int threshold = permissif ? 15 : 30;
        int minLineLength = permissif ? 20 : 40;
        int maxLineGap = permissif ? 40 : 20;
        HoughLinesP(edges, linesP, 1, CV_PI/180, threshold, minLineLength, maxLineGap);
        
        int numLines = 0;
        double totalLength = 0;
        int img_center = result.cols / 2;
        int nb_verticales = 0;
        int nb_centrales = 0;
        
        for(const auto& l : linesP) {
            double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
            double length = sqrt(pow(l[2] - l[0], 2) + pow(l[3] - l[1], 2));
            int x1 = l[0], x2 = l[2];
            int x_center = (x1 + x2) / 2;
            bool is_center = abs(x_center - img_center) < img_center * (roi_mode ? 0.6 : 0.45);
            bool is_vertical = (abs(angle) < 25 || abs(angle) > 155);
            
            // On garde plus large si mode permissif
            if ((is_vertical || permissif) && length > (permissif ? 15 : 50) && is_center) {
                Point pt1(l[0], l[1] + roi_y);
                Point pt2(l[2], l[3] + roi_y);
                line(result, pt1, pt2, Scalar(0, 255, 0), 3, LINE_AA);
                numLines++;
                totalLength += length;
                if(is_vertical) nb_verticales++;
                if(is_center) nb_centrales++;
            }
        }
        
        // Détection route droite : majorité de lignes verticales et centrales
        is_straight = (numLines > 1 && nb_verticales > 0.7 * numLines && nb_centrales > 0.7 * numLines);
        
        // Mise à jour des métriques
        metrics.numLinesDetected = numLines;
        metrics.averageLineLength = numLines > 0 ? totalLength / numLines : 0;
        metrics.verticalLines = nb_verticales;
        metrics.centralLines = nb_centrales;
        metrics.isRoadStraight = is_straight;
        
        return numLines;
    }
    
public:
    RoadDetector(shared_ptr<SystemMonitor> mon = nullptr) : monitor(mon) {
        // Constructeur - rien à faire de spécial
    }
    
    void resetMetrics() {
        metrics = Metrics(); // Reset avec une nouvelle instance
    }
    
    Mat processImage(const string& imagePath) {
        resetMetrics(); // Réinitialiser toutes les métriques
        
        // Démarrer le chronomètre total
        metrics.startTotal = high_resolution_clock::now();
        
        // Mettre à jour l'opération en cours dans le monitor
        if (monitor) monitor->setOperation("Démarrage - " + imagePath);
        
        // --- Collecte des métriques système initiales ---
        cout << "\n=== Démarrage du traitement de l'image: " << imagePath << " ===" << endl;
        metrics.initialMetrics = collectSystemMetrics(true);
        printSystemMetrics(metrics.initialMetrics, "Démarrage");
        
        // --- ÉTAPE 1: CHARGEMENT DE L'IMAGE ---
        if (monitor) monitor->setOperation("Lecture Image");
        auto startReadImage = high_resolution_clock::now();
        Mat input = imread(imagePath);
        if(input.empty()) {
            throw runtime_error("Impossible de charger l'image: " + imagePath);
        }
        auto endReadImage = high_resolution_clock::now();
        metrics.readImageTime = duration_cast<milliseconds>(endReadImage - startReadImage).count();
        metrics.readImageMetrics = collectSystemMetrics(false);
        printSystemMetrics(metrics.readImageMetrics, "Lecture Image");
        
        // --- ÉTAPE 2: CONVERSION EN NIVEAUX DE GRIS ---
        if (monitor) monitor->setOperation("Niveaux de Gris");
        auto startGrayscale = high_resolution_clock::now();
        Mat gray;
        cvtColor(input, gray, COLOR_BGR2GRAY);
        auto endGrayscale = high_resolution_clock::now();
        metrics.grayscaleTime = duration_cast<milliseconds>(endGrayscale - startGrayscale).count();
        metrics.grayscaleMetrics = collectSystemMetrics(false);
        printSystemMetrics(metrics.grayscaleMetrics, "Conversion Niveaux de Gris");
        
        // --- ÉTAPE 3: FLOU GAUSSIEN ---
        if (monitor) monitor->setOperation("Flou Gaussien");
        auto startGaussian = high_resolution_clock::now();
        GaussianBlur(gray, gray, Size(3, 3), 1.5);
        auto endGaussian = high_resolution_clock::now();
        metrics.gaussianTime = duration_cast<milliseconds>(endGaussian - startGaussian).count();
        metrics.gaussianMetrics = collectSystemMetrics(false);
        printSystemMetrics(metrics.gaussianMetrics, "Flou Gaussien");
        
        // --- ÉTAPE 4: MASQUE COULEUR HSV ---
        if (monitor) monitor->setOperation("Masque HSV");
        auto startMasking = high_resolution_clock::now();
        Mat hsv, mask_yellow;
        cvtColor(input, hsv, COLOR_BGR2HSV);
        Scalar lower_yellow(10, 60, 60), upper_yellow(45, 255, 255);
        inRange(hsv, lower_yellow, upper_yellow, mask_yellow);
        auto endMasking = high_resolution_clock::now();
        metrics.hsvMaskTime = duration_cast<milliseconds>(endMasking - startMasking).count();
        metrics.maskingMetrics = collectSystemMetrics(false);
        printSystemMetrics(metrics.maskingMetrics, "Masque HSV");
        
        // --- ÉTAPE 5: DÉFINITION DE LA ROI ---
        if (monitor) monitor->setOperation("Définition ROI");
        int roi_mode = 0;
        int roi_y = gray.rows * ROI_FACTOR;
        int roi_height = gray.rows - roi_y;
        Rect roi_rect(0, roi_y, gray.cols, roi_height);
        Mat roi_gray = gray(roi_rect);
        Mat roi_mask = mask_yellow(roi_rect);
        
        // --- ÉTAPE 6: FILTRE SOBEL MULTITHREADÉ ---
        auto startSobel = high_resolution_clock::now();
        Mat edges = applySobel(roi_gray, 30);
        bitwise_and(edges, roi_mask, edges);
        auto endSobel = high_resolution_clock::now();
        metrics.sobelTime = duration_cast<milliseconds>(endSobel - startSobel).count();
        metrics.sobelMetrics = collectSystemMetrics(false);
        printSystemMetrics(metrics.sobelMetrics, "Filtre Sobel");
        
        // --- ÉTAPE 7: TRANSFORMÉE DE HOUGH ET DÉTECTION DE LIGNES ---
        auto startHough = high_resolution_clock::now();
        Mat result = input.clone();
        bool is_straight = false;
        int numLines = applyHoughLinesP_adaptatif(edges, result, roi_y, roi_mode, is_straight);
        
        // Mode adaptatif si nécessaire (pas assez de lignes détectées)
        if(numLines < 2) {
            if (monitor) monitor->setOperation("Hough Adaptatif");
            roi_mode = 1;
            roi_y = gray.rows / 3;
            roi_height = gray.rows - roi_y;
            roi_rect = Rect(0, roi_y, gray.cols, roi_height);
            roi_gray = gray(roi_rect);
            roi_mask = mask_yellow(roi_rect);
            edges = applySobel(roi_gray, 20);
            bitwise_and(edges, roi_mask, edges);
            result = input.clone();
            is_straight = false;
            numLines = applyHoughLinesP_adaptatif(edges, result, roi_y, roi_mode, is_straight, true);
        }
        auto endHough = high_resolution_clock::now();
        metrics.houghTime = duration_cast<milliseconds>(endHough - startHough).count();
        metrics.houghMetrics = collectSystemMetrics(false);
        printSystemMetrics(metrics.houghMetrics, "Transformée de Hough");
        
        // --- FINALISATION ET CALCUL DES MÉTRIQUES GLOBALES ---
        if (monitor) monitor->setOperation("Finalisation");
        metrics.endTotal = high_resolution_clock::now();
        metrics.totalTime = duration_cast<milliseconds>(metrics.endTotal - metrics.startTotal).count();
        metrics.processingFPS = 1000.0 / metrics.totalTime;
        
        metrics.finalMetrics = collectSystemMetrics(true);
        printSystemMetrics(metrics.finalMetrics, "Finalisation");
        
        return result;
    }
    
    // Fonction d'affichage détaillé des métriques
    void printDetailedMetrics() const {
        cout << "\n====== RAPPORT DÉTAILLÉ DES MÉTRIQUES ======" << endl;
        
        // Section 1: Performance temporelle
        cout << "\n--- PERFORMANCE TEMPORELLE ---" << endl;
        cout << "Temps total de traitement: " << fixed << setprecision(2) << metrics.totalTime << " ms" << endl;
        cout << "FPS estimé: " << metrics.processingFPS << endl;
        cout << "Répartition du temps:" << endl;
        cout << "  - Lecture image: " << metrics.readImageTime << " ms (" 
             << (100.0 * metrics.readImageTime / metrics.totalTime) << "%)" << endl;
        cout << "  - Niveaux de gris: " << metrics.grayscaleTime << " ms (" 
             << (100.0 * metrics.grayscaleTime / metrics.totalTime) << "%)" << endl;
        cout << "  - Flou gaussien: " << metrics.gaussianTime << " ms (" 
             << (100.0 * metrics.gaussianTime / metrics.totalTime) << "%)" << endl;
        cout << "  - Masque HSV: " << metrics.hsvMaskTime << " ms (" 
             << (100.0 * metrics.hsvMaskTime / metrics.totalTime) << "%)" << endl;
        cout << "  - Filtre Sobel: " << metrics.sobelTime << " ms (" 
             << (100.0 * metrics.sobelTime / metrics.totalTime) << "%)" << endl;
        cout << "  - Transformée Hough: " << metrics.houghTime << " ms (" 
             << (100.0 * metrics.houghTime / metrics.totalTime) << "%)" << endl;
             
        // Section 2: Résultats algorithmiques
        cout << "\n--- RÉSULTATS ALGORITHMIQUES ---" << endl;
        cout << "Nombre de lignes détectées: " << metrics.numLinesDetected << endl;
        cout << "Longueur moyenne des lignes: " << metrics.averageLineLength << " pixels" << endl;
        cout << "Lignes verticales: " << metrics.verticalLines << " (" 
             << (metrics.numLinesDetected > 0 ? (100.0 * metrics.verticalLines / metrics.numLinesDetected) : 0)
             << "%)" << endl;
        cout << "Lignes centrales: " << metrics.centralLines << " (" 
             << (metrics.numLinesDetected > 0 ? (100.0 * metrics.centralLines / metrics.numLinesDetected) : 0)
             << "%)" << endl;
        cout << "Détection de route droite: " << (metrics.isRoadStraight ? "OUI" : "NON") << endl;
        
        // Section 3: Tableau comparatif des métriques système
        cout << "\n--- MÉTRIQUES SYSTÈME COMPARATIVES ---" << endl;
        cout << "                    | Initial | ReadImg | Grayscale | Gaussian | HSV Mask | Sobel   | Hough   | Final   " << endl;
        cout << "--------------------+---------+---------+-----------+----------+----------+---------+---------+---------" << endl;
        
        // CPU Idle global
        cout << "CPU Idle (%)        | " << setw(7) << metrics.initialMetrics.cpuIdle << " | " 
                                         << setw(7) << metrics.readImageMetrics.cpuIdle << " | "
                                         << setw(9) << metrics.grayscaleMetrics.cpuIdle << " | "
                                         << setw(8) << metrics.gaussianMetrics.cpuIdle << " | "
                                         << setw(8) << metrics.maskingMetrics.cpuIdle << " | "
                                         << setw(7) << metrics.sobelMetrics.cpuIdle << " | "
                                         << setw(7) << metrics.houghMetrics.cpuIdle << " | "
                                         << setw(7) << metrics.finalMetrics.cpuIdle << endl;
        
        // CPU par cœur
        for (int i = 0; i < NUM_THREADS; i++) {
            cout << "Core" << i << " usage (%)    | " << setw(7) << metrics.initialMetrics.cpuUsagePerCore[i] << " | " 
                                                       << setw(7) << metrics.readImageMetrics.cpuUsagePerCore[i] << " | "
                                                       << setw(9) << metrics.grayscaleMetrics.cpuUsagePerCore[i] << " | "
                                                       << setw(8) << metrics.gaussianMetrics.cpuUsagePerCore[i] << " | "
                                                       << setw(8) << metrics.maskingMetrics.cpuUsagePerCore[i] << " | "
                                                       << setw(7) << metrics.sobelMetrics.cpuUsagePerCore[i] << " | "
                                                       << setw(7) << metrics.houghMetrics.cpuUsagePerCore[i] << " | "
                                                       << setw(7) << metrics.finalMetrics.cpuUsagePerCore[i] << endl;
        }
        
        // Température
        cout << "Température (°C)    | " << setw(7) << metrics.initialMetrics.temperature << " | " 
                                         << setw(7) << metrics.readImageMetrics.temperature << " | "
                                         << setw(9) << metrics.grayscaleMetrics.temperature << " | "
                                         << setw(8) << metrics.gaussianMetrics.temperature << " | "
                                         << setw(8) << metrics.maskingMetrics.temperature << " | "
                                         << setw(7) << metrics.sobelMetrics.temperature << " | "
                                         << setw(7) << metrics.houghMetrics.temperature << " | "
                                         << setw(7) << metrics.finalMetrics.temperature << endl;
        
        // RAM Usage MB
        cout << "RAM usage (MB)      | " << setw(7) << metrics.initialMetrics.ramUsageMB << " | " 
                                         << setw(7) << metrics.readImageMetrics.ramUsageMB << " | "
                                         << setw(9) << metrics.grayscaleMetrics.ramUsageMB << " | "
                                         << setw(8) << metrics.gaussianMetrics.ramUsageMB << " | "
                                         << setw(8) << metrics.maskingMetrics.ramUsageMB << " | "
                                         << setw(7) << metrics.sobelMetrics.ramUsageMB << " | "
                                         << setw(7) << metrics.houghMetrics.ramUsageMB << " | "
                                         << setw(7) << metrics.finalMetrics.ramUsageMB << endl;
        
        // RAM usage %
        cout << "RAM usage (%)       | " << setw(7) << metrics.initialMetrics.ramUsagePercent << " | " 
                                         << setw(7) << metrics.readImageMetrics.ramUsagePercent << " | "
                                         << setw(9) << metrics.grayscaleMetrics.ramUsagePercent << " | "
                                         << setw(8) << metrics.gaussianMetrics.ramUsagePercent << " | "
                                         << setw(8) << metrics.maskingMetrics.ramUsagePercent << " | "
                                         << setw(7) << metrics.sobelMetrics.ramUsagePercent << " | "
                                         << setw(7) << metrics.houghMetrics.ramUsagePercent << " | "
                                         << setw(7) << metrics.finalMetrics.ramUsagePercent << endl;
                                         
        cout << "\n================================================" << endl;
    }
    
    // Accesseur pour la structure Metrics
    const Metrics& getMetrics() const {
        return metrics;
    }
    
    // Export des métriques en CSV
    void exportMetricsCSV(const string& filename) const {
        ofstream csvFile(filename);
        if (!csvFile.is_open()) {
            cerr << "Erreur: Impossible d'ouvrir le fichier " << filename << endl;
            return;
        }
        
        // En-têtes
        csvFile << "Operation,Duration_ms,CPU_Idle_Pct";
        for (int i = 0; i < NUM_THREADS; i++) {
            csvFile << ",Core" << i << "_Usage_Pct";
        }
        csvFile << ",Temperature_C,RAM_Usage_MB,RAM_Pct" << endl;
        
        // Données par opération
        auto writeMetricsRow = [&csvFile](const string& op, double duration, const SystemMetrics& m) {
            csvFile << op << "," << duration << "," << m.cpuIdle;
            for (int i = 0; i < NUM_THREADS; i++) {
                csvFile << "," << m.cpuUsagePerCore[i];
            }
            csvFile << "," << m.temperature << "," << m.ramUsageMB << "," << m.ramUsagePercent << endl;
        };
        
        writeMetricsRow("Initial", 0, metrics.initialMetrics);
        writeMetricsRow("ReadImage", metrics.readImageTime, metrics.readImageMetrics);
        writeMetricsRow("Grayscale", metrics.grayscaleTime, metrics.grayscaleMetrics);
        writeMetricsRow("Gaussian", metrics.gaussianTime, metrics.gaussianMetrics);
        writeMetricsRow("HSVMask", metrics.hsvMaskTime, metrics.maskingMetrics);
        writeMetricsRow("Sobel", metrics.sobelTime, metrics.sobelMetrics);
        writeMetricsRow("Hough", metrics.houghTime, metrics.houghMetrics);
        writeMetricsRow("Final", metrics.totalTime, metrics.finalMetrics);
        
        // Métriques algorithmiques
        csvFile << "\nAlgorithmic Metrics" << endl;
        csvFile << "NumLines,AvgLineLength,VerticalLines,CentralLines,IsRoadStraight,FPS" << endl;
        csvFile << metrics.numLinesDetected << "," << metrics.averageLineLength << ","
                << metrics.verticalLines << "," << metrics.centralLines << ","
                << (metrics.isRoadStraight ? "true" : "false") << "," << metrics.processingFPS << endl;
        
        csvFile.close();
    }
};

// Programme principal
int main() {
    try {
        // Créer le moniteur système pour échantillonnage continu (toutes les 10ms)
        auto monitor = make_shared<SystemMonitor>("continuous_monitoring.csv", 10);
        
        // Créer le détecteur avec le moniteur
        RoadDetector detector(monitor);
        
        // Traitement de la première image
        cout << "\n*** TRAITEMENT IMAGE 1 ***" << endl;
        Mat result1 = detector.processImage("route_low.jpg");
        detector.printDetailedMetrics();
        detector.exportMetricsCSV("metrics_route_low.csv");
        imwrite("result_route_low.jpg", result1);
        
        // Traitement de la deuxième image
        cout << "\n*** TRAITEMENT IMAGE 2 ***" << endl;
        Mat result2 = detector.processImage("route_virage.jpg");
        detector.printDetailedMetrics();
        detector.exportMetricsCSV("metrics_route_virage.csv");
        imwrite("result_route_virage.jpg", result2);
        
        // Arrêter le monitoring avant de quitter
        monitor->stop();
        
        cout << "\nTraitement terminé avec succès!" << endl;
        cout << "Les données de monitoring continu sont disponibles dans 'continuous_monitoring.csv'" << endl;
        
    } catch(const exception& e) {
        cerr << "\nErreur: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}