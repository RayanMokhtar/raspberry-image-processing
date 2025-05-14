#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <numeric>
#include <iomanip>
#include <filesystem>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace chrono;

// Constantes
const int NUM_THREADS = 4; // demain tester avec 3 4 5 threads
const double ROI_FACTOR = 0.5; //  commence par 50% plus hauts
const int THETA_STEP = 4;
const double PI = 3.14159265;
const std::string input_dir = "./data/"; // Changed from ..data/
const std::string output_dir = "./output-data/"; 
const int nombreMinimalLignes = 2 ;
mutex acc_mutex;


// Structure pour les métriques
struct Metrics {
    double sobelTime;
    double houghTime;
    double totalTime;
    double gaussianTime ; 
    double readImageTime;
    int numLinesDetected;
    double averageLineLength;
    double processingFPS;
    double modeSoupleTime;
    double color ;
};


// Structure pour les paramètres des threads
struct ThreadParams {
    Mat* input;
    Mat* output;
    int startRow;
    int endRow;
    int threshold;
};

struct HoughThreadParams {
    Mat* edges;
    int startRow;
    int endRow;
    vector<Vec4i>* lines;
    int threshold;
    int minLineLength;
    int maxLineGap;
    mutex* linesMutex;
};

struct ColorThreadParams {
    const Mat* input;
    Mat* output;
    int startRow;
    int endRow;
    bool isDarkImage;
    bool detectYellow;
    bool detectWhite;
};


//fonctions utilitaires pour la lecture des images dans notre dataset 
namespace fs = std::filesystem;

bool isImageFile(const fs::path& path) {
    if (!fs::is_regular_file(path)) return false;
    
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    // List of supported image extensions
    const std::vector<std::string> validExtensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    };
    
    return std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end();
}

// Function to get all image files from a directory
std::vector<fs::path> getImagesFromDirectory(const std::string& dirPath) {
    std::vector<fs::path> imageFiles;
    
    try {
        for (const auto& entry : fs::directory_iterator(dirPath)) {
            if (isImageFile(entry.path())) {
                imageFiles.push_back(entry.path());
            }
        }
        
        if (imageFiles.empty()) {
            std::cout << "No image files found in directory: " << dirPath << std::endl;
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
    }
    
    return imageFiles;
}

bool ensureDirectoryExists(const std::string& dirPath) {
    try {
        if (!fs::exists(dirPath)) {
            if (!fs::create_directories(dirPath)) {
                std::cerr << "Failed to create directory: " << dirPath << std::endl;
                return false;
            }
            std::cout << "Created output directory: " << dirPath << std::endl;
        }
        return true;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
        return false;
    }
}



void colorMaskThreadFunction(ColorThreadParams params) {
    // Create ROI for this thread
    Rect roi(0, params.startRow, params.input->cols, params.endRow - params.startRow);
    Mat inputROI = (*params.input)(roi);
    Mat outputROI = Mat::zeros(inputROI.size(), CV_8UC1);
    
    // Convert to HSV
    Mat hsv;
    cvtColor(inputROI, hsv, COLOR_BGR2HSV);
    
    // Yellow mask with adaptive thresholds
    if (params.detectYellow) {
        Scalar lower_yellow, upper_yellow;
        if(params.isDarkImage) {
            lower_yellow = Scalar(10, 40, 40);
            upper_yellow = Scalar(45, 255, 255);
        } else {
            lower_yellow = Scalar(10, 80, 80);
            upper_yellow = Scalar(40, 255, 255);
        }
        
        Mat mask_yellow;
        inRange(hsv, lower_yellow, upper_yellow, mask_yellow);
        
        // Apply morphological operations
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(mask_yellow, mask_yellow, MORPH_OPEN, kernel);
        
        // Add to final mask
        bitwise_or(outputROI, mask_yellow, outputROI);
    }
    
    // White mask with adaptive thresholds
    if (params.detectWhite) {
        Scalar lower_white, upper_white;
        if(params.isDarkImage) {
            lower_white = Scalar(0, 0, 140);
            upper_white = Scalar(180, 45, 255);
        } else {
            lower_white = Scalar(0, 0, 200);
            upper_white = Scalar(180, 30, 255);
        }
        
        Mat mask_white;
        inRange(hsv, lower_white, upper_white, mask_white);
        
        // Apply morphological operations
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(mask_white, mask_white, MORPH_OPEN, kernel);
        
        // Add to final mask
        bitwise_or(outputROI, mask_white, outputROI);
    }
    
    // Final morphological closing
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(outputROI, outputROI, MORPH_CLOSE, kernel);
    
    // Copy result to output
    outputROI.copyTo((*params.output)(roi));
}


//TODO à déplacer par la suite 
void houghThreadFunction(HoughThreadParams params) {
    // Créer une ROI pour la section d'image traitée par ce thread
    Rect roi(0, params.startRow, params.edges->cols, params.endRow - params.startRow);
    Mat edgesROI = (*params.edges)(roi);
    
    // Exécuter HoughLinesP sur cette région
    vector<Vec4i> localLines;
    HoughLinesP(edgesROI, localLines, 1, CV_PI*THETA_STEP/180, params.threshold, //TODO revenir sur cette logique de theta steps ? 
                params.minLineLength, params.maxLineGap);
    
    // Ajuster les coordonnées des lignes pour correspondre à l'image globale
    for(auto& line : localLines) {
        line[1] += params.startRow;
        line[3] += params.startRow;
    }
    
    // Ajouter les lignes trouvées au vecteur global avec protection mutex
    {
        lock_guard<mutex> lock(*params.linesMutex);
        for(const auto& line : localLines) {
            params.lines->push_back(line);
        }
    }
}


// Classe principale pour le traitement d'image
class DetecteurLignes {
private:
    Metrics metrics;
    vector<thread> threads;
    
public:
    DetecteurLignes() {
        resetMetrics();
    }
    
    void resetMetrics() {
        metrics.readImageTime = 0 ; 
        metrics.sobelTime = 0;
        metrics.houghTime = 0;
        metrics.totalTime = 0;
        metrics.gaussianTime = 0;
        metrics.numLinesDetected = 0;
        metrics.averageLineLength = 0;
        metrics.processingFPS = 0;
        metrics.modeSoupleTime = 0;
        metrics.color = 0;
    }
    


    // Ajouter cette fonction à votre classe DetecteurLignes
    cv::Mat augmenterConstraste(const Mat& input) {
        Mat enhanced;
        
        // 1. Déterminer si l'image est sombre
        Scalar meanIntensity = mean(input);
        bool isDarkImage = (meanIntensity[0] + meanIntensity[1] + meanIntensity[2])/3 < 100;
        
        if(isDarkImage) {
            // 2. Convertir en espace LAB pour travailler sur la luminosité
            Mat labImg;
            cvtColor(input, labImg, COLOR_BGR2Lab);
            
            // 3. Séparer les canaux
            vector<Mat> labChannels(3);
            split(labImg, labChannels);
            
            Ptr<CLAHE> clahe = createCLAHE();
            clahe->setClipLimit(3.0);  // Limite de contraste
            clahe->setTilesGridSize(Size(8, 8)); // Taille de la grille
            clahe->apply(labChannels[0], labChannels[0]);
            
            // 5. Fusionner les canaux
            merge(labChannels, labImg);
            
            // 6. Convertir de nouveau en BGR
            cvtColor(labImg, enhanced, COLOR_Lab2BGR);
        } else {
            enhanced = input.clone();
        }
        
        return enhanced;
    }

    Mat cannyMultiThread(cv::Mat& input, int seuil = 50) {
        // Création de la matrice de sortie
        Mat output = Mat::zeros(input.size(), CV_8UC1);
        
        //nbr ligne par thread
        int rowsPerThread = input.rows / NUM_THREADS;

        //suffisamment de lignes pour chaque thread
        rowsPerThread = max(rowsPerThread, 10);
        
        // Définir les seuils pour Canny
        double threshold1 = seuil;
        double threshold2 = seuil * 2.5;
        int apertureSize = 3; // Taille standard pour Canny
        
        vector<thread> threads;
        for(int i = 0; i < NUM_THREADS; i++) {
            int startRow = i * rowsPerThread;
            int endRow = (i == NUM_THREADS-1) ? input.rows : (i+1) * rowsPerThread;
            
            // Garantir un chevauchement pour éviter les artefacts aux frontières
            int padding = 3; // Pour correspondre à apertureSize
            startRow = max(0, startRow - (i > 0 ? padding : 0));
            endRow = min(input.rows, endRow + (i < NUM_THREADS-1 ? padding : 0));
            
            //thread avec lambda , equivalent apply en python ou en js on aura par exemple les arrow functions ?? vérifier si c optimisé ? 

            threads.emplace_back([&input, &output, startRow, endRow, threshold1, threshold2, apertureSize]() {
                // Créer des ROI pour l'entrée et la sortie
                Rect roi(0, startRow, input.cols, endRow - startRow);
                Mat inputROI = input(roi);
                Mat tempOutput;
                
                // Appliquer Canny sur cette région
                Canny(inputROI, tempOutput, threshold1, threshold2, apertureSize);
                
                // Copier le résultat dans la matrice de sortie
                tempOutput.copyTo(output(roi));
            });
        }
        
        // Attendre la fin de tous les threads
        for(auto& t : threads) {
            t.join();
        }
        
        return output;
    }
    
    Mat sobelMultiThread(Mat& input, int threshold = 50) {
        // Création de la matrice de sortie
        Mat output = Mat::zeros(input.size(), CV_8UC1);
        int rowsPerThread = input.rows / NUM_THREADS;
        
        rowsPerThread = max(rowsPerThread, 5);
        
        vector<thread> threads;
        for(int i = 0; i < NUM_THREADS; i++) {
            int startRow = i * rowsPerThread;
            int endRow = (i == NUM_THREADS-1) ? input.rows : (i+1) * rowsPerThread;
            
            // Garantir au moins 1 ligne de chevauchement aux frontières
            startRow = max(0, startRow - (i > 0 ? 1 : 0)); 
            endRow = min(input.rows, endRow + (i < NUM_THREADS-1 ? 1 : 0));
            
            threads.emplace_back([&input, &output, startRow, endRow, threshold]() {
                // Création de régions d'intérêt pour ce thread
                Mat inputROI = input(Range(startRow, endRow), Range(0, input.cols));
                Mat outputROI = output(Range(startRow, endRow), Range(0, input.cols));
                
                // Création de matrices temporaires
                Mat grad_x, grad_y;
                Mat abs_grad_x, abs_grad_y;
                Mat gradient;
                
                // Calcul des gradients dans les directions x et y
                Sobel(inputROI, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
                Sobel(inputROI, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
                
                // Conversion en valeurs absolues
                convertScaleAbs(grad_x, abs_grad_x);
                convertScaleAbs(grad_y, abs_grad_y);
                
                // Combinaison des gradients
                addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradient);
                
                // Application du seuil
                cv::threshold(gradient, outputROI, threshold, 255, THRESH_BINARY);
            });
        }
        
        // Attendre la fin de tous les threads
        for(auto& t : threads) {
            t.join();
        }
        
        return output;
    }
    
    

    int houghMultiThreadAdaptative(Mat& edges, Mat& result, int roi_y, int roi_mode, bool& is_straight, bool is_souple=false) {
        
        //à ajuster TODO , mais j'ai l'impression que dépend bcp de qualité ilmage , 
        int threshold = is_souple ? 15 : 30;
        int minLineLength = is_souple ? 20 : 30;
        int maxLineGap = is_souple ? 20 : 15; // à voir comment on traite les lignes discontinues comme ça , vu qu'on tolére distnace amax entre 2 pts de 20 pixels
        

        //logique de parallèlisation pour détection des ligens , à noter qu'on utilsie un mutex pour éviter les conflits d'accès
        vector<Vec4i> allLines;
        mutex linesMutex;
        
        int rowsPerThread = edges.rows / NUM_THREADS;
        int overlap = minLineLength; // Chevauchement équivalent à la longueur minimale de ligne
        
        // Lancer les threads
        vector<thread> threads;
        for(int i = 0; i < NUM_THREADS; i++) {
            int startRow = max(0, i * rowsPerThread - overlap);
            int endRow = min(edges.rows, (i + 1) * rowsPerThread + overlap);
            
            // Éviter des régions trop petites pour le dernier thread
            if(i == NUM_THREADS - 1) {
                endRow = edges.rows;
            }
            
            // Créer les paramètres du thread
            HoughThreadParams params;
            params.edges = &edges;
            params.startRow = startRow;
            params.endRow = endRow;
            params.lines = &allLines;
            params.threshold = threshold;
            params.minLineLength = minLineLength;
            params.maxLineGap = maxLineGap;
            params.linesMutex = &linesMutex;
            
            // Lancer le thread
            threads.emplace_back(houghThreadFunction, params); //emplacement de la fonction threaded
        }
        
        // Attendre la fin de tous les threads
        for(auto& t : threads) {
            t.join();
        }
        
        // Post-traitement : filtrer les lignes en double
        vector<Vec4i> filteredLines;
        if(!allLines.empty()) {
            // Première étape : tri des lignes par position et angle
            sort(allLines.begin(), allLines.end(), [](const Vec4i& a, const Vec4i& b) {
                // Calculer les points milieux
                Point2f midA((a[0] + a[2]) / 2.0f, (a[1] + a[3]) / 2.0f);
                Point2f midB((b[0] + b[2]) / 2.0f, (b[1] + b[3]) / 2.0f);
                
                // Comparer d'abord par position y
                if(abs(midA.y - midB.y) > 10)
                    return midA.y < midB.y;
                
                // Puis par position x
                if(abs(midA.x - midB.x) > 10)
                    return midA.x < midB.x;
                
                // Enfin par angle
                float angleA = atan2(a[3] - a[1], a[2] - a[0]);
                float angleB = atan2(b[3] - b[1], b[2] - b[0]);
                return angleA < angleB;
            });
            
            // Deuxième étape : éliminer les doublons
            filteredLines.push_back(allLines[0]);
            for(size_t i = 1; i < allLines.size(); i++) {
                const Vec4i& prev = filteredLines.back();
                const Vec4i& curr = allLines[i];
                
                // Calculer les points milieux
                Point2f midPrev((prev[0] + prev[2]) / 2.0f, (prev[1] + prev[3]) / 2.0f);
                Point2f midCurr((curr[0] + curr[2]) / 2.0f, (curr[1] + curr[3]) / 2.0f);
                
                // Calculer les angles
                float anglePrev = atan2(prev[3] - prev[1], prev[2] - prev[0]) * 180.0 / CV_PI;
                float angleCurr = atan2(curr[3] - curr[1], curr[2] - curr[0]) * 180.0 / CV_PI;
                
                // Distance entre points milieux
                float distance = norm(midPrev - midCurr);
                
                // Différence d'angle
                float angleDiff = abs(anglePrev - angleCurr);
                while(angleDiff > 180) angleDiff = 360 - angleDiff;
                
                // Si les lignes sont proches et ont un angle similaire, les considérer comme doublons
                if(distance > 20 || angleDiff > 15) {
                    filteredLines.push_back(curr);
                }
            }
        }
        
        // Application des critères de sélection et dessin des lignes
        int numLines = 0;
        double totalLength = 0;
        int img_center = result.cols / 2;
        int nb_verticales = 0;
        int nb_centrales = 0;
        
        for(const auto& l : filteredLines) {
            double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
            double length = sqrt(pow(l[2] - l[0], 2) + pow(l[3] - l[1], 2));
            int x1 = l[0], x2 = l[2];
            int x_center = (x1 + x2) / 2;
            bool is_center = abs(x_center - img_center) < img_center * (roi_mode ? 0.6 : 0.45);
            bool is_vertical = (abs(angle) < 25 || abs(angle) > 155);
            
            // On garde plus large si mode is_souple
            if ((is_vertical || is_souple) && length > (is_souple ? 15 : 50) && is_center) {
                Point pt1(l[0], l[1] + roi_y);
                Point pt2(l[2], l[3] + roi_y);
                line(result, pt1, pt2, Scalar(0, 255, 0), 3, LINE_AA);
                numLines++;
                totalLength += length;
                if(is_vertical) nb_verticales++;
                if(is_center) nb_centrales++;
            }
        }
        
        // Détection automatique de route droite : majorité de lignes verticales et centrales
        is_straight = (numLines > 1 && nb_verticales > 0.7 * numLines && nb_centrales > 0.7 * numLines);
        
        // Mise à jour des métriques
        metrics.numLinesDetected = numLines;
        metrics.averageLineLength = numLines > 0 ? totalLength / numLines : 0;
        
        return numLines;
    }

    // Fonction de réduction de bruit avancée
    Mat reduireBruitAvance(const Mat& inputGray) {
        Mat result;
        // 1. Filtre bilatéral - préserve les bords mieux que le flou gaussien
        Mat bilateral;
        bilateralFilter(inputGray, bilateral, 9, 75, 75);
        
        // 2. Filtre médian - efficace contre le bruit "sel et poivre"
        Mat median;
        medianBlur(bilateral, median, 5);
        
        // 3. Morphologie pour nettoyer davantage
        Mat morpho;
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(median, morpho, MORPH_OPEN, kernel);
        
        // 4. Normalisation du contraste
        normalize(morpho, result, 0, 255, NORM_MINMAX);
        
        return result;
    }

    // Fonction pour extraire la ROI (Region of Interest)
    std::tuple<Mat, Mat, int, int> extractROI(const Mat& grayImage, const Mat& colorMask, bool extendedROI = false) {
        // Calcul de la position de la ROI en fonction du mode
        int roi_mode = extendedROI ? 1 : 0;
        
        //ici roi devient 2/3 bas de l'image . 
        int roi_y = extendedROI ? grayImage.rows / 3 : grayImage.rows * ROI_FACTOR;
        int roi_height = grayImage.rows - roi_y;
        
        // Création du rectangle ROI
        Rect roi_rect(0, roi_y, grayImage.cols, roi_height);
        
       
        Mat roi_gray = grayImage(roi_rect).clone();
        Mat roi_mask = colorMask(roi_rect).clone();
        
        return std::make_tuple(roi_gray, roi_mask, roi_y, roi_mode);
    }
    // Fonction pour créer des masques de couleur pour détecter les lignes jaunes et blanches
    Mat createRoadLineMasksMultiThread(const Mat& input, bool detectYellow = true, bool detectWhite = true) {
        // Create output mask
        Mat final_mask = Mat::zeros(input.size(), CV_8UC1);
        
        // Determine if the image is dark
        Scalar meanIntensity = mean(input);
        bool isDarkImage = (meanIntensity[0] + meanIntensity[1] + meanIntensity[2])/3 < 100;
        
        // Calculate rows per thread
        int rowsPerThread = input.rows / NUM_THREADS;
        rowsPerThread = max(rowsPerThread, 10); // Minimum rows per thread
        
        // Overlap for morphological operations
        int overlap = 5; // Based on kernel size
        
        vector<thread> threads;
        for(int i = 0; i < NUM_THREADS; i++) {
            int startRow = i * rowsPerThread;
            int endRow = (i == NUM_THREADS-1) ? input.rows : (i+1) * rowsPerThread;
            
            // Add overlap between regions
            startRow = max(0, startRow - (i > 0 ? overlap : 0));
            endRow = min(input.rows, endRow + (i < NUM_THREADS-1 ? overlap : 0));
            
            // Create thread parameters
            ColorThreadParams params;
            params.input = &input;
            params.output = &final_mask;
            params.startRow = startRow;
            params.endRow = endRow;
            params.isDarkImage = isDarkImage;
            params.detectYellow = detectYellow;
            params.detectWhite = detectWhite;
            
            // Launch thread
            threads.emplace_back(colorMaskThreadFunction, params);
        }
        
        // Wait for all threads to finish
        for(auto& t : threads) {
            t.join();
        }
        
        // Final cleanup to ensure consistent results across thread boundaries
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(final_mask, final_mask, MORPH_CLOSE, kernel);
        
        return final_mask;
    }

    Mat gaussianBlurMultiThread(const Mat& input, Size kernelSize = Size(5, 5), double sigmaX = 1.5, double sigmaY = 0) {
        // Vérification de l'entrée
        if (input.empty()) {
            std::cerr << "gaussianBlurMultiThread: Image d'entrée vide" << std::endl;
            return Mat();
        }
        
        // Création de la matrice de sortie
        Mat output = Mat::zeros(input.size(), input.type());
        
        // Calcul du nombre de lignes par thread
        int rowsPerThread = input.rows / NUM_THREADS;
        rowsPerThread = max(rowsPerThread, 10); // Minimum pour éviter overhead des threads
        
        // Déterminer le chevauchement nécessaire (basé sur la taille du noyau)
        int overlap = max(kernelSize.height / 2, 2);
        
        vector<thread> threads;
        for(int i = 0; i < NUM_THREADS; i++) {
            int startRow = i * rowsPerThread;
            int endRow = (i == NUM_THREADS-1) ? input.rows : (i+1) * rowsPerThread;
            
            // Ajouter un chevauchement entre les régions pour éviter les artefacts
            startRow = max(0, startRow - (i > 0 ? overlap : 0));
            endRow = min(input.rows, endRow + (i < NUM_THREADS-1 ? overlap : 0));
            
            // Lancer le thread avec lambda
            threads.emplace_back([&input, &output, startRow, endRow, kernelSize, sigmaX, sigmaY, overlap]() {
                try {
                    // Créer une ROI pour la section à traiter
                    Rect roi(0, startRow, input.cols, endRow - startRow);
                    
                    // Vérifier que la ROI est dans les limites
                    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > input.cols || roi.y + roi.height > input.rows) {
                        std::cerr << "gaussianBlurMultiThread: ROI invalide: " << roi << std::endl;
                        return;
                    }
                    
                    // Extraire la ROI et appliquer le flou gaussien
                    Mat inputROI = input(roi);
                    Mat blurredROI;
                    GaussianBlur(inputROI, blurredROI, kernelSize, sigmaX, sigmaY);
                    
                    // Déterminer quelle partie copier (exclure les zones de chevauchement)
                    int copyStartY = (startRow > 0) ? overlap : 0;
                    int copyHeight = (endRow < input.rows) ? (endRow - startRow - overlap) : (endRow - startRow);
                    
                    if (copyHeight <= 0) {
                        return;
                    }
                    
                    // Copier le résultat dans la matrice de sortie
                    Rect targetRoi(0, startRow + copyStartY, input.cols, copyHeight);
                    Rect sourceRoi(0, copyStartY, input.cols, copyHeight);
                    
                    // Vérifier les dimensions avant de copier
                    if (targetRoi.y + targetRoi.height <= output.rows && 
                        sourceRoi.y + sourceRoi.height <= blurredROI.rows) {
                        blurredROI(sourceRoi).copyTo(output(targetRoi));
                    }
                } catch (const cv::Exception& e) {
                    std::cerr << "OpenCV exception dans thread GaussianBlur: " << e.what() << std::endl;
                }
            });
        }
        
        // Attendre la fin de tous les threads
        for(auto& t : threads) {
            t.join();
        }
        
        return output;
    }
            
    Mat pretraitementImage(const string& imagePath) {
        auto startTotal = high_resolution_clock::now();
        
        // Lecture de l'image
        auto startImage = high_resolution_clock::now();
        Mat input = imread(imagePath);
        if(input.empty()) {
            throw runtime_error("Impossible de charger l'image: " + imagePath);
        }
        input = augmenterConstraste(input);
        auto endImage = high_resolution_clock::now();
        metrics.readImageTime = duration_cast<milliseconds>(endImage - startImage).count();
        

        
        // Création du masque couleur AVANT le flou gaussien
        auto startColor = high_resolution_clock::now();
        Mat colorMask = createRoadLineMasksMultiThread(input);
        auto endColor = high_resolution_clock::now();
        metrics.color = duration_cast<milliseconds>(endColor - startColor).count();
        
        // Conversion en niveaux de gris
        Mat gray;
        cvtColor(input, gray, COLOR_BGR2GRAY);
        
        // Application du flou gaussien APRÈS la création du masque couleur
        auto startGaussienne = high_resolution_clock::now();
        GaussianBlur(gray, gray, Size(5, 5), 1.5); // multithreading pas nécessaire ici
        //gray = gaussianBlurMultiThread(gray, Size(5, 5), 1.5); // multithreading pas nécessaire ici
        // Mat gray = reduireBruitAvance(input); // manque optimisation par rapport au bluring gaussian . 
        auto endGaussienne = high_resolution_clock::now();
        metrics.gaussianTime = duration_cast<milliseconds>(endGaussienne - startGaussienne).count();
        
        // Extraction de la ROI avec l'image floutée
        auto [roi_gray, roi_mask, roi_y, roi_mode] = extractROI(gray, colorMask);
        // Application du filtre de Sobel sur la ROI
        auto startSobel = high_resolution_clock::now();
        Mat edges = cannyMultiThread(roi_gray, 25);
        bitwise_and(edges, roi_mask, edges); // Appliquer le masque de couleur
        auto endSobel = high_resolution_clock::now();
        metrics.sobelTime = duration_cast<milliseconds>(endSobel - startSobel).count();
        
        // Application de la transformée de Hough
        auto startHough = high_resolution_clock::now();
        Mat result = input.clone();//chargement de la matrice à ce niveau ??? TODO a voir si optimal ? 
        bool is_straight = false;
        //direct permissif
        int numLines = houghMultiThreadAdaptative(edges, result, roi_y, roi_mode, is_straight , true); 
        auto endHough = high_resolution_clock::now();

        //Remarque ici le modeSouple peut ou ne pas être pris en compte dans le temps de Hough 
        //TODO éventuellement meilleure 
        metrics.houghTime = duration_cast<milliseconds>(endHough - startHough).count();
        
        // Calcul des métriques finales
        auto endTotal = high_resolution_clock::now();
        metrics.totalTime = duration_cast<milliseconds>(endTotal - startTotal).count();
        metrics.processingFPS = 1000.0 / metrics.totalTime;
        
        return result;
    }

    void printMetrics() {
        // affichage des métriques . 
        cout << "\n=== Métriques de Performance ===" << endl;
        cout << "Temps de lecture de l'image: " << metrics.readImageTime << " ms" << endl;
        cout << "Temps de traitement Gaussienne : " << metrics.gaussianTime << " ms" << endl;
        cout << "Temps de traitement couleur: " << metrics.color << " ms" << endl;
        cout << "Temps de traitement Sobel: " << fixed << setprecision(2) << metrics.sobelTime << " ms" << endl;
        cout << "Temps de traitement Hough: " << metrics.houghTime << " ms" << endl;
        cout << "Temps passé dans mode souple ?? : " << metrics.modeSoupleTime << " ms" << endl;
        cout << "Temps total de traitement: " << metrics.totalTime << " ms" << endl;
        cout << "FPS: " << metrics.processingFPS << endl;
        cout << "Nombre de lignes détectées: " << metrics.numLinesDetected << endl;
        cout << "Longueur moyenne des lignes: " << metrics.averageLineLength << " pixels" << endl;
        cout << "==============================\n" << endl;
    }
};

int main() {
    try {
        DetecteurLignes detector;
        
    
        std::cout << "=== Début récupération données " << endl;
        auto imageFiles = getImagesFromDirectory(input_dir);
        std::cout << "Found " << imageFiles.size() << " images to process" << std::endl;
    
        int processedCount = 0;
        for (const auto& imagePath : imageFiles) {
            std::string filename = imagePath.filename().string();
            std::cout << "\nProcessing " << filename << " (" << (++processedCount) << "/" 
                        << imageFiles.size() << ")..." << std::endl;

            detector.resetMetrics();
            Mat result = detector.pretraitementImage(imagePath.string());
            detector.printMetrics();
            
            fs::path outputPath = fs::path(output_dir) / fs::path("result_" + filename);
            std::cout << "Saving result to: " << outputPath.string() << std::endl;
            
            imwrite(outputPath.string(), result);
        }
        
        std::cout << "\nSuccessfully processed " << processedCount << " image(s)" << std::endl;
        
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
