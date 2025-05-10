all: line_detector

# Répertoires de sortie
DIR_OBJ = ./objs
DIR_EXEC = ./

# Flags de compilation
CFLAGS = -g -Wall -std=c++11

# Création des répertoires d'objets si nécessaires
$(shell mkdir -p $(DIR_OBJ))
$(shell mkdir -p $(DIR_OBJ)/core)
$(shell mkdir -p $(DIR_OBJ)/acquisition)
$(shell mkdir -p $(DIR_OBJ)/preprocessing)

# Règle principale pour la compilation du programme
line_detector: main.cpp acquisition/acquisition.cpp preprocessing/preprocessing.cpp core/metrics.cpp core/thread_pool.cpp
    g++ -o line_detector main.cpp acquisition/acquisition.cpp preprocessing/preprocessing.cpp core/metrics.cpp core/thread_pool.cpp `pkg-config --cflags --libs opencv4` $(CFLAGS) -lpthread

# Règle pour tester uniquement le module d'acquisition
test_acquisition: acquisition/test_acquisition.cpp acquisition/acquisition.cpp
    g++ -o test_acquisition acquisition/test_acquisition.cpp acquisition/acquisition.cpp `pkg-config --cflags --libs opencv4` $(CFLAGS)

# Règle pour tester uniquement le module de prétraitement
test_preprocessing: preprocessing/test_preprocessing.cpp preprocessing/preprocessing.cpp core/metrics.cpp
    g++ -o test_preprocessing preprocessing/test_preprocessing.cpp preprocessing/preprocessing.cpp core/metrics.cpp `pkg-config --cflags --libs opencv4` $(CFLAGS) -lpthread

# Nettoyage
clean:
    rm -f line_detector test_acquisition test_preprocessing
    rm -rf $(DIR_OBJ)