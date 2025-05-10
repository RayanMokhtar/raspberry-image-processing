CXX = g++
CXXFLAGS = -std=c++11 -Wall -g `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4` -lpthread

# Directories
SRC_DIR = .
CORE_DIR = $(SRC_DIR)/core
ACQ_DIR = $(SRC_DIR)/acquisition
PREP_DIR = $(SRC_DIR)/preprocessing
OBJ_DIR = ./obj

# Source files
CORE_SRC = $(wildcard $(CORE_DIR)/*.cpp)
ACQ_SRC = $(wildcard $(ACQ_DIR)/*.cpp)
PREP_SRC = $(wildcard $(PREP_DIR)/*.cpp)
MAIN_SRC = $(SRC_DIR)/main.cpp

# Object files
CORE_OBJ = $(patsubst $(CORE_DIR)/%.cpp, $(OBJ_DIR)/core/%.o, $(CORE_SRC))
ACQ_OBJ = $(patsubst $(ACQ_DIR)/%.cpp, $(OBJ_DIR)/acquisition/%.o, $(ACQ_SRC))
PREP_OBJ = $(patsubst $(PREP_DIR)/%.cpp, $(OBJ_DIR)/preprocessing/%.o, $(PREP_SRC))
MAIN_OBJ = $(OBJ_DIR)/main.o

# Target executable
TARGET = line_detector

# Create directories
$(shell mkdir -p $(OBJ_DIR)/core $(OBJ_DIR)/acquisition $(OBJ_DIR)/preprocessing)

# Phony targets
.PHONY: all clean test_acquisition test_preprocessing test_metrics

all: $(TARGET)

# Main target
$(TARGET): $(MAIN_OBJ) $(CORE_OBJ) $(ACQ_OBJ) $(PREP_OBJ)
    $(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Main object file
$(OBJ_DIR)/main.o: $(MAIN_SRC)
    $(CXX) $(CXXFLAGS) -c -o $@ $<

# Core module object files
$(OBJ_DIR)/core/%.o: $(CORE_DIR)/%.cpp
    $(CXX) $(CXXFLAGS) -c -o $@ $<

# Acquisition module object files
$(OBJ_DIR)/acquisition/%.o: $(ACQ_DIR)/%.cpp
    $(CXX) $(CXXFLAGS) -c -o $@ $<

# Preprocessing module object files
$(OBJ_DIR)/preprocessing/%.o: $(PREP_DIR)/%.cpp
    $(CXX) $(CXXFLAGS) -c -o $@ $<

# Test targets
test_acquisition:
    $(MAKE) -C $(ACQ_DIR)

test_preprocessing: $(PREP_OBJ) $(CORE_OBJ)
    $(CXX) $(CXXFLAGS) -o test_preprocessing $(PREP_DIR)/test_preprocessing.cpp $(PREP_OBJ) $(CORE_OBJ) $(LDFLAGS)

test_metrics: $(CORE_OBJ)
    $(CXX) $(CXXFLAGS) -o test_metrics $(CORE_DIR)/test_metrics.cpp $(CORE_OBJ) $(LDFLAGS)

# Clean
clean:
    rm -rf $(OBJ_DIR)/* $(TARGET) test_preprocessing test_metrics
    $(MAKE) -C $(ACQ_DIR) clean