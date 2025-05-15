all : main


VERSION = 0.3


DIR_OBJ = ./objs
DIR_EXEC = ./


LIBRARIES = -lGL -lGLU -lglut -lm -lX11   # LINUX
#LIBRARIES = -L"/System/Library/Frameworks/OpenGL.framework/Libraries" # MacOs

FRAMEWORK = -framework GLUT   #Macos
FRAMEWORK += -framework OpenGL #Macos

LIBPATH += $(LIBRARIES)


COMPILERFLAGS = -Wall


CFLAGS = $(COMPILERFLAGS)


main : main.cpp
	g++ -std=c++17 -O3 -Wall -g main.cpp -o main `pkg-config --cflags --libs opencv4` -pthread

run : main
	./main

clean :
	rm  $(DIR_OBJ)/*.o ; rm main ;
