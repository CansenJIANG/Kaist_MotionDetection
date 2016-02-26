#-------------------------------------------------
#
# Project created by QtCreator 2016-02-26T11:26:36
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = KaistMovingObject
CONFIG   += console
CONFIG   -= app_bundle
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS +=  -fopenmp
QMAKE_CXXFLAGS += -msse3

TEMPLATE = app
INCLUDEPATH += /usr/local/include/opencv-2.4.10
INCLUDEPATH += /usr/include/eigen3
LIBS += `pkg-config opencv-2.4.10 –cflags –libs`
LIBS += -lopencv_core
LIBS += -lopencv_imgproc
LIBS += -lopencv_highgui
LIBS += -lopencv_ml
LIBS += -lopencv_video
LIBS += -lopencv_features2d
LIBS += -lopencv_calib3d
LIBS += -lopencv_objdetect
LIBS += -lopencv_contrib
LIBS += -lopencv_legacy
LIBS += -lopencv_flann
LIBS += -lopencv_gpu
LIBS += -lopencv_nonfree
LIBS += -fopenmp
LIBS += -lceres -lcsparse -lcholmod -fopenmp -lgflags -L/usr/local/lib -lglog -pthread -lprotobuf -lz -lpthread
LIBS += -lcsparse -lcholmod -lcblas -lblas -llapack

SOURCES += main.cpp \
    visualodom.cpp \
    stereocam.cpp \
    MiscTool.cpp \
    elasFiles/triangle.cpp \
    elasFiles/matrix.cpp \
    elasFiles/filter.cpp \
    elasFiles/elas.cpp \
    elasFiles/descriptor.cpp \
    MotionDetect.cpp

HEADERS += \
    visualodom.h \
    stereocam.h \
    MiscTool.h \
    elasFiles/triangle.h \
    elasFiles/timer.h \
    elasFiles/matrix.h \
    elasFiles/image.h \
    elasFiles/filter.h \
    elasFiles/elas.h \
    elasFiles/descriptor.h \
    MotionDetect.h
