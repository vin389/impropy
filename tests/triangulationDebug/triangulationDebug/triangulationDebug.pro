TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

win32 {
    win32:CONFIG(release, debug|release): LIBS += -LC:/opencv/opencv451x/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_world451
    else:win32:CONFIG(debug, debug|release): LIBS += -LC:/opencv/opencv451x/opencv-4.5.1/build/install/x64/vc16/lib/ -lopencv_world451d
    INCLUDEPATH += C:/opencv/opencv451x/opencv-4.5.1/build/install/include
    DEPENDPATH += C:/opencv/opencv451x/opencv-4.5.1/build/install/include
}
macx {
    macx: LIBS += -L /usr/local/Cellar/opencv/4.5.0_5/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
    INCLUDEPATH += /usr/local/Cellar/opencv/4.5.0_5/include/opencv4
    DEPENDPATH += /usr/local/Cellar/opencv/4.5.0_5/include/opencv4v
}

