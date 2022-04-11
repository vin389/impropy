QT -= gui

TEMPLATE = lib
DEFINES += C_PICKTM_LIBRARY

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    c_picktm.cpp

HEADERS +=

# Settings for OpenCV (Windows)
win32:CONFIG(release, debug|release): \
    LIBS += -LC:/opencv/opencv411/opencv/build/x64/vc15/lib/ \
    -lopencv_world411
else:win32:CONFIG(debug, debug|release): \
    LIBS += -LC:/opencv/opencv411/opencv/build/x64/vc15/lib/ \
    -lopencv_world411d
INCLUDEPATH += C:/opencv/opencv411/opencv/build/include


# Default rules for deployment.
unix {
    target.path = /usr/lib
}
!isEmpty(target.path): INSTALLS += target
