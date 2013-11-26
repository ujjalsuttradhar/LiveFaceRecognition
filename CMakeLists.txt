project(livefacerecognition)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
TARGET_LINK_LIBRARIES(livefacerecognition opencv_core opencv_highgui opencv_imgproc opencv_objdetect opencv_contrib)

