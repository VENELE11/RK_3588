cmake_minimum_required(VERSION 3.9)
project(rknn_deploy VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 17)

#asan
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
endif()

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")    #添加一行，指向自己编译的opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP)

#glob source files
file(GLOB SRC_FILES *.cpp)

add_executable(rknn_deploy ${SRC_FILES})

target_link_libraries(rknn_deploy PUBLIC
    ${CMAKE_SOURCE_DIR}/lib/librknnrt.so
    ${OpenCV_LIBS}
    pthread
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(rknn_deploy PUBLIC OpenMP::OpenMP_CXX)
endif()
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
