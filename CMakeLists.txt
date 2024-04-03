cmake_minimum_required(VERSION 3.0.0)
project(RGBD_Reconstruction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(source_files
    src/pcdMapping.cpp
)

add_executable(pcdMapping ${source_files})
target_link_libraries(pcdMapping ${OpenCV_LIBS})
target_link_libraries(pcdMapping Eigen3::Eigen)
target_link_libraries(pcdMapping ${PCL_LIBRARIES})

set(source_files
    src/surfelMapping.cpp
)

add_executable(surfelMapping ${source_files})
target_link_libraries(surfelMapping ${OpenCV_LIBS})
target_link_libraries(surfelMapping Eigen3::Eigen)
target_link_libraries(surfelMapping ${PCL_LIBRARIES})

set(source_files
    src/octomapMapping.cpp
)

add_executable(octomapMapping ${source_files})
target_link_libraries(octomapMapping ${OpenCV_LIBS})
target_link_libraries(octomapMapping Eigen3::Eigen)
target_link_libraries(octomapMapping ${PCL_LIBRARIES})

# Add the -pthread flag to the compiler and linker flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
