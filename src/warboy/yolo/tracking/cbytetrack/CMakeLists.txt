project(bytetrack)

add_definitions(-std=c++11)

set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include_directories(/usr/local/include/eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(bytetrack SHARED ${My_Source_Files})
# target_link_libraries(bytetrack /Users/kevin/anaconda3/envs/py39torch/lib/python3.9/site-packages/cv2/cv2.abi3.so)

add_definitions(-O2 -pthread)
