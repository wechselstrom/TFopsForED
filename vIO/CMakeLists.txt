cmake_minimum_required(VERSION 2.8)
 
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
 
# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(YARP REQUIRED)
#find_package(EVENTDRIVEN REQUIRED)

 
include_directories(${Tensorflow_INCLUDE_DIRS} ${YARP_INCLUDE_DIRS})
add_library(vInput SHARED vInput.cc)
add_library(vOutput SHARED vOutput.cc)
target_link_libraries(vInput ${YARP_LIBRARIES} ${EVENTDRIVEN_LIBRARIES} eventdriven)
target_link_libraries(vOutput ${YARP_LIBRARIES} ${EVENTDRIVEN_LIBRARIES} eventdriven)
 
#include_directories(${Tensorflow_INCLUDE_DIRS})
#add_library(inner_product_grad SHARED inner_product_grad.cc)
