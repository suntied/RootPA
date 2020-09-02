cmake_minimum_required(VERSION 3.16)
project(2020_3A_IABD1_Correction_modele_lineaire)

set(CMAKE_CXX_STANDARD 14)

include_directories("${PROJECT_SOURCE_DIR}\\Eigen")

add_library(2020_3A_IABD1_Correction_modele_lineaire SHARED library.cpp library.h LinearModel.cpp LinearModel.h)
target_link_libraries(2020_3A_IABD1_Correction_modele_lineaire ${PROJECT_SOURCE_DIR}\\Eigen)