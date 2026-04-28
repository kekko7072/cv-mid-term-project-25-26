cmake_minimum_required(VERSION 3.16)

project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/app.cpp
    src/localization.cpp
    src/evaluation.cpp
    src/motion_extraction.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    MOTION_EXTRACTION_APP_BUILD
)
