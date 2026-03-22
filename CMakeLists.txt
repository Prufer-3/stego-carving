cmake_minimum_required(VERSION 3.10)
project(stegocarver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(COMPILER_FLAGS -Wall -Wextra -Wpedantic -Werror)
set(SANITIZER_FLAGS -fsanitize=address,undefined)
set(COVERAGE_FLAGS --coverage)

IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Debug
       CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel Coverage."
       FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
message("* Current build type is : ${CMAKE_BUILD_TYPE}")

# Dependencies
Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.13.0
)
FetchContent_MakeAvailable(Catch2)

find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# Build
add_library(stegocarver_core src/picture.cpp src/seamcarver.cpp)
target_include_directories(stegocarver_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stegocarver_core PUBLIC ${OpenCV_LIBS})
target_compile_options(stegocarver_core PRIVATE
  ${COMPILER_FLAGS}
  $<$<CONFIG:Debug>:${SANITIZER_FLAGS}>
  $<$<CONFIG:Coverage>:${COVERAGE_FLAGS}>
)

# Tests
add_executable(tests tests/picture_tests.cpp tests/seamcarver_tests.cpp)
target_link_libraries(tests PRIVATE stegocarver_core Catch2::Catch2WithMain)
target_compile_options(tests PRIVATE
  ${COMPILER_FLAGS}
  $<$<CONFIG:Debug>:${SANITIZER_FLAGS}>
  $<$<CONFIG:Coverage>:${COVERAGE_FLAGS}>
)
target_link_options(tests PRIVATE
  $<$<CONFIG:Debug>:${SANITIZER_FLAGS}>
  $<$<CONFIG:Coverage>:${COVERAGE_FLAGS}>
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(tests)