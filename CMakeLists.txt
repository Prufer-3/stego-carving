cmake_minimum_required(VERSION 3.10)
project(stegocarver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(COMPILER_FLAGS -Wall -Wextra -Wpedantic -Werror)
# ASan/UBsan are enabled only in Debug builds for CI testing
set(SANITIZER_FLAGS -fsanitize=address,undefined)

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
add_library(stegocarver_core src/picture.cpp src/seam_carver.cpp)
target_include_directories(stegocarver_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stegocarver_core PUBLIC ${OpenCV_LIBS})
target_compile_options(stegocarver_core PRIVATE ${COMPILER_FLAGS} "$<$<CONFIG:Debug>:${SANITIZER_FLAGS}>")

# Tests
add_executable(tests tests/picture_tests.cpp tests/seam_carver_tests.cpp)
target_link_libraries(tests PRIVATE stegocarver_core Catch2::Catch2WithMain)
target_compile_options(tests PRIVATE ${COMPILER_FLAGS} "$<$<CONFIG:Debug>:${SANITIZER_FLAGS}>")
set_property(TARGET tests APPEND_STRING PROPERTY LINK_FLAGS_DEBUG " ${SANITIZER_FLAGS}")
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(tests)