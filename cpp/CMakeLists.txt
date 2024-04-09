cmake_minimum_required(VERSION 3.15)
project(sprl VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TODO: this is temporary to re-enable asserts in release mode
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

include(CTest)
enable_testing()

list(APPEND CMAKE_PREFIX_PATH "C:\\libtorch\\share\\cmake\\Torch")
find_package(Torch REQUIRED)

set(src_dir ${PROJECT_SOURCE_DIR}/src)

include_directories(${src_dir})

file(GLOB_RECURSE srcs
    ${src_dir}/*/*.cpp
)

file(GLOB headers
    ${src_dir}/*/*.hpp
)

add_executable(self_play src/self_play.cpp ${srcs} ${headers})
add_executable(challenge src/challenge.cpp ${srcs} ${headers})

target_link_libraries(self_play "${TORCH_LIBRARIES}")
target_link_libraries(challenge "${TORCH_LIBRARIES}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
