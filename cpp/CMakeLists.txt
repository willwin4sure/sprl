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

add_executable(Challenge src/Challenge.cpp ${srcs} ${headers})
add_executable(Evaluate src/Evaluate.cpp ${srcs} ${headers})
add_executable(TestGame src/TestGame.cpp ${srcs} ${headers})
add_executable(C4SelfPlay src/C4SelfPlay.cpp ${srcs} ${headers})
# add_executable(PTGSelfPlay src/PTGSelfPlay.cpp ${srcs} ${headers})

target_link_libraries(Challenge "${TORCH_LIBRARIES}")
target_link_libraries(Evaluate "${TORCH_LIBRARIES}")
target_link_libraries(TestGame "${TORCH_LIBRARIES}")
target_link_libraries(C4SelfPlay "${TORCH_LIBRARIES}")
# target_link_libraries(PTGSelfPlay "${TORCH_LIBRARIES}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
