cmake_minimum_required(VERSION 3.22)
project(sprl VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The following line is for if you want to re-enable asserts in release mode.
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

include(CTest)
enable_testing()

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
# add_executable(TestGame src/TestGame.cpp ${srcs} ${headers})
# add_executable(TestSymmetries src/TestSymmetries.cpp ${srcs} ${headers})
# add_executable(C4Worker src/C4Worker.cpp ${srcs} ${headers})
# add_executable(OTHWorker src/OTHWorker.cpp ${srcs} ${headers})
add_executable(GoWorker src/GoWorker.cpp ${srcs} ${headers})
add_executable(TestJson src/TestJson.cpp ${srcs} ${headers})
# add_executable(RobinWorker src/RobinWorker.cpp ${srcs} ${headers})
# add_executable(Time src/Time.cpp ${srcs} ${headers})

target_link_libraries(Challenge ${TORCH_LIBRARIES})
target_link_libraries(Evaluate ${TORCH_LIBRARIES})
# target_link_libraries(TestGame ${TORCH_LIBRARIES})
# target_link_libraries(TestSymmetries ${TORCH_LIBRARIES})
# target_link_libraries(C4Worker ${TORCH_LIBRARIES})
# target_link_libraries(OTHWorker ${TORCH_LIBRARIES})
target_link_libraries(GoWorker ${TORCH_LIBRARIES})
target_link_libraries(TestJson ${TORCH_LIBRARIES})
# target_link_libraries(RobinWorker ${TORCH_LIBRARIES})
# target_link_libraries(Time ${TORCH_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# add_subdirectory(tests)