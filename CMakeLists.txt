cmake_minimum_required(VERSION 3.23)
project(complex_cp)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
        BTAS
        GIT_REPOSITORY      https://github.com/ValeevGroup/btas.git
)
FetchContent_MakeAvailable(BTAS)
FetchContent_GetProperties(BTAS
        SOURCE_DIR BTAS_SOURCE_DIR
        BINARY_DIR BTAS_BINARY_DIR
        )
#add_library(BTAS)
message(STATUS "BTAS: ${BTAS}")


add_executable(complex_cp
        comp_cp.cpp)

target_link_libraries(complex_cp BTAS)