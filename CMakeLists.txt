cmake_minimum_required( VERSION 3.4.3 )

# Project name and programming language
project( rotst CXX )

#build type is set here
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Gather files and store paths in CMake variables
file( GLOB LIBRARY_SOURCE_FILES library/src/*.cpp )
#file( GLOB TEST_SOURCE_FILES library/test/*.cpp )
file( GLOB HEADER_FILES inc/*.h* )

# Specify the folders with header files, otherwise our includes are not found
include_directories( ../eigen )
include_directories( library/inc )

# When compiling with gcc we need to compile libraries with fPIC (stands for
# position independent code).
if( CMAKE_COMPILER_IS_GNUCXX )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -Wall -Wextra" )
endif( CMAKE_COMPILER_IS_GNUCXX )


# Add compilation target for our library
add_library( rotst SHARED ${LIBRARY_SOURCE_FILES} ${HEADER_FILES} )

# Add compilation target for our executables
#add_executable( vectorkernel_testrunner ${TEST_SOURCE_FILES} ${HEADER_FILES} )
add_executable( main drivers/main.cpp ${HEADER_FILES} )

#eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# Add link depenencies
#target_link_libraries( vectorkernel_testrunner vectorkernel )
target_link_libraries (rotst  Eigen3::Eigen)
target_link_libraries( main rotst )
