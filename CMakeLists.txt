cmake_minimum_required(VERSION 2.8)
project( stereo_block_match )

set( INCLUDE_DIR "./srcs" )
set( SRCS )
aux_source_directory( "./srcs" SRCS)

set( ld_flag /home/choyg/workspace/opencv_ffmpeg/opencv-3.4.2/install/lib )
set( include_flag /home/choyg/workspace/opencv_ffmpeg/opencv-3.4.2/install/include )
set( libs opencv_img_hash opencv_world OpenMP::OpenMP_CXX )
find_package(OpenMP REQUIRED)

include_directories( ${include_flag} ${INCLUDE_DIR} )
link_directories( ${ld_flag} )

add_executable( "stereo_block_match.out" ${SRCS} ${PLANE_TEST_SRC} )

target_link_libraries(stereo_block_match.out ${libs} ${CMAKE_THREAD_LIBS_INIT} ${CERES_LIBRARIES} )
