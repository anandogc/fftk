# H5SI
#
# Copyright (C) 2020, Mahendra K. Verma, Anando Gopal Chatterjee
#
# Mahendra K. Verma
# Indian Institute of Technology, Kanpur-208016
# UP, India
#
# mkv@iitk.ac.in
#
# This file is part of H5SI.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
#
# \file  CMakeLists.txt
# @author  M. K. Verma, A. G. Chatterjee
# @date Jan 2017
# @bug  No known bugs


PROJECT(FFTK CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.2)

#CMake usage:
#CXX=mpicxx cmake [-DREAL=FLOAT] [-DFIND_LIBRARIES=OFF] [-DTHREADS=TRUE] [-DCMAKE_BUILD_TYPE=DEBUG] [-DFFTW_PLAN=PATIENT] [-DCMAKE_INSTALL_PREFIX=$HOME/local] /path/to/CMakeLists.txt

#Example:
#CC=mpicc CXX=mpicxx cmake ../fftk/trunk
############################################

#Default options
SET (DEFAULT_REAL "DOUBLE")                      #FLOAT, DOUBLE
SET (DEFAULT_FIND_LIBRARIES ON)                  #ON, OFF - Try to locate packages using pkg-config when ON
SET (DEFAULT_THREADS FALSE)                #TRUE, FALSE
SET (DEFAULT_CMAKE_BUILD_TYPE "DISTRIBUTION")    #DISTRIBUTION, DEBUG
SET (DEFAULT_FFTW_PLAN "MEASURE")                #ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE

SET (INCLUDE_DIRS
        #/path/to/include
    )
SET (LIBRARY_DIRS
       #/path/to/lib
    )

#Required libraries, One of FFTW_FLOAT_LIBS or FFTW_DOUBLE_LIBS is selected depending on REAL
SET (FFTW_FLOAT_LIBS fftw3f_mpi fftw3f)
SET (FFTW_DOUBLE_LIBS fftw3_mpi fftw3)

SET (FFTW_FLOAT_OMP_LIBS fftw3f_omp)
SET (FFTW_DOUBLE_OMP_LIBS fftw3_omp)

SET (OTHER_LIBS blitz)


#################################################
#Users need not modify any thing beyond this line
#################################################

#Set version
SET (VERSION "1")

#Set default value for REAL
IF (NOT DEFINED REAL)
    SET(REAL ${DEFAULT_REAL})
ENDIF()

#Set default value for FIND_LIBRARIES
IF(NOT DEFINED FIND_LIBRARIES)
    SET (FIND_LIBRARIES ${DEFAULT_FIND_LIBRARIES})
ENDIF()

#Set default value for CMAKE_BUILD_TYPE
IF(NOT CMAKE_BUILD_TYPE)
   SET (CMAKE_BUILD_TYPE ${DEFAULT_CMAKE_BUILD_TYPE})
ENDIF()

#Set default value for THREADS
IF(NOT THREADS)
   SET (THREADS ${DEFAULT_THREADS})
ENDIF()

#Set default value for FFTW_PLAN
IF (NOT DEFINED FFTW_PLAN)
    SET(FFTW_PLAN ${DEFAULT_FFTW_PLAN})
ENDIF()

#Check VERSION is not empty
IF (NOT VERSION)
    MESSAGE (FATAL_ERROR "VERSION can not be empty")
ENDIF()

SET (REAL ${REAL} CACHE STRING "REAL")
SET (FIND_LIBRARIES ${FIND_LIBRARIES} CACHE BOOL "FIND_LIBRARIES")
SET (FFTW_PLAN ${FFTW_PLAN} CACHE STRING "FFTW_PLAN")

#Check CXX compiler supports MPI
IF (NOT CXX_SUPPORTS_MPI)        #CXX_SUPPORTS_MPI is undefined when run for the first time.
    MESSAGE(STATUS "MPI support in CXX compiler: ${CMAKE_CXX_COMPILER}")
    FILE(WRITE ${PROJECT_BINARY_DIR}/CMakeFiles/CXX_has_MPI.cc 
               "#include <mpi.h>
                int main(int argc, char** argv)
                { MPI_Init(&argc, &argv);
                  MPI_Finalize();} ")

    EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} ${PROJECT_BINARY_DIR}/CMakeFiles/CXX_has_MPI.cc
                    ERROR_VARIABLE err
                    RESULT_VARIABLE CXX_HAS_MPI)

    IF (CXX_HAS_MPI MATCHES 0)
        MESSAGE(STATUS "MPI support in CXX compiler: ${CMAKE_CXX_COMPILER} -- supported")
    ELSE()
        MESSAGE(FATAL_ERROR "${CMAKE_CXX_COMPILER} does not support MPI")
    ENDIF()
    SET (CXX_SUPPORTS_MPI TRUE CACHE BOOL "CXX_SUPPORTS_MPI")
ENDIF()

#Set version string
ADD_DEFINITIONS(-DVERSION=${VERSION})

#Configure for FFTW library
MESSAGE("")
IF (REAL MATCHES "FLOAT")
    ADD_DEFINITIONS (-DREAL_FLOAT)
    SET(LIB_SUFFIX "f")
ELSEIF(REAL MATCHES "DOUBLE")
    ADD_DEFINITIONS (-DREAL_DOUBLE)
    SET(LIB_SUFFIX "")
ELSE()
    MESSAGE(FATAL_ERROR "REAL must be FLOAT or DOUBLE")
ENDIF()

IF (NOT REAL_DISPLAYED)
	MESSAGE(STATUS "REAL: ${REAL}")
    SET(REAL_DISPLAYED TRUE CACHE BOOL "REAL_DISPLAYED")
ENDIF()

#Find the required libraries
IF (NOT ALL_LIBRARIES_FOUND AND FIND_LIBRARIES)
    FUNCTION(DISPLAY_FOUND_LIBRARY DISPLAY_NAME LIB_NAME INCLUDE_PATH ERROR_MESSAGE)
        IF (${LIB_NAME}_FOUND)
            EXECUTE_PROCESS(COMMAND echo ${INCLUDE_PATH}
                            COMMAND rev
                            COMMAND cut -d/ -f2-
                            COMMAND rev
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                            OUTPUT_VARIABLE ${LIB_NAME}_ROOT_DIR)

            MESSAGE(STATUS "Found ${DISPLAY_NAME}: ${${LIB_NAME}_ROOT_DIR} (version \"${${LIB_NAME}_VERSION}\")")
        ELSE()
            MESSAGE(FATAL_ERROR "${DISPLAY_NAME} not found. ${ERROR_MESSAGE}")
        ENDIF()
    ENDFUNCTION()

#Check whether pkg-config is installed
    EXECUTE_PROCESS(COMMAND which pkg-config
                    RESULT_VARIABLE PKG_CONFIG_FOUND
                    OUTPUT_VARIABLE PKG_CONFIG_LOCATION)

    IF (PKG_CONFIG_FOUND MATCHES 0)
        FIND_PACKAGE(PkgConfig REQUIRED)
        
        PKG_CHECK_MODULES(BLITZ QUIET blitz)
        DISPLAY_FOUND_LIBRARY(Blitz++ BLITZ "${BLITZ_INCLUDE_DIRS}" "Set PKG_CONFIG_PATH, e.g. $HOME/local/lib/pkgconfig") 
        
        IF (REAL MATCHES "FLOAT")
            PKG_CHECK_MODULES(FFTW3F QUIET fftw3f)
            DISPLAY_FOUND_LIBRARY(FFTW3F FFTW3F "${FFTW3F_INCLUDE_DIRS}" "Set PKG_CONFIG_PATH, e.g. $HOME/local/lib/pkgconfig") 
        ELSEIF(REAL MATCHES "DOUBLE")
            PKG_CHECK_MODULES(FFTW3 QUIET fftw3)
            DISPLAY_FOUND_LIBRARY(FFTW3 FFTW3 "${FFTW3_INCLUDE_DIRS}" "Set PKG_CONFIG_PATH, e.g. $HOME/local/lib/pkgconfig")
        ELSE()
            MESSAGE(FATAL_ERROR "REAL must be FLOAT or DOUBLE")
        ENDIF()
    ELSE()
        MESSAGE(STATUS "Warning: pkg-config not found, libraries will not be searched for. Supply -DFIND_LIBRARIES=OFF to disable library search.")
    ENDIF()

    IF (THREADS)
       FIND_PACKAGE(OpenMP)
       IF (OPENMP_FOUND)
           SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
           SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
           ADD_DEFINITIONS(-DFFTK_THREADS)

           IF (REAL MATCHES "FLOAT")
               SET(SYSTEM_LIBRARIES ${SYSTEM_LIBRARIES} ${FFTW_FLOAT_OMP_LIBS})
           ELSEIF(REAL MATCHES "DOUBLE")
               SET(SYSTEM_LIBRARIES ${SYSTEM_LIBRARIES} ${FFTW_DOUBLE_OMP_LIBS})
           ENDIF()

       ENDIF()
    ENDIF()

    SET(ALL_LIBRARIES_FOUND TRUE CACHE BOOL "ALL_LIBRARIES_FOUND")
ENDIF()


#Set FFTW_PLAN
IF (FFTW_PLAN MATCHES "ESTIMATE")
    ADD_DEFINITIONS(-DESTIMATE)
ELSEIF (FFTW_PLAN MATCHES "MEASURE")
    ADD_DEFINITIONS(-DMEASURE)
ELSEIF (FFTW_PLAN MATCHES "PATIENT")
    ADD_DEFINITIONS(-DPATIENT)
ELSEIF (FFTW_PLAN MATCHES "EXHAUSTIVE")
    ADD_DEFINITIONS(-DPATIENT)
ELSE()
    MESSAGE(FATAL_ERROR "FFTW_PLAN can be one of ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE.")
ENDIF()

IF (NOT FFTW_PLAN_DISPLAYED)
    MESSAGE(STATUS "FFTW_PLAN: ${FFTW_PLAN}")
    SET (FFTW_PLAN_DISPLAYED TRUE CACHE BOOL "FFTW_PLAN_DISPLAYED")
    MESSAGE("")
ENDIF()


#Folders where header files will be searched for
INCLUDE_DIRECTORIES(
${CMAKE_SOURCE_DIR}/lib/
${BLITZ_INCLUDE_DIRS}
${FFTW3F_INCLUDE_DIRS}
${FFTW3_INCLUDE_DIRS}
${INCLUDE_DIRS}
)

#Folders where libraries will be searched for
LINK_DIRECTORIES(
${BLITZ_LIBRARY_DIRS}
${FFTW3F_LIBRARY_DIRS}
${FFTW3_LIBRARY_DIRS}
${LIBRARY_DIRS}
)


IF (REAL MATCHES "FLOAT")
    SET(SYSTEM_LIBRARIES ${SYSTEM_LIBRARIES} ${FFTW_FLOAT_LIBS} ${FFTW3F_LIBRARIES})
ELSEIF(REAL MATCHES "DOUBLE")
    SET(SYSTEM_LIBRARIES ${SYSTEM_LIBRARIES} ${FFTW_DOUBLE_LIBS} ${FFTW3_LIBRARIES})
ENDIF()

SET(SYSTEM_LIBRARIES ${SYSTEM_LIBRARIES} ${OTHER_LIBS})

#Build type
IF (NOT CMAKE_BUILD_TYPE_DISPLAYED)
    IF ( (NOT ${CMAKE_BUILD_TYPE} MATCHES "DISTRIBUTION") AND (NOT ${CMAKE_BUILD_TYPE} MATCHES "DEBUG") )
        MESSAGE(FATAL_ERROR "Build type can be one of DISTRIBUTION, DEBUG")
    ENDIF()

    MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
    SET (CMAKE_BUILD_TYPE_DISPLAYED TRUE CACHE BOOL "CMAKE_BUILD_TYPE_DISPLAYED")
ENDIF ()

MESSAGE("")

SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3")

SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

ADD_SUBDIRECTORY(lib)
ADD_SUBDIRECTORY(examples)



