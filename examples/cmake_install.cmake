# Install script for directory: /home/ritesh/Downloads/dealii_local/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "DebugRelease")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "examples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/examples" TYPE DIRECTORY MESSAGE_LAZY FILES "/home/ritesh/Downloads/dealii_local/examples/" FILES_MATCHING REGEX "/CMakeFiles[^/]*$" EXCLUDE REGEX "/doc[^/]*$" EXCLUDE REGEX "/[^/]*\\.cu$" REGEX "/[^/]*\\.cc$" REGEX "/[^/]*\\.prm$" REGEX "/[^/]*\\.inp$" REGEX "/[^/]*\\.ipynb$" REGEX "/step[^/]*\\/CMakeLists\\.txt$" REGEX "/output\\.reference\\.dat$" REGEX "/postprocess\\.pl$" REGEX "/obstacle\\.pbm$" REGEX "/example\\.geo$" REGEX "/example\\.msh$" REGEX "/topography\\.txt\\.gz$" REGEX "/input\\/initial\\_mesh\\_3d\\.vtk$" REGEX "/input\\/DTMB\\-5415\\_bulbous\\_bow\\.iges$" REGEX "/sphere\\_r6\\.geo$" REGEX "/sphere\\_r6\\.msh$" REGEX "/sphere\\_r7\\.geo$" REGEX "/sphere\\_r7\\.msh$" REGEX "/sphere\\_r8\\.geo$" REGEX "/sphere\\_r8\\.msh$" REGEX "/sphere\\_r9\\.geo$" REGEX "/sphere\\_r9\\.msh$")
endif()

