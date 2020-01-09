# Install script for directory: /home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build/../../..")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglut.so.3.10.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglut.so.3"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglut.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build/lib/libglut.so.3.10.0"
    "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build/lib/libglut.so.3"
    "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build/lib/libglut.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglut.so.3.10.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglut.so.3"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglut.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build/lib/libglut.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GL" TYPE FILE FILES
    "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0/include/GL/freeglut.h"
    "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0/include/GL/freeglut_ext.h"
    "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0/include/GL/freeglut_std.h"
    "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0/include/GL/glut.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE RENAME "freeglut.pc" FILES "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build/freeglut.pc")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/kvnsnchz/Documents/ENSTA/PP/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
