cmake_minimum_required(VERSION 3.12)

project(vtkRender)

find_package(VTK COMPONENTS 
  CommonColor
  CommonCore
  FiltersGeometry
  FiltersSources
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
)

if (NOT VTK_FOUND)
  message(FATAL_ERROR "Unable to find the VTK build folder.")
endif()

# Prevent a "command line is too long" failure in Windows.
set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")

add_library(vtkRender STATIC ${PROJECT_SOURCE_DIR}/include/render.h)

if (VTK094)
    target_compile_definitions(vtkRender PUBLIC -DVTK094=1)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include/)

target_sources(vtkRender PRIVATE ${PROJECT_SOURCE_DIR}/src/render.cxx)

set_target_properties(vtkRender PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(vtkRender PRIVATE ${VTK_LIBRARIES})

install(TARGETS vtkRender DESTINATION .)

# add_executable(Cylinder MACOSX_BUNDLE src/render.cxx )
#   target_link_libraries(Cylinder PRIVATE ${VTK_LIBRARIES}
# )

# vtk_module_autoinit is needed
vtk_module_autoinit(
  TARGETS vtkRender
  MODULES ${VTK_LIBRARIES}
)

# vtk_module_autoinit is needed
# vtk_module_autoinit(
#   TARGETS Cylinder
#   MODULES ${VTK_LIBRARIES}
# )
