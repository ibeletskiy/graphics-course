
add_executable(local_shadertoy2
  main.cpp
  App.cpp
)

target_link_libraries(local_shadertoy2
  PRIVATE glfw etna glm::glm wsi gui)

target_add_shaders(local_shadertoy2
  shaders/fullscreen.vert
  shaders/texture.frag
  shaders/toy.frag
)

target_compile_definitions(local_shadertoy2 PRIVATE
  LOCAL_SHADERTOY2_CMAKE="${CMAKE_COMMAND}"
  LOCAL_SHADERTOY2_BUILD_DIR="${CMAKE_BINARY_DIR}"
  LOCAL_SHADERTOY2_CONFIG="$<CONFIG>"
)
