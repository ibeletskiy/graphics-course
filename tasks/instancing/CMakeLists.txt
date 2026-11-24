
add_executable(instancing
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  Rocks.cpp
)

target_link_libraries(instancing
  PRIVATE glfw etna glm::glm wsi gui scene)

target_add_shaders(instancing
  shaders/terrain.frag
  shaders/terrain.vert
  shaders/rocks.vert
  shaders/rocks.frag
)
