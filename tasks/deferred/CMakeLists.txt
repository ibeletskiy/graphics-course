
add_executable(deferred
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  Rocks.cpp
)

target_link_libraries(deferred
  PRIVATE glfw etna glm::glm wsi gui scene)

target_add_shaders(deferred
  shaders/terrain.frag
  shaders/terrain.vert
  shaders/rocks.vert
  shaders/rocks.frag
  shaders/fullscreen.vert
  shaders/lighting.frag
  shaders/details.frag
)
