
add_executable(skybox
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  Rocks.cpp
  Sky.cpp
)

target_link_libraries(skybox
  PRIVATE glfw etna glm::glm wsi gui scene)

target_add_shaders(skybox
  shaders/terrain.frag
  shaders/terrain.vert
  shaders/rocks.vert
  shaders/rocks.frag
  shaders/fullscreen.vert
  shaders/lighting.frag
  shaders/details.frag
  shaders/sky.comp
)
