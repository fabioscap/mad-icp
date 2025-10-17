add_library(tools STATIC
  mad_tree.cpp
  mad_tree_old.cpp
)
target_link_libraries(tools PUBLIC Eigen3::Eigen)
target_compile_features(tools PUBLIC)