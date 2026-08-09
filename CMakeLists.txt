cmake_minimum_required(VERSION 3.21)
project(luna-shougi CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
  add_compile_options(/W4 /utf-8)
else()
  add_compile_options(-Wall -Wextra)
endif()

option(LUNA_BUILD_TESTS "Build unit tests" ON)

add_library(luna_engine
  src/core/attacks.cpp
  src/core/move.cpp
  src/core/movegen.cpp
  src/core/perft.cpp
  src/core/position.cpp
  src/core/zobrist.cpp
  src/search/eval.cpp
  src/search/search.cpp
  src/search/timeman.cpp
  src/search/tt.cpp
  src/engine/usi_engine.cpp
)
target_include_directories(luna_engine PUBLIC src)
target_compile_features(luna_engine PUBLIC cxx_std_20)

add_executable(luna-shougi src/main.cpp)
target_link_libraries(luna-shougi PRIVATE luna_engine)

if(LUNA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
