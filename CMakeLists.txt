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

# NNUE inference is int8/int16 arithmetic that only pays off vectorized. On
# x86-64 that means AVX2, which every desktop CPU since 2013 has and which the
# compiler will not emit without being told. arm64 gets NEON unconditionally,
# because it is part of the base architecture there and needs no flag.
#
# Turn this off to produce a binary for a pre-AVX2 machine; the scalar kernels
# are always compiled in and the engine will simply evaluate more slowly.
option(LUNA_ENABLE_AVX2 "Compile with AVX2 on x86-64" ON)

# The width of the feature transformer, which src/nnue/network.hpp defaults to
# 256. A network file records its own shape and the engine refuses one that
# does not match the binary, so comparing two widths means building two
# binaries and giving both to luna-match:
#
#   cmake -B build-wide -DLUNA_TRANSFORMED_DIM=512
set(LUNA_TRANSFORMED_DIM "" CACHE STRING
    "Feature transformer width; empty keeps network.hpp's default of 256")

# The search itself is multi-threaded (Lazy SMP), so the engine library needs
# a thread implementation, not only the tools that drive several engines.
find_package(Threads REQUIRED)

add_library(luna_engine
  src/core/attacks.cpp
  src/core/move.cpp
  src/core/movegen.cpp
  src/core/perft.cpp
  src/core/position.cpp
  src/core/zobrist.cpp
  src/nnue/evaluate.cpp
  src/nnue/features.cpp
  src/nnue/network.cpp
  src/search/eval.cpp
  src/search/evaldump.cpp
  src/search/pst.cpp
  src/search/search.cpp
  src/search/see.cpp
  src/search/timeman.cpp
  src/search/tt.cpp
  src/engine/bench.cpp
  src/engine/usi_engine.cpp
)
target_include_directories(luna_engine PUBLIC src)
if(LUNA_TRANSFORMED_DIM)
  target_compile_definitions(luna_engine PUBLIC LUNA_TRANSFORMED_DIM=${LUNA_TRANSFORMED_DIM})
endif()
target_compile_features(luna_engine PUBLIC cxx_std_20)
target_link_libraries(luna_engine PUBLIC Threads::Threads)

if(LUNA_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
  if(MSVC)
    target_compile_options(luna_engine PUBLIC /arch:AVX2)
  else()
    target_compile_options(luna_engine PUBLIC -mavx2)
  endif()
endif()

add_executable(luna-shougi src/main.cpp)
target_link_libraries(luna-shougi PRIVATE luna_engine)

# The self-play measurement tool. It reuses luna_engine as the arbiter so that
# the rules it judges games by are the same code the engine plays by.
add_library(luna_match
  src/match/arbiter.cpp
  src/match/elo.cpp
  src/match/game.cpp
  src/match/match.cpp
  src/match/process.cpp
  src/match/usi_client.cpp
)
target_link_libraries(luna_match PUBLIC luna_engine Threads::Threads)

add_executable(luna-match src/match/main.cpp)
target_link_libraries(luna-match PRIVATE luna_match)

# Training data generation. It self-plays in-process rather than through USI
# pipes, which is several times faster, and reuses the arbiter so the games it
# labels end for the same reasons a measured game would.
add_library(luna_datagen
  src/datagen/datagen.cpp
  src/datagen/sample.cpp
)
target_link_libraries(luna_datagen PUBLIC luna_engine luna_match Threads::Threads)

add_executable(luna-datagen src/datagen/main.cpp)
target_link_libraries(luna-datagen PRIVATE luna_datagen)

if(LUNA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
