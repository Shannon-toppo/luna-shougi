include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(luna_tests
  test_movegen.cpp
  test_perft.cpp
  test_position.cpp
  test_usi_engine.cpp
)
target_link_libraries(luna_tests PRIVATE luna_engine Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(luna_tests)
