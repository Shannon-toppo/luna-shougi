include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(luna_tests
  test_arbiter.cpp
  test_elo.cpp
  test_eval.cpp
  test_movegen.cpp
  test_nnue.cpp
  test_perft.cpp
  test_position.cpp
  test_search.cpp
  test_timeman.cpp
  test_tt.cpp
  test_usi_engine.cpp
)
target_include_directories(luna_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(luna_tests PRIVATE luna_engine luna_match luna_datagen
                                        Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(luna_tests)
