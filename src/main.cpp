#include <iostream>

#include "engine/usi_engine.hpp"

int main() {
  luna::UsiEngine engine;
  engine.Run(std::cin, std::cout);
  return 0;
}
