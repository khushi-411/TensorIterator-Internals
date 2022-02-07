#include <iostream>
#include <cassert>  // For error handling / debugging, in C it is used as <assert.h>
#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

void example1() {
  at::Tensor a = at::ones({10});
  at::Tensor b = at::ones({10});
  at::Tensor out = at::ones({0});
  std::cout
    << std::endl
    << "example1:"
    << std::endl;

  at::TensorIteratorConfig iter_config;
  iter_config
    .add_output(out)
    .add_input(a)
    .add_input(b);

  auto iter = iter_config.build();

  std::cout << "PASS" << std::endl;
}

int main() {
  at::manual_seed(0);
  example1();
}
