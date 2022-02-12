#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <cassert> // For error handling / debugging, in C it is used as <assert.h>
#include <iostream>

// TensorIterator Configurations (immutable):
// Shape/Size, Dtype, memory-layout, strides, device

// Basic Configuration
void example1() {
  at::Tensor a = at::ones({10});
  at::Tensor b = at::ones({10});
  at::Tensor out = at::ones({0});
  std::cout << std::endl << "example1:" << std::endl;

  at::TensorIteratorConfig iter_config;
  iter_config.add_output(out).add_input(a).add_input(b);

  auto iter = iter_config.build();

  std::cout << "PASS" << std::endl;
}

// TensorIterator classification:
// 1. Point-wise: freely parallelized along any dimension and grain size
// 2. Reduction: have to be either parallelized along dimensions that you're not
// iterating 	over or by performing bisect and reduce operations along the
// dimension being iterated
//      FOR CUDA: possible to parallelize along the reduction dimension

// Parallelization in TensorIterator
void example2() {
  at::Tensor a = at::randn({10});
  at::Tensor out = at::randn({10});
  std::cout << std::endl << "example2:" << std::endl;

  at::TensorIteratorConfig iter_config;
  iter_config.add_output(out)
      .add_input(a)

      // call if output was already allocated
      .resize_outputs(false)

      // call if inputs/outputs have different types
      .check_all_same_dtype(false);

  auto iter = iter_config.build();

  // Copies data from input into output.
  // TODO: meaning of []
  auto copy_loop = [](char **data, const int64_t *strides, int64_t n) {
    auto *out_data = data[0];
    auto *in_data = data[1];

    // adding strides to reach the next element
    for (int64_t i = 0; i < n; i++) {
      // casting to floating dtype
      // TODO: What if it overlaps?
      *reinterpret_cast<float *>(out_data) =
          *reinterpret_cast<float *>(in_data);
      out_data += strides[0];
      in_data += strides[1];
    }
  };

  // TODO: (How?) Implicitly parallelizes the operation
  // 		If internal size is more than the decided right amount
  // 		of data to iterate
  // For serial implementation: Use `serial_for_each` loop
  iter.for_each(copy_loop);

  assert(at::allclose(a, out));
  std::cout << "PASS" << std::endl;
}

// Using Kernels
// No need to worry about stride, data type, or parallelism
void example3() {
  at::Tensor a = at::randn({10});
  at::Tensor b = at::randn({10});
  at::Tensor c = at::randn({10});
  std::cout << std::endl << "example3:" << std::endl;

  at::TensorIteratorConfig iter_config;
  iter_config.add_output(c).add_input(a).add_input(b);

  auto iter = iter_config.build();

  // Help in performing point-wise operations/broadcasting
  // Other macros: AT_DISPATCH_ALL_TYPES*
  // This loop is run until the output shape is equal to or greater than
  // input dimension.
  at::native::cpu_kernel(iter, [](float a, float b) -> float { return a + b; });

  assert(at::allclose(a + b, c));
  std::cout << "PASS" << std::endl;
}

// Setting tensor iteration dimensions
// TODO
void example4() {
  std::cout << std::endl << "example4:" << std::endl;

  at::Tensor self = at::randn({7, 5});
  int64_t dim = 1;

  at::Tensor result = at::empty_like(self);

  at::TensorIteratorConfig iter_config;
  auto iter = iter_config.check_all_same_dtype(false)
                  .resize_outputs(false)
                  .declare_static_shape(self.sizes(), /*squash_dim=*/dim)
                  .add_output(result)
                  .add_input(self)
                  .build();

  // size of dimension, to calculate the cumulative sum
  int64_t self_dim_size = at::native::ensure_nonempty_size(self, dim);

  // these strides indicates the number of contigious elements
  auto result_dim_stride = at::native::ensure_nonempty_stride(result, dim);
  auto self_dim_stride = at::native::ensure_nonempty_stride(self, dim);

  auto loop = [&](char **data, const int64_t *strides, int64_t n) {
    auto *result_data_bytes = data[0];
    const auto *self_data_bytes = data[1];

    for (int64_t vector_idx = 0; vector_idx < n; ++vector_idx) {
      // Calculate cumulative sum for each element of the vector
      auto cumulative_sum = (at::acc_type<float, false>)0;
      for (int64_t elem_idx = 0; elem_idx < self_dim_size; ++elem_idx) {
        const auto *self_data =
            reinterpret_cast<const float *>(self_data_bytes);
        auto *result_data = reinterpret_cast<float *>(result_data_bytes);
        cumulative_sum += self_data[elem_idx * self_dim_stride];
        result_data[elem_idx * result_dim_stride] = (float)cumulative_sum;
      }

      // Go to the next vector
      result_data_bytes += strides[0];
      self_data_bytes += strides[1];
    }
  };

  iter.for_each(loop);

  assert(at::allclose(result, self.cumsum(dim)));
  std::cout << "PASS" << std::endl;
}

// Helper function
// TODO: Checkout other helper functions
void example5() {
  std::cout << std::endl << "example5:" << std::endl;

  at::Tensor self = at::randn({10, 10, 10});
  int64_t dim = 1;
  bool keepdim = false;

  // Setting output size to (0)
  at::Tensor result = at::empty({0}, self.options());

  // make_reductions resizes size of one input and one output
  // Handles TensorIteratorConfig internally

  // TODO: (How?) converts 2d inputs to 1d
  auto iter = at::native::make_reduction("sum_reduce", result, self, dim,
                                         keepdim, self.scalar_type());

  // Sum reduce data from input into output
  auto sum_reduce_loop = [](char **data, const int64_t *strides, int64_t n) {
    auto *out_data = data[0];
    auto *in_data = data[1];

    assert(strides[0] == 0);

    *reinterpret_cast<float *>(out_data) = 0;

    for (int64_t i = 0; i < n; i++) {
      // assume float data type for this example
      *reinterpret_cast<float *>(out_data) +=
          *reinterpret_cast<float *>(in_data);
      in_data += strides[1];
    }
  };

  iter.for_each(sum_reduce_loop);

  assert(at::allclose(result, self.sum(dim, keepdim)));
  std::cout << "PASS" << std::endl;
}

int main() {
  at::manual_seed(0);
  example1();
  example2();
  example3();
  example4();
  example5();
}
