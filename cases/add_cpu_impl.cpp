// Sample Code

namespace at {
namespace native {
namespace {
	
void add_kernel(at::Tensor& ret, const at::Tensor& a, const at::Tensor& b, Scalar alpha_scalar) {
  auto builder = at::TensorIterator::Builder();
    builder.add_output(ret);
    builder.add_input(a);
    builder.add_input(b);
    auto iter = builder.build();

  AT_DISPATCH_ALL_TYPES(iter.type(), "add", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    at::native::binary_kernel(iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; },
     );
  });
}

} // namespace anonymous

}} // namespace native, at
