// Sample code
`
namespace {
template <typename scalar_t>
void lerp_cpu(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& end,
    scalar_t weight_val) {
  auto builder = at::TensorIterator::Builder();
  builder.add_output(ret);
  builder.add_input(self);
  builder.add_input(end);
  auto iter = builder.build();
  at::native::binary_kernel(*iter, [=](scalar_t self_val, scalar_t end_val) {
    return (weight_val < 0.5)
        ? self_val + weight_val * (end_val - self_val)
        : end_val - (end_val - self_val) * (1 - weight_val);
  });
}
} // namespace
