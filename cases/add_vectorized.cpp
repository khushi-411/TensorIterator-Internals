// Sample Code

namespace at {
namespace native {
namespace {

using namespace vec256;

void add_kernel(TensorIterator& iter, Scalar alpha_scalar) {
  AT_DISPATCH_ALL_TYPES(iter.type(), "add", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    auto alpha_vec = Vec256<scalar_t>(alpha);
    binary_kernel_vec(iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
        return vec256::fmadd(b, alpha_vec, a);
      });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(add_stub, &add_kernel);

}} // namespace at::native
