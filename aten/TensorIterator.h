// Sample Code

struct TensorIterator::Builder {
  friend struct TensorIterator;

  Builder() : iter_(new TensorIterator()){};

  void add_output(const Tensor &output) {
    iter_->operands_.emplace_back(output);
    iter_->num_outputs_++;
  }

  void add_output(const Tensor &input, Device device, ScalarType dtype) {
    iter_->operands_.emplace_back(input, device, dtype);
    iter_->num_outputs_++;
  }

  void add_input(const Tensor &input) { iter_->operands_.emplace_back(input); }

  void add_input(const Tensor &input, Device device, ScalarType dtype) {
    iter_->operands_.emplace_back(input, device, dtype);
  }

  void dont_compute_common_dtype() { iter_->compute_common_dtype_ = false; }

  void dont_resize_outputs() { iter_->resize_outputs_ = false; }

  std::unique_ptr<TensorIterator> build();

protected:
  std::unique_ptr<TensorIterator> iter_;
};
