// Sample Code
// This code is copied from PyTorch codebase for demostration

TensorIteratorBase::TensorIteratorBase() = default;

void TensorIteratorBase::build(TensorIteratorConfig& config) {
  // populate some persistent configuration fields
  is_reduction_ = config.is_reduction_;
  enforce_linear_iteration_ = config.enforce_linear_iteration_;

  // fill in operands_ based on configuration
  populate_operands(config);
  // set is_output and is_read_write flags on appropriate tensors
  mark_outputs();
  // Check that the outputs have no internal overlap
  // and do not share memory with inputs.
  compute_mem_overlaps(config);
  // Check that input dimensions are aligned correctly & compute outnames.
  compute_names(config);
  // compute the broadcasted shape
  compute_shape(config);
  // mark outputs for resizing if necessary
  mark_resize_outputs(config);
  // compute the result dtype and device
  compute_types(config);
  // try fast setup output tensor, if failed, fallback to normal setup
  if (!fast_set_up(config)) {
    // compute each tensor's stride after broadcasting
    compute_strides(config);
    // re-order dimensions to improve coalescing
    reorder_dimensions();
    // allocate the output tensor if it's not provided
    allocate_or_resize_outputs();
    // coalesce adjacent dimensions when possible
    if (!is_meta_) coalesce_dimensions();
  }

  if (is_meta_) return;

  // XLA and lazy tensors don't have storage, so they don't have an underlying data pointer.
  // Nothing beyond this point is important for meta functions, so it's fine to exit early here.
  // Extend the condition to ORT tesnors as ORT tensors also don't have storage.
  if (common_device_.type() == DeviceType::XLA  ||
      common_device_.type() == DeviceType::Lazy ||
      common_device_.type() == DeviceType::ORT) return;

  for (auto& op : operands_) {
    TORCH_INTERNAL_ASSERT(op.tensor_base().defined());
    op.data = op.tensor_base().data_ptr();
  }

  // zero out offsets
  // If the tensor is a scalar, we leave room for it
  // So index translations in reduction can access
  // a valid value for the offset
  int64_t ndim_offsets = (ndim() ? ndim() : 1);
  view_offsets_ = DimVector(ndim_offsets, 0);
}
