#include <torch/extension.h>

// Forward declarations
torch::Tensor flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal
);

torch::Tensor flash_attention_backward(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor L,
    bool causal
);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "FlashAttention forward (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("causal") = false);
    m.def("backward", &flash_attention_backward, "FlashAttention backward (CUDA)",
          py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"), 
          py::arg("O"), py::arg("L"), py::arg("causal") = false);
}
