# fabular
[WIP] Framework transparent Flash Attention support。

Current state:

- [x] Add the basic fabular module, exported by nanobind. PyTorch & Paddle & NumPy are tested (simply): python end inputs the tensors/arrays from different framework, exporting the dlpack `PyCapsule`, while cpp utilize it (convert to dlpack `DLManagedTensor`) and print the basic meta info and underlying tensor data. Totally viable.
- [ ] More to be done...
