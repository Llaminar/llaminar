# llaminar
An LLM inferencing engine in C++.

Llaminar tries to solve a variety of problems encountered in other projects:

* Tensor and Pipeline Parallelism: natively supported, mix and match heterogenous domains.
* Multiple vendors: Mix and match CPU, ROCm and CUDA, simultaneously and natively.
* Easy scaling: Built from the ground-up on OpenMPI so you can scale out across clusters. NUMA-aware by default. 
* An IaC-like experience: plan, then deploy.

## The Llaminar Philosophy

* Tensors want to be open and free: so is Llaminar.
* Tensors want to be sliced, sharded, and pipelined: Llaminar lets them be.
* Tensors want to run on a variety of hardware types without artificial handicaps: Llaminar helps them to do so.
