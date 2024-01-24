
Install 

It is often easiest to install `jax` and CUDA via `conda`.
Installing locally can lead to challenges.

https://github.com/google/jax/discussions/6843

Note: On Ubuntu, use the 545 NVIDIA driver to work with CUDA 12.3. Install JAX first using `pip` and `cuda12_pip` option.
Then use `conda` to install CUDA.
``` 
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install -c nvidia cuda-nvcc
```
