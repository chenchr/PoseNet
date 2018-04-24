TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
nvcc -c -o corr.cu.o corr_kernel.cu --gpu-architecture=compute_50 --gpu-code=compute_50 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC
