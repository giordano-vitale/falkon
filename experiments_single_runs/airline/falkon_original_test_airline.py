import torch
import falkon
from falkon.benchmarks.common.datasets import FlightsDataset
import sklearn
import time

dtype = torch.float64

airline = FlightsDataset()
Xtr, Ytr, Xts, Yts, _ = airline.load_data(dtype=dtype, as_torch=True)

keops_active = "no"

##### FALKON #####

# max_gpu_mem = 104_857_600_044 # 100_000 MiB
# max_gpu_mem = 52_428_800_022  # 50_000 MiB, su 4 GPU ritorniamo alla stessa memoria delle due H200 di CIL
# options = falkon.options.FalkonOptions(keops_active="no", m0=m0, max_gpu_mem=max_gpu_mem, debug=True)  # default setting, keops does not work
options = falkon.options.FalkonOptions(keops_active=keops_active)

'''
options = falkon.options.FalkonOptions(min_cuda_pc_size_32=0,
                                       min_cuda_pc_size_64=0,
                                       min_cuda_iter_size_32=0,
                                       min_cuda_iter_size_64=0,
                                       num_fmm_streams=2,  # default = 2
                                       keops_active="no",
                                       cg_tolerance=0,  # force to do max_it iterations, also because they use || ||_2
                                       cg_full_gradient_every=999,  # same as our basic cg implementation
                                       chol_force_in_core=False  # force the Cholesky in GPU if True
                                       )
'''

s = 0.9
kernel = falkon.kernels.GaussianKernel(sigma=s, opt=options)

lam = 1e-8
m = 100_000

max_it = 20
# seed = 0

flk = falkon.models.Falkon(kernel=kernel,
                           penalty=lam,
                           M=m,
                           maxiter=max_it,
                           # seed=seed,
                           options=options
                           )

start = time.time()
flk.fit(Xtr,Ytr)
end = time.time()

print(f"Precision: {dtype}")
print(f"max_it = {max_it}")
print(f"m = {m}")
print(f"Training time: {end - start} seconds")

start = time.time()
ypred = flk.predict(Xts)
end = time.time()
print(f"Inference time on the test set: {end - start} seconds")

mse = sklearn.metrics.mean_squared_error(Yts, ypred)

print(f"MSE: {mse}")