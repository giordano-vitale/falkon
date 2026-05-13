import torch
import falkon
from falkon.benchmarks.common.datasets import MillionSongsDataset
from falkon.benchmarks.common.error_metrics import ms_calc_relerr
import sklearn
import time

dtype = torch.float64

msd = MillionSongsDataset()
Xtr, Ytr, Xts, Yts, kwargs = msd.load_data(dtype=dtype, as_torch=True)
n = Xtr.shape[0]
Y_mean = kwargs["Y_mean"]
Y_std = kwargs["Y_std"]

##### BALKON #####

left_preconditioning = True
test_partial_iterations = True
keops_active = "no"
debug = False

m0 = 50_000

# max_gpu_mem = 104_857_600_044 # 100_000 MiB
# max_gpu_mem = 52_428_800_022  # 50_000 MiB, su 4 GPU ritorniamo alla stessa memoria delle due H200 di CIL
# options = falkon.options.FalkonOptions(keops_active="no", m0=m0, max_gpu_mem=max_gpu_mem, debug=True)  # default setting, keops does not work
options = falkon.options.FalkonOptions(keops_active=keops_active, m0=m0, test_partial_iterations=test_partial_iterations, debug=debug)  # default setting, keops does not work

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

s = 7.0
kernel = falkon.kernels.GaussianKernel(sigma=s, opt=options)

lam = 2e-6
m = 400_000

max_it = 10
# seed = 0

if left_preconditioning:
    blk = falkon.models.BalkonLeftPreconditioning(kernel=kernel,
                                                  penalty=lam,
                                                  n=n,
                                                  M=m,
                                                  maxiter=max_it,
                                                  # seed=seed,
                                                  options=options
                                                  )
else:
    blk = falkon.models.Balkon(kernel=kernel,
                               penalty=lam,
                               M=m,
                               maxiter=max_it,
                               # seed=seed,
                               options=options
                               )

start = time.time()
blk.fit(Xtr,Ytr)
end = time.time()

print(f"Precision: {dtype}")
print(f"max_it = {max_it}")
print(f"m = {m}")
print(f"m0 = {m0}")
print(f"Preconditioning: {'Left' if left_preconditioning else 'Bilateral'}")
print(f"Training time: {end - start} seconds")

start = time.time()
ypred = blk.predict(Xts)
end = time.time()
print(f"Inference time on the test set: {end - start} seconds")

mse = sklearn.metrics.mean_squared_error(Yts, ypred)
relerr, _ = ms_calc_relerr(Yts, ypred, Y_mean=Y_mean, Y_std=Y_std)

print(f"MSE: {mse}")
print(f"Relative Error: {relerr}")

# torch.save(blk.model_evolution_, "higgs_model_evolution_")

if test_partial_iterations:
    print("MSE and Relative Error evolution during the iterations")
    mse_evolution = torch.zeros(blk.model_evolution_.shape[1])
    relerr_evolution = torch.zeros(blk.model_evolution_.shape[1])

    for j in range(blk.model_evolution_.shape[1]):
        print(f"ITERATION {j}")
        blk.alpha_=blk.model_evolution_[:, j:j+1]

        ypred = blk.predict(Xts)
        mse = sklearn.metrics.mean_squared_error(Yts, ypred)
        relerr, _ = ms_calc_relerr(Yts, ypred, Y_mean=Y_mean, Y_std=Y_std)

        mse_evolution[j] = mse
        relerr_evolution[j] = relerr

        print(f"MSE: {mse}")
        print(f"Relative Error: {relerr}")

    # torch.save(roc_auc_evolution, "higgs_roc_auc_evolution_")
    # torch.save(acc_evolution, "higgs_acc_evolution_")
