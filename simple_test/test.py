import torch
import falkon
from falkon.benchmarks.common.datasets import HiggsDataset
import sklearn
import time

dtype = torch.float64

higgs = HiggsDataset()
Xtr, Ytr, Xts, Yts, _ = higgs.load_data(dtype=dtype, as_torch=True)

##### BALKON #####

# m0 = 50_000
m0 = 1_000
options = falkon.options.FalkonOptions(keops_active="no", m0=m0)  # default setting, keops does not work

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

s = 5.0
# s = 3.8
kernel = falkon.kernels.GaussianKernel(sigma=s, opt=options)

lam = 1e-8
# lam = 1e-7
# m = 200_000
m = 2_000
# m = 120_000

max_it = 10
seed = 0
blk = falkon.Balkon(kernel=kernel,
                    penalty=lam,
                    M=m,
                    maxiter=max_it,
                    seed=seed,
                    options=options
                    )

start = time.time()
blk.fit(Xtr,Ytr)
end = time.time()

print(f"Precision: {dtype}")
print(f"max_it = {max_it}")
print(f"m = {m}")
print(f"m0 = {m0}")
print(f"Training time: {end - start} seconds")

ypred = blk.predict(Xts)
roc_auc = sklearn.metrics.roc_auc_score(Yts, ypred)
acc = sklearn.metrics.accuracy_score(Yts, ypred.sign())

print(f"ROC AUC: {roc_auc}")
print(f"Accuracy: {acc}")