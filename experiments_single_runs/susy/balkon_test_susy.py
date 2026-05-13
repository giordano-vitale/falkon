import torch
import falkon
from falkon.benchmarks.common.datasets import SusyDataset
import sklearn
import time

dtype = torch.float64

susy = SusyDataset()
Xtr, Ytr, Xts, Yts, _ = susy.load_data(dtype=dtype, as_torch=True)

##### BALKON #####

m0 = 100_000  # here the m of FALKON was actually 30_000

max_gpu_mem = 104_857_600_044 # 100_000 MiB
options = falkon.options.FalkonOptions(keops_active="no", m0=m0, max_gpu_mem=max_gpu_mem)  # default setting, keops does not work

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

s = 3
kernel = falkon.kernels.GaussianKernel(sigma=s, opt=options)

lam = 1e-6
m = 800_000

max_it = 20
# seed = 0
blk = falkon.Balkon(kernel=kernel,
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
print(f"Training time: {end - start} seconds")

ypred = blk.predict(Xts)
roc_auc = sklearn.metrics.roc_auc_score(Yts, ypred)
acc = sklearn.metrics.accuracy_score(Yts, ypred.sign())

print(f"ROC AUC: {roc_auc}")
print(f"Accuracy: {acc}")

torch.save(blk.model_evolution_, "susy_model_evolution_")

print("ROC AUC and Accuracy evolution during the iterations")
roc_auc_evolution = torch.zeros(blk.model_evolution_.shape[1])
acc_evolution = torch.zeros(blk.model_evolution_.shape[1])

for j in range(blk.model_evolution_.shape[1]):
    print(f"ITERATION {j}")
    blk.alpha_=blk.model_evolution_[:, j:j+1]

    ypred = blk.predict(Xts)
    roc_auc = sklearn.metrics.roc_auc_score(Yts, ypred)
    acc = sklearn.metrics.accuracy_score(Yts, ypred.sign())

    roc_auc_evolution[j] = roc_auc
    acc_evolution[j] = acc

    print(f"ROC AUC: {roc_auc}")
    print(f"Accuracy: {acc}")

torch.save(roc_auc_evolution, "susy_roc_auc_evolution_")
torch.save(acc_evolution, "susy_acc_evolution_")