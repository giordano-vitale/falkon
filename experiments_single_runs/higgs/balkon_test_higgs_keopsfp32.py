import torch
import falkon
from falkon.benchmarks.common.datasets import HiggsDataset
import sklearn
import time

dtype = torch.float32

higgs = HiggsDataset()
Xtr, Ytr, Xts, Yts, _ = higgs.load_data(dtype=dtype, as_torch=True)
n = Xtr.shape[0]

##### BALKON #####

left_preconditioning = True
test_partial_iterations = True
keops_active = "yes"
debug = False
if dtype == torch.float32:
    memory_slack = 0.68
else:
    memory_slack = 0.9

# m0 = 100_000
m0 = 500

# max_gpu_mem = 104_857_600_044 # 100_000 MiB
# max_gpu_mem = 52_428_800_022  # 50_000 MiB, su 4 GPU ritorniamo alla stessa memoria delle due H200 di CIL
# options = falkon.options.FalkonOptions(keops_active="no", m0=m0, max_gpu_mem=max_gpu_mem, debug=True)  # default setting, keops does not work
options = falkon.options.FalkonOptions(keops_active=keops_active, m0=m0, test_partial_iterations=test_partial_iterations, memory_slack=memory_slack, debug=debug)  # default setting, keops does not work

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
# m = 800_000
# m = 120_000
m = 2_000

max_it = 20
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

ypred = blk.predict(Xts)
roc_auc = sklearn.metrics.roc_auc_score(Yts, ypred)
acc = sklearn.metrics.accuracy_score(Yts, ypred.sign())

print(f"ROC AUC: {roc_auc}")
print(f"Accuracy: {acc}")

# torch.save(blk.model_evolution_, "higgs_model_evolution_")

if test_partial_iterations:
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

    # torch.save(roc_auc_evolution, "higgs_roc_auc_evolution_")
    # torch.save(acc_evolution, "higgs_acc_evolution_")
