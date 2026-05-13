from typing import Optional, Union, Tuple

import torch

from falkon.la_helpers import copy_triang, mul_triang, trsm, vec_mul_triang
from falkon.options import FalkonOptions
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils import TicToc, decide_cuda
from falkon.utils.helpers import check_same_device
from falkon.utils.tensor_helpers import create_fortran, create_same_stride, is_f_contig

from .pc_utils import check_init, inplace_add_diag_th, inplace_set_diag_th, lauum_wrapper, potrf_wrapper
from .preconditioner import Preconditioner


class BalkonLeftPreconditioner(Preconditioner):

    def __init__(self, penalty: float, n: int, kernel, opt: FalkonOptions):
        super().__init__()
        self.params = opt
        self._use_cuda = decide_cuda(self.params) and not self.params.cpu_preconditioner

        self._lambda = penalty
        self._n = n
        self.kernel = kernel

        self.fC: Optional[torch.Tensor] = None
        self.dT: Optional[torch.Tensor] = None
        self.dA: Optional[torch.Tensor] = None

        self.X_nys: Optional[torch.Tensor] = None
        self.m0 = opt.m0

    def check_inputs(self, X: Union[torch.Tensor, SparseTensor], weight_vec: Optional[torch.Tensor] = None):
        if X.is_cuda and not self._use_cuda:
            raise RuntimeError("use_cuda is set to False, but data is CUDA tensor. Check your options.")
        if weight_vec is not None and not check_same_device(X, weight_vec):
            raise ValueError(f"Weights and data are not on the same device ({weight_vec.device}, {X.device})")
        if weight_vec is not None and weight_vec.shape[0] != X.shape[0]:
            raise ValueError(
                f"Weights and Nystrom centers should have the same first dimension. "
                f"Found instead {weight_vec.shape[0]}, {X.shape[0]}."
            )

    def init_kernel_mat(self, X) -> torch.Tensor:
        dtype = X.dtype
        dev = X.device
        M = X.size(0)
        with TicToc(f"Preconditioner kernel of size {M}. {X.shape=} {dev=}", debug=self.params.debug):
            if isinstance(X, torch.Tensor):
                C = create_same_stride((M, M), X, dtype=dtype, device=dev, pin_memory=self._use_cuda)
            else:  # If sparse tensor we need fortran for kernel calculation
                C = create_fortran((M, M), dtype=dtype, device=dev, pin_memory=self._use_cuda)
            self.kernel(X, X, out=C, opt=self.params)
        if not is_f_contig(C):
            C = C.T
        return C

    def init(self, X: Union[torch.Tensor, SparseTensor], weight_vec: Optional[torch.Tensor] = None):
        """Initialize the preconditioner matrix.

        This method must be called before the preconditioner can be used.

        Parameters
        ----------
        X : torch.Tensor
            The (M x D) matrix of Nystroem centers
        weight_vec
            An optional vector of size (M x 1) which is used for reweighted least-squares.
            This vector should contain the weights corresponding to the Nystrom centers.
        """
        """
        self.check_inputs(X, weight_vec)
        C = self.init_kernel_mat(X)
        M = C.shape[0]
        eps = self.params.pc_epsilon(C.dtype)

        with TicToc("Cholesky 1", debug=self.params.debug):
            # Compute T: lower(fC) = T.T
            inplace_add_diag_th(C, eps * M)
            C = potrf_wrapper(C, clean=False, upper=False, use_cuda=self._use_cuda, opt=self.params)
            # Save the diagonal which will be overwritten when computing A
            self.dT = C.diag()

        with TicToc("Copy triangular", debug=self.params.debug):
            # Copy lower(fC) to upper(fC):  upper(fC) = T.
            copy_triang(C, upper=False)

        # Weighted least-squares needs to weight the A matrix. We can weigh once before LAUUM,
        # but since CUDA-LAUUM touches both sides of C, weighting before LAUUM will also modify
        # the matrix T. Therefore for CUDA inputs we weigh twice after LAUUM!
        if weight_vec is not None and not self._use_cuda:
            with TicToc("Weighting(CPU)", debug=self.params.debug):
                weight_vec.sqrt_()
                vec_mul_triang(C, weight_vec, side=1, upper=False)

        if self._use_cuda:
            with TicToc("LAUUM(CUDA)", debug=self.params.debug):
                # Product upper(fC) @ upper(fC).T, store in lower(fC) = T @ T.T
                C = lauum_wrapper(C, upper=True, use_cuda=self._use_cuda, opt=self.params)
        else:
            with TicToc("LAUUM(CPU)", debug=self.params.debug):
                # Product lower(fC).T @ lower(fC), store in lower(fC) = T @ T.T
                C = lauum_wrapper(C, upper=False, use_cuda=self._use_cuda, opt=self.params)

        if weight_vec is not None and self._use_cuda:
            with TicToc("Weighting(CUDA)", debug=self.params.debug):
                weight_vec.sqrt_()
                vec_mul_triang(C, weight_vec, side=0, upper=False)
                vec_mul_triang(C, weight_vec, side=1, upper=False)

        with TicToc("Cholesky 2", debug=self.params.debug):
            # lower(fC) = 1/M * T@T.T
            mul_triang(C, upper=False, preserve_diag=False, multiplier=1 / M)
            # lower(fC) = 1/M * T@T.T + lambda * I
            inplace_add_diag_th(C, self._lambda)
            # Cholesky on lower(fC) : lower(fC) = A.T
            C = potrf_wrapper(C, clean=False, upper=False, use_cuda=self._use_cuda, opt=self.params)
            self.dA = C.diag()

        self.fC = C
        """
        self.X_nys = X



    def to(self, device):
        if self.fC is not None:
            self.fC = self.fC.to(device)
        if self.dT is not None:
            self.dT = self.dT.to(device)
        if self.dA is not None:
            self.dA = self.dA.to(device)
        return self

    @check_init("fC", "dT", "dA")
    def invA(self, v: torch.Tensor) -> torch.Tensor:
        r"""Solve the system of equations :math:`Ax = v` for unknown vector :math:`x`.

        Multiple right-hand sides are supported (by simply passing a 2D tensor for `v`)

        Parameters
        ----------
        v
            The right-hand side of the triangular system of equations

        Returns
        -------
        x
            The solution, computed with the `trsm` function.

        See Also
        --------
        :func:`~falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        inplace_set_diag_th(self.fC, self.dA)
        return trsm(v, self.fC, alpha=1.0, lower=1, transpose=1)

    @check_init("fC", "dT", "dA")
    def invAt(self, v: torch.Tensor) -> torch.Tensor:
        r"""Solve the system of equations :math:`A^\top x = v` for unknown vector :math:`x`.

        Multiple right-hand sides are supported (by simply passing a 2D tensor for `v`)

        Parameters
        ----------
        v
            The right-hand side of the triangular system of equations

        Returns
        -------
        x
            The solution, computed with the `trsm` function.

        See Also
        --------
        :func:`falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        inplace_set_diag_th(self.fC, self.dA)
        return trsm(v, self.fC, alpha=1.0, lower=1, transpose=0)

    @check_init("fC", "dT", "dA")
    def invT(self, v: torch.Tensor) -> torch.Tensor:
        r"""Solve the system of equations :math:`Tx = v` for unknown vector :math:`x`.

        Multiple right-hand sides are supported (by simply passing a 2D tensor for `v`)

        Parameters
        ----------
        v
            The right-hand side of the triangular system of equations

        Returns
        -------
        x
            The solution, computed with the `trsm` function.

        See Also
        --------
        :func:`falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        inplace_set_diag_th(self.fC, self.dT)
        return trsm(v, self.fC, alpha=1.0, lower=0, transpose=0)

    @check_init("fC", "dT", "dA")
    def invTt(self, v: torch.Tensor) -> torch.Tensor:
        r"""Solve the system of equations :math:`T^\top x = v` for unknown vector :math:`x`.

        Multiple right-hand sides are supported (by simply passing a 2D tensor for `v`)

        Parameters
        ----------
        v
            The right-hand side of the triangular system of equations

        Returns
        -------
        x
            The solution, computed with the `trsm` function.

        See Also
        --------
        :func:`falkon.preconditioner.pc_utils.trsm` : the function used to solve the system of equations
        """
        inplace_set_diag_th(self.fC, self.dT)
        return trsm(v, self.fC, alpha=1.0, lower=0, transpose=1)



    def _build_block(self, X_block: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        C = self.init_kernel_mat(X_block)

        with TicToc("Cholesky 1", debug=self.params.debug):
            # Compute T: lower(fC) = T.T
            inplace_add_diag_th(C, eps * self.m0)
            C = potrf_wrapper(C, clean=False, upper=False, use_cuda=self._use_cuda, opt=self.params)
            # Save the diagonal which will be overwritten when computing A
            dT = C.diag()

        with TicToc("Copy triangular", debug=self.params.debug):
            # Copy lower(fC) to upper(fC):  upper(fC) = T.
            copy_triang(C, upper=False)

        # Weighted least-squares needs to weight the A matrix. We can weigh once before LAUUM,
        # but since CUDA-LAUUM touches both sides of C, weighting before LAUUM will also modify
        # the matrix T. Therefore for CUDA inputs we weigh twice after LAUUM!
        # if weight_vec is not None and not self._use_cuda:
        #     with TicToc("Weighting(CPU)", debug=self.params.debug):
        #         weight_vec.sqrt_()
        #         vec_mul_triang(C, weight_vec, side=1, upper=False)

        if self._use_cuda:
            with TicToc("LAUUM(CUDA)", debug=self.params.debug):
                # Product upper(fC) @ upper(fC).T, store in lower(fC) = T @ T.T
                C = lauum_wrapper(C, upper=True, use_cuda=self._use_cuda, opt=self.params)
        else:
            with TicToc("LAUUM(CPU)", debug=self.params.debug):
                # Product lower(fC).T @ lower(fC), store in lower(fC) = T @ T.T
                C = lauum_wrapper(C, upper=False, use_cuda=self._use_cuda, opt=self.params)

        # if weight_vec is not None and self._use_cuda:
        #     with TicToc("Weighting(CUDA)", debug=self.params.debug):
        #         weight_vec.sqrt_()
        #         vec_mul_triang(C, weight_vec, side=0, upper=False)
        #         vec_mul_triang(C, weight_vec, side=1, upper=False)

        with TicToc("Cholesky 2", debug=self.params.debug):
            # lower(fC) = 1/M * T@T.T
            mul_triang(C, upper=False, preserve_diag=False, multiplier=1 / self.m0)
            # lower(fC) = 1/M * T@T.T + lambda * I
            inplace_add_diag_th(C, self._lambda)
            # Cholesky on lower(fC) : lower(fC) = A.T
            C = potrf_wrapper(C, clean=False, upper=False, use_cuda=self._use_cuda, opt=self.params)
            dA = C.diag()

        return C, dA, dT

    # @check_init("fC", "dT", "dA")
    def apply(self, v: torch.Tensor) -> torch.Tensor:
        r"""action of the whole preconditioner BB^T from the left
        """
        # return self.invAt(self.invTt(v))

        M = self.X_nys.shape[0]
        num_blocks = M // self.m0

        eps = self.params.pc_epsilon(self.X_nys.dtype)

        for i in range(num_blocks):
            i_start = i * self.m0
            i_end = (i + 1) * self.m0

            C, dA, dT = self._build_block(self.X_nys[i_start:i_end], eps)

            # action of (the block of) B^T
            inplace_set_diag_th(C, dT)
            v[i_start:i_end] = trsm(v[i_start:i_end], C, alpha=1.0, lower=0, transpose=1)

            inplace_set_diag_th(C, dA)
            v[i_start:i_end] = trsm(v[i_start:i_end], C, alpha=1.0, lower=1, transpose=0)

            # action of (the block of) B
            v[i_start:i_end] = trsm(v[i_start:i_end], C, alpha=1.0, lower=1, transpose=1)

            inplace_set_diag_th(C, dT)
            v[i_start:i_end] = trsm(v[i_start:i_end], C, alpha=1.0, lower=0, transpose=0)

        v = v.div_(self._n * num_blocks)

        return v
    
    def apply_t(self, v):
        pass  # Preconditioner has an abstract method called apply_t that must be implemented

    def __str__(self):
        return f"FalkonPreconditioner(_lambda={self._lambda}, kernel={self.kernel})"
