"""
https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
Minimal version of S4D with extra options and features stripped out, for pedagogical purposes.
"""

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified


class S4DStream(nn.Module):
    """
    ストリーミング推論用 S4D ラッパー (RNN モード).

    学習済み S4D モジュールの FFT 畳み込みを以下の再帰式に変換します::

        h_t = Ā · h_{t-1} + B̄ · u_t   (複素対角 SSM)
        y_t = 2 · Re(C · h_t) + D · u_t

    ZOH 離散化パラメータ::

        Ā = exp(dt · A)
        B̄ = (exp(dt · A) − 1) / A      (B=1 を吸収した形)

    FFT 版と数学的に完全等価のため、学習済みパラメータをそのまま使用でき
    **性能劣化ゼロ** でリアルタイム（チャンクごと）推論が可能です。

    Parameters
    ----------
    s4d : S4D
        学習済み S4D モジュール。
    """

    def __init__(self, s4d: S4D):
        super().__init__()

        self.h = s4d.h          # d_model
        self.n = s4d.n          # d_state
        self.transposed = s4d.transposed

        kernel: S4DKernel = s4d.kernel

        with torch.no_grad():
            dt = torch.exp(kernel.log_dt)                                       # (H,)
            C  = torch.view_as_complex(kernel.C.data.clone().contiguous())      # (H, N/2) complex
            A  = -torch.exp(kernel.log_A_real) + 1j * kernel.A_imag            # (H, N/2) complex

            dtA   = A * dt.unsqueeze(-1)                                        # (H, N/2)
            A_bar = torch.exp(dtA)                                              # (H, N/2)
            B_bar = (torch.exp(dtA) - 1.0) / A                                 # (H, N/2)

        # 複素テンソルを実数ビュー (…, 2) でバッファ登録 → .to(device) に自動追従
        self.register_buffer("_A_bar", torch.view_as_real(A_bar.contiguous()))  # (H, N/2, 2)
        self.register_buffer("_B_bar", torch.view_as_real(B_bar.contiguous()))  # (H, N/2, 2)
        self.register_buffer("_C",     torch.view_as_real(C.contiguous()))      # (H, N/2, 2)
        self.register_buffer("_D",     s4d.D.data.clone())                      # (H,)

        # 活性化・出力変換は元モジュールから参照（重み共有）
        self.activation    = s4d.activation     # GELU
        self.output_linear = s4d.output_linear  # Conv1d(H, 2H, kernel=1) + GLU

    # ------------------------------------------------------------------
    # 状態管理
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        """
        ゼロ初期隠れ状態を返す.

        Returns
        -------
        h : Tensor, shape (B, H, N/2, 2)
            複素隠れ状態の実数ビュー（:func:`torch.view_as_real` 形式）。
        """
        dev = device if device is not None else self._A_bar.device
        return torch.zeros(batch_size, self.h, self.n // 2, 2, device=dev)

    # ------------------------------------------------------------------
    # 単一タイムステップ推論
    # ------------------------------------------------------------------

    def step(self, u: torch.Tensor, h: torch.Tensor):
        """
        1 タイムステップを処理する.

        Parameters
        ----------
        u : Tensor, shape (B, H)
            時刻 t の入力フィーチャ。
        h : Tensor, shape (B, H, N/2, 2)
            前ステップの隠れ状態（:meth:`initial_state` または本メソッドの戻り値）。

        Returns
        -------
        y : Tensor, shape (B, H)
            出力（activation・output_linear を適用済み）。
        h_new : Tensor, shape (B, H, N/2, 2)
            更新後の隠れ状態（実数ビュー）。
        """
        A_bar_c = torch.view_as_complex(self._A_bar)           # (H, N/2)
        B_bar_c = torch.view_as_complex(self._B_bar)           # (H, N/2)
        C_c     = torch.view_as_complex(self._C)                # (H, N/2)
        h_c     = torch.view_as_complex(h.contiguous())         # (B, H, N/2)

        # h_t = Ā · h_{t-1} + B̄ · u_t
        # u: (B, H) → unsqueeze(-1) で (B, H, 1) にブロードキャスト
        h_new_c = A_bar_c * h_c + B_bar_c * u.unsqueeze(-1)    # (B, H, N/2)

        # y_t = 2 · Re(C · h_t) + D · u_t
        y = 2.0 * (C_c * h_new_c).real.sum(-1) + self._D * u   # (B, H)

        # activation (GELU) → output_linear (pointwise Conv1d + GLU)
        y = self.activation(y)                                   # (B, H)
        y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)      # (B, H)

        return y, torch.view_as_real(h_new_c)

    # ------------------------------------------------------------------
    # チャンク推論（学習時の S4D.forward と同一シグネチャ）
    # ------------------------------------------------------------------

    def forward(self, u: torch.Tensor, h=None):
        """
        チャンク単位のストリーミング推論.

        学習時の :class:`S4D` の ``forward`` と同じシグネチャなので
        そのまま差し替えて使用できます。

        Parameters
        ----------
        u : Tensor, shape (B, H, L)  [``transposed=True`` 時]
              または  (B, L, H)       [``transposed=False`` 時]
        h : Tensor, shape (B, H, N/2, 2) or None
            引き継ぐ隠れ状態。``None`` の場合はゼロ初期化。

        Returns
        -------
        y_out : Tensor, same shape as u
        h_new : Tensor, shape (B, H, N/2, 2)
        """
        if not self.transposed:
            u = u.transpose(-1, -2)   # → (B, H, L)

        B, H, L = u.shape
        if h is None:
            h = self.initial_state(B, device=u.device)

        ys = []
        for t in range(L):
            y_t, h = self.step(u[..., t], h)   # (B, H)
            ys.append(y_t)

        y_out = torch.stack(ys, dim=-1)          # (B, H, L)

        if not self.transposed:
            y_out = y_out.transpose(-1, -2)

        return y_out, h