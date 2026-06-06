import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid.masknn.convolutional import Conv1DBlock  # Conv-TasNet style 1-D convolution block
from model.s4d import S4D, S4DStream  # S4D layer implementation
from model.adapt_layers import MulAddAdaptLayer  # Multiplicative adaptation (or FiLM) layer
from tools import get_speaker_embeddings_batch

# =====================
#  Encoder
# =====================
class Encoder(nn.Module):
    """
    1D Convolutional Encoder.

    This module maps the input waveform into a latent space.
    It applies a single Conv1d followed by a ReLU activation.

    Input shape: (batch, 1, time)
    Output shape: (batch, out_channels, time')
    """

    def __init__(self, kernel_size=320, stride=160, out_channels=4096):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,  # mono input
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        x = self.conv(x)  # Convolve to produce latent representation: (B, 4096, time')
        x = F.relu(x)  # Apply ReLU non-linearity
        return x


# ------------------------------------------------------------------------
# S4D Block
# ------------------------------------------------------------------------
class S4DBlock(nn.Module):
    """
    S4D Block as interpreted from Figure 1(b) of the paper.

    The block consists of:
      1) LayerNorm followed by S4D layer,
      2) A small feed-forward block: GELU activation followed by a 1x1 conv,
      3) A gated mechanism using GLU to merge the original input and the S4D output,
         with a skip connection: A = x + gated_output,
      4) A second feed-forward block (LN -> Linear -> GELU -> Linear),
         followed by a final residual addition: B = A + feed_forward_output.

    Note: The exact architecture is an interpretation. The "Linear" layers are implemented
          as 1x1 convolutions, which perform a per-time-frame linear transform on the channel dimension.
    """

    def __init__(self, d_model=256, d_state=32):
        super().__init__()

        # 1) Apply LayerNorm (channel-wise) before S4D
        self.ln_s4d = nn.LayerNorm(d_model)
        # S4D layer: models long-term dependencies using state-space modeling
        self.s4d = S4D(
            d_model=d_model,
            d_state=d_state,
            dropout=0.0,
            transposed=True  # works on (B, C, T)
        )
        self.gelu1 = nn.GELU()
        # 1x1 Conv to implement a linear transform (per time frame)
        self.linear1 = nn.Conv1d(d_model, d_model, kernel_size=1)

        # 2) GLU branch: merge the original input and the S4D branch output
        # Concatenate along the channel dimension and use a 1x1 conv to produce 2*d_model channels,
        # then apply GLU (which splits channels into two halves, one as gate)
        self.glu_conv = nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)  # Apply GLU along the channel dimension

        # 3) Second feed-forward block: LN -> Linear -> GELU -> Linear
        self.ln_ff2 = nn.LayerNorm(d_model)
        self.ff2_linear1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.ff2_gelu = nn.GELU()
        self.ff2_linear2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        """
        Input:
            x: Tensor of shape (batch, channels, time), where channels == d_model.
        Returns:
            B: Processed tensor with the same shape as x.
        """
        # --- Step 1: Apply LN -> S4D -> GELU -> Linear ---
        # Transpose to (B, time, channels) for LayerNorm
        y = x.transpose(1, 2)
        y = self.ln_s4d(y)  # Apply LayerNorm along channel dimension
        y = y.transpose(1, 2)  # Transpose back to (B, channels, time)

        # Pass through S4D layer (returns output and dummy state)
        y, _ = self.s4d(y)

        # Apply GELU and then a 1x1 conv (i.e., linear transformation)
        y = self.gelu1(y)
        y = self.linear1(y)

        # --- Step 2: Merge via GLU ---
        # Concatenate original input and processed output along channel dimension
        cat_xy = torch.cat([x, y], dim=1)  # (B, 2*d_model, time)
        z = self.glu_conv(cat_xy)  # (B, 2*d_model, time)
        z = self.glu(z)  # GLU reduces channel dimension back to d_model

        # Skip connection: add the gated output to the original input
        A = x + z

        # --- Step 3: Second feed-forward block ---
        # Apply another LayerNorm and 1x1 convs with GELU in between
        a_ = A.transpose(1, 2)  # (B, time, channels)
        a_ = self.ln_ff2(a_)  # LayerNorm along channels
        a_ = a_.transpose(1, 2)  # Back to (B, channels, time)
        a_ = self.ff2_linear1(a_)  # Linear transform
        a_ = self.ff2_gelu(a_)  # GELU activation
        a_ = self.ff2_linear2(a_)  # Linear transform

        # Final residual addition
        B = A + a_
        return B


class S4DBlockStream(nn.Module):
    """
    ストリーミング推論用 S4DBlock ラッパー.

    学習済み :class:`S4DBlock` を受け取り、全演算をタイムステップごとに実行します。
    S4D 以外の全演算（LayerNorm, GLU, 1x1 Conv）はポイントワイズのため
    状態管理は S4D の複素隠れ状態 ``h`` のみで完結します。

    Parameters
    ----------
    s4d_block : S4DBlock
        学習済み S4DBlock モジュール。
    """

    def __init__(self, s4d_block: S4DBlock):
        super().__init__()

        self.ln_s4d      = s4d_block.ln_s4d
        self.s4d_stream  = S4DStream(s4d_block.s4d)   # FFT → RNN モード変換
        self.gelu1       = s4d_block.gelu1
        self.linear1     = s4d_block.linear1           # Conv1d(H, H, kernel=1)

        self.glu_conv    = s4d_block.glu_conv          # Conv1d(2H, 2H, kernel=1)
        self.glu         = s4d_block.glu               # GLU(dim=1)

        self.ln_ff2      = s4d_block.ln_ff2
        self.ff2_linear1 = s4d_block.ff2_linear1       # Conv1d(H, H, kernel=1)
        self.ff2_gelu    = s4d_block.ff2_gelu
        self.ff2_linear2 = s4d_block.ff2_linear2       # Conv1d(H, H, kernel=1)

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        """S4D 隠れ状態のゼロ初期化. shape: (B, H, N/2, 2)"""
        return self.s4d_stream.initial_state(batch_size, device)

    def step(self, x_t: torch.Tensor, h: torch.Tensor):
        """
        1 タイムステップを処理する.

        Parameters
        ----------
        x_t : Tensor, shape (B, H)
            時刻 t の入力フィーチャ。
        h : Tensor, shape (B, H, N/2, 2)
            S4D の隠れ状態（実数ビュー）。

        Returns
        -------
        out : Tensor, shape (B, H)
        h_new : Tensor, shape (B, H, N/2, 2)
        """
        # --- Step 1: LN → S4D(RNN step) → GELU → 1x1 Conv ---
        y = self.ln_s4d(x_t)                                       # (B, H)
        y, h_new = self.s4d_stream.step(y, h)                      # (B, H)
        y = self.gelu1(y)                                           # (B, H)
        y = self.linear1(y.unsqueeze(-1)).squeeze(-1)               # (B, H)

        # --- Step 2: GLU merge ---
        cat_xy = torch.cat([x_t, y], dim=1)                        # (B, 2H)
        z = self.glu_conv(cat_xy.unsqueeze(-1)).squeeze(-1)        # (B, 2H)
        z = self.glu(z)                                            # (B, H)  [GLU(dim=1) on 2D]
        A = x_t + z                                                # Skip connection

        # --- Step 3: Feed-Forward (LN → Linear → GELU → Linear) ---
        a_ = self.ln_ff2(A)                                        # (B, H)
        a_ = self.ff2_linear1(a_.unsqueeze(-1)).squeeze(-1)        # (B, H)
        a_ = self.ff2_gelu(a_)                                     # (B, H)
        a_ = self.ff2_linear2(a_.unsqueeze(-1)).squeeze(-1)        # (B, H)
        out = A + a_                                               # Skip connection

        return out, h_new

    def forward(self, x: torch.Tensor, h=None):
        """
        チャンク単位のストリーミング推論.

        Parameters
        ----------
        x : Tensor, shape (B, H, L)
        h : Tensor, shape (B, H, N/2, 2) or None

        Returns
        -------
        y_out : Tensor, shape (B, H, L)
        h_new : Tensor, shape (B, H, N/2, 2)
        """
        B, H, L = x.shape
        if h is None:
            h = self.initial_state(B, device=x.device)

        outs = []
        for t in range(L):
            out_t, h = self.step(x[..., t], h)
            outs.append(out_t)

        return torch.stack(outs, dim=-1), h


# =====================
# Separator
# =====================
class Separator(nn.Module):
    """
    SpeakerBeam-SS Style Separator — per-block speaker conditioning.

    改善点:
      - スピーカー埋め込みを全 (Conv1DBlock + S4DBlock) ブロックで毎回適用
      - ブロック数を増加 (num_blocks1=4, num_blocks2=2)
      - スピーカー射影を MLP (Linear→ReLU→Linear) に強化
    """

    def __init__(self, channels=4096, num_blocks1=4, num_blocks2=2, emb_dim=192, dropout=0.1):
        super().__init__()
        out_channels = 256
        hidden_channels = 512

        self.layer_norm_in = nn.LayerNorm(channels)
        self.in_conv1x1 = nn.Conv1d(channels, out_channels, kernel_size=1)

        # FiLM 条件付け: 192 → 512 (スケール 256 + シフト 256)
        # 乗算のみ(v3)と異なり加算項があるため、干渉パターンを積極的にキャンセルできる
        self.spk_proj = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2 * out_channels),  # 512: scale 256 + shift 256
        )

        # 第1ステージ: Conv1DBlock + S4DBlock + FiLM 条件付け + Dropout
        self.blocks1 = nn.ModuleList()
        self.adapt1 = nn.ModuleList()
        self.drop1 = nn.ModuleList()
        for i in range(num_blocks1):
            self.blocks1.append(nn.ModuleList([
                Conv1DBlock(
                    in_chan=out_channels,
                    hid_chan=hidden_channels,
                    skip_out_chan=0,
                    kernel_size=3,
                    padding=(3 - 1) * (2 ** i),
                    dilation=2 ** i,
                    norm_type="gLN",
                    causal=True,
                ),
                S4DBlock(d_model=out_channels),
            ]))
            self.adapt1.append(
                MulAddAdaptLayer(indim=out_channels, enrolldim=2 * out_channels, ninputs=1, do_addition=True)
            )
            self.drop1.append(nn.Dropout(p=dropout))

        # 第2ステージ: Conv1DBlock + S4DBlock + FiLM 条件付け + Dropout
        self.blocks2 = nn.ModuleList()
        self.adapt2 = nn.ModuleList()
        self.drop2 = nn.ModuleList()
        for i in range(num_blocks2):
            self.blocks2.append(nn.ModuleList([
                Conv1DBlock(
                    in_chan=out_channels,
                    hid_chan=hidden_channels,
                    skip_out_chan=0,
                    kernel_size=3,
                    padding=(3 - 1) * (2 ** i),
                    dilation=2 ** i,
                    norm_type="gLN",
                    causal=True,
                ),
                S4DBlock(d_model=out_channels),
            ]))
            self.adapt2.append(
                MulAddAdaptLayer(indim=out_channels, enrolldim=2 * out_channels, ninputs=1, do_addition=True)
            )
            self.drop2.append(nn.Dropout(p=dropout))

        # Temporal Gate: フレームレベルで話者非存在区間を抑制する学習可能ゲート
        # 初期バイアス=+3.0 → sigmoid ≈ 0.95 (fine-tune開始時はほぼ素通り)
        # self.temporal_gate は v5 で試みたが全体的な抑制が強すぎたため除去

        self.out_conv1x1 = nn.Conv1d(out_channels, channels, kernel_size=1)
        self.layer_norm_out = nn.LayerNorm(channels)

    def forward(self, x, spk_embedding):
        input_orig = x

        x = x.transpose(1, 2)
        x = self.layer_norm_in(x)
        x = x.transpose(1, 2)
        x = self.in_conv1x1(x)  # (B, 256, T)

        # スピーカー埋め込みを射影 (B, 192) → (B, 256)
        spk = self.spk_proj(spk_embedding)

        # 第1ステージ: 各ブロック後にスピーカー条件付け + Dropout
        for (conv_block, s4d_block), adapt, drop in zip(self.blocks1, self.adapt1, self.drop1):
            x = conv_block(x)
            x = s4d_block(x)
            x = adapt(x, spk)  # ← 毎ブロックで適用
            x = drop(x)

        # 第2ステージ: 各ブロック後にスピーカー条件付け + Dropout
        for (conv_block, s4d_block), adapt, drop in zip(self.blocks2, self.adapt2, self.drop2):
            x = conv_block(x)
            x = s4d_block(x)
            x = adapt(x, spk)  # ← 毎ブロックで適用
            x = drop(x)

        x = self.out_conv1x1(x)
        x = x.transpose(1, 2)
        x = self.layer_norm_out(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = x * input_orig

        return x


# =====================
#  Decoder
# =====================
class Decoder(nn.Module):
    """
    Decoder module:

    Converts the separated latent representation back to a time-domain waveform.
    It uses a ConvTranspose1d layer to perform the reconstruction.

    Input shape: (batch, channels, time) where channels is the encoder output dimension (4096)
    Output shape: (batch, 1, reconstructed_time)
    """

    def __init__(self, in_channels=4096, kernel_size=320, stride=160):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        x = self.deconv(x)  # Reconstruct waveform: (B, 1, time')
        return x


# =====================
# SpeakerBeam-SS: Complete Model
# =====================
class SpeakerBeamSS(nn.Module):
    """
    SpeakerBeam-SS model.

    Overall architecture:
      Encoder -> Separator -> Decoder

    - The encoder converts the waveform to a latent representation.
    - The separator refines the latent representation using target speaker information
        (via speaker embeddings, multiplicative adaptation, and S4D blocks).
    - The decoder converts the refined latent representation back to a time-domain waveform.
    """

    def __init__(self, emb_dim=192):
        super().__init__()
        self.encoder = Encoder()  # Maps waveform to latent space (4096 channels)
        self.separator = Separator(emb_dim=emb_dim)  # Processes latent representation (projects to 256, processes, then projects back)
        self.decoder = Decoder()  # Reconstructs waveform from latent representation

    def forward(self, mixture, enrollment):
        """
        Args:
            mixture: Tensor of shape (batch, 1, time) -- the input mixed waveform.
            enrollment: Tensor of shape (batch, 1, time') -- target speaker's enrollment waveform.
                        (The speaker embedding extraction is assumed to be handled inside adapt layer or externally.)
        Returns:
            out_wav: Tensor of shape (batch, 1, reconstructed_time) -- the separated target speech waveform.
        """
        # Encode the input mixture into latent representation.
        enc_out = self.encoder(mixture)  # (B, 4096, time')
        # Process latent representation with the separator.
        sep_out = self.separator(enc_out, enrollment)  # (B, 4096, time')
        # Decode the latent representation back to a waveform.
        out_wav = self.decoder(sep_out)  # (B, 1, reconstructed_time)
        return out_wav


# =====================
# example
# =====================
if __name__ == "__main__":
    import torchaudio
    from tools import load_ecapa_model
    # 1つ目の音声をロード
    waveform1, sample_rate1 = torchaudio.load("../data/sample/20250306170609.wav")
    if sample_rate1 != 16000:
        waveform1 = torchaudio.transforms.Resample(orig_freq=sample_rate1, new_freq=16000)(waveform1)

    # 2つ目の音声をロード
    waveform2, sample_rate2 = torchaudio.load("../data/sample/20250306170609.wav")  # 別のファイル
    if sample_rate2 != 16000:
        waveform2 = torchaudio.transforms.Resample(orig_freq=sample_rate2, new_freq=16000)(waveform2)

    # バッチ化（異なる長さの場合は padding が必要）
    max_length = max(waveform1.shape[1], waveform2.shape[1])
    waveform1 = torch.nn.functional.pad(waveform1, (0, max_length - waveform1.shape[1]))
    waveform2 = torch.nn.functional.pad(waveform2, (0, max_length - waveform2.shape[1]))

    # バッチサイズ2の Tensor にする
    batch_waveform = torch.stack([waveform1, waveform2], dim=0)

    speaker_encoder = VoiceEncoder(device="cpu")
    speaker_embeddings = get_speaker_embeddings_batch(speaker_encoder, batch_waveform)

    batch_size = 2
    input_len = 16000  # 1秒分 @16kHz 32bit float
    mixture = torch.randn(batch_size, 1, input_len)

    model = SpeakerBeamSS()
    with torch.no_grad():
        out = model(mixture, speaker_embeddings)
    print("Input:", mixture.shape, "Output:", out.shape)
