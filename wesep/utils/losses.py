import auraloss
import torch
import torch.nn as nn
import torchmetrics.audio as audio_metrics
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

# --- 1. 新增 PCMLoss 类实现 ---
class PCMLoss(nn.Module):
    """
    Phase Constrained Magnitude (PCM) Loss
    论文实现: L_PCM = 0.5 * L_SM(speech) + 0.5 * L_SM(noise)
    其中 noise = mixture - speech
    """
    def __init__(self, **kwargs):
        super().__init__()
        # 基础 SM Loss 使用 auraloss 的 STFTLoss
        # kwargs 可以透传 fft_size, hop_size 等参数给 STFTLoss
        self.sm_loss = auraloss.freq.STFTLoss(**kwargs)

    def forward(self, preds, target, mixture):
        """
        注意：调用此 Loss 时必须传入三个参数
        Args:
            preds:   预测的语音 (Batch, Channels, Time)
            target:  真实的纯净语音 (Batch, Channels, Time)
            mixture: 原始混合信号 (Batch, Channels, Time)
        """
        # 1. 计算语音部分的幅度损失
        loss_speech = self.sm_loss(preds, target)

        # 2. 计算隐式噪声信号 (Residual Noise)
        # 预测的噪声 = 混合 - 预测语音
        est_noise = mixture - preds
        # 真实的噪声 = 混合 - 真实语音
        clean_noise = mixture - target

        # 3. 计算噪声部分的幅度损失
        loss_noise = self.sm_loss(est_noise, clean_noise)

        # 4. 返回加权和 (通常各占 0.5)
        return 0.5 * loss_speech + 0.5 * loss_noise

# --- 原有字典定义 ---
valid_losses = {}

torch_losses = {
    "L1": nn.L1Loss(),
    "L2": nn.MSELoss(),
    "CE": nn.CrossEntropyLoss(),
}

torchmetrics_losses = {
    # Not tested
    "PIT":
    audio_metrics.PermutationInvariantTraining(
        scale_invariant_signal_noise_ratio),
}

auraloss_losses = {
    "STFT": auraloss.freq.STFTLoss(),
    "MultiResolutionSTFT": auraloss.freq.MultiResolutionSTFTLoss(),
    "SISDR": auraloss.time.SISDRLoss(),
    "SISNR": auraloss.time.SISDRLoss(),
    "SNR": auraloss.time.SNRLoss(),
}

# --- 2. 在这里注册 PCM Loss ---
custom_losses = {
    "PCM": PCMLoss(),
}

valid_losses.update(torch_losses)
valid_losses.update(auraloss_losses)
valid_losses.update(torchmetrics_losses)
valid_losses.update(custom_losses)  

def parse_loss(loss):
    loss_functions = []
    if not isinstance(loss, list):
        loss = [loss]
    for i in range(len(loss)):
        loss_name = loss[i]
        loss_functions.append(valid_losses.get(loss_name))
    return loss_functions
