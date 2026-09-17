# Ported from RoyChao19477/SEMamba (models/loss.py, train.py, SEMamba_advanced.yaml weights).
import numpy as np
import torch
import torch.nn.functional as F


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_losses(phase_r, phase_g, n_fft):
    dim_freq = n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (
        torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1)
        - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2)
        - torch.eye(dim_freq)
    ).to(phase_g.device)

    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (
        torch.triu(torch.ones(dim_time, dim_time), diagonal=1)
        - torch.triu(torch.ones(dim_time, dim_time), diagonal=2)
        - torch.eye(dim_time)
    ).to(phase_g.device)

    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))

    return ip_loss, gd_loss, iaf_loss


def semamba_generator_loss_terms(clean_mag, clean_pha, clean_com, clean_audio,
                                 mag_g, pha_g, com_g, audio_g, rec_com, n_fft):
    loss_mag = F.mse_loss(clean_mag, mag_g)

    loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, n_fft)
    loss_pha = loss_ip + loss_gd + loss_iaf

    loss_com = F.mse_loss(clean_com, com_g) * 2
    loss_time = F.l1_loss(clean_audio, audio_g)
    loss_con = F.mse_loss(com_g, rec_com) * 2

    return {
        "loss_mag": loss_mag,
        "loss_pha": loss_pha,
        "loss_com": loss_com,
        "loss_time": loss_time,
        "loss_con": loss_con,
    }
