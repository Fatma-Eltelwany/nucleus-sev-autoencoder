import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.mixture import GaussianMixture
import cait as ai


def pulse_selection_with_autoencoder(dh, pulse_type, ch,
                                     linear_range=None,
                                     percent=None,
                                     downsample=1,
                                     epochs=50,
                                     use_gmm=True):
    """
    Trains a 1D convolutional autoencoder on waveforms using MSE loss only,
    restricted to events within specified linear range intervals.

    The linear_range argument accepts the output of find_linear_range()
    directly, so the two functions chain naturally:

        selected_ranges = find_linear_range(dh, ch=0,..)
        errs = pulse_selection_with_autoencoder(
                   dh, 'event', linear_range=selected_range, ch=0)

    Parameters
    ----------
    dh : cait DataHandler
    pulse_type : str
        'event' or 'noise'
    linear_range : list of (ph_lo, ph_hi) tuples or None
        Output of find_linear_range(). If None, all events are used.
    percent : float or None
        Percentile fallback if GMM fails (e.g. 80 keeps best 80%).
    ch : int
    downsample : int
    epochs : int
    use_gmm : bool

    Returns
    -------
    errs : np.ndarray (N_all,)
        Per-pulse reconstruction MSE for ALL events.
        Events outside the linear range are set to np.nan.
        Events inside the linear range have their actual AE error.
        This makes it easy to see which events were considered.
        
     Applies an "autoencoder_cut_{ch}" cut that is saved to the dh file
     to be used for SEV later.
    """

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    if pulse_type == 'event':
        pulses = dh.get('events', 'event',        ch)
        onsets = dh.get('events', 'onset',        ch).ravel()
        ph_all = dh.get('events', 'pulse_height', ch).ravel()
    elif pulse_type == 'noise':
        pulses = dh.get('noise', 'event',        ch)
        onsets = dh.get('noise', 'onset',        ch).ravel()
        ph_all = dh.get('noise', 'pulse_height', ch).ravel()
    else:
        raise ValueError("pulse_type must be 'event' or 'noise'")

    N_all, L0 = pulses.shape
    L         = L0 // downsample

    # ------------------------------------------------------------------
    # 2. Linear range mask
    # ------------------------------------------------------------------
    if linear_range is not None and len(linear_range) > 0:
        lr_mask = np.zeros(N_all, dtype=bool)
        for lo, hi in linear_range:
            lr_mask |= (ph_all >= lo) & (ph_all <= hi)
        print(f"Linear range mask: {lr_mask.sum()} / {N_all} events "
              f"({100 * lr_mask.sum() / N_all:.1f}%)")
        print(f"  Ranges: {[(f'{lo:.4f}', f'{hi:.4f}') for lo, hi in linear_range]}")
    else:
        lr_mask = np.ones(N_all, dtype=bool)
        print("No linear range provided — using all events.")

    # work only on linear-range events from here
    pulses_lr = pulses[lr_mask]
    onsets_lr = onsets[lr_mask]
    N_lr      = pulses_lr.shape[0]

    # ------------------------------------------------------------------
    # 3. Dynamic pre-trigger region
    # ------------------------------------------------------------------
    min_onset = np.percentile(onsets_lr, 1)
    max_onset = np.percentile(onsets_lr, 99)
    pre       = int(np.abs(min_onset)) + 20
    pre       = max(pre, 20)

    print(f"Onset range (1st-99th pct): [{min_onset:.1f}, {max_onset:.1f}]")
    print(f"Pre-trigger region: {pre} samples  |  "
          f"Waveform length: {L} samples  |  "
          f"Pulse region: {L - pre} samples")

    # ------------------------------------------------------------------
    # 4. Preprocessing: align + baseline subtract
    # ------------------------------------------------------------------
    def preprocess(wf, onset, ds):
        s       = int(round(onset))
        shifted = np.roll(wf, pre - s)
        shifted = shifted - np.mean(shifted[:pre])
        if ds > 1:
            shifted = shifted[::ds]
        return shifted

    proc = np.vstack([
        preprocess(pulses_lr[i], onsets_lr[i], downsample)
        for i in range(N_lr)
    ]).astype(np.float32)   # (N_lr, L)

    # ------------------------------------------------------------------
    # 5. DataLoader
    # ------------------------------------------------------------------
    loader = DataLoader(
        TensorDataset(torch.from_numpy(proc), torch.from_numpy(proc)),
        batch_size=64, shuffle=True
    )

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    class ConvAE(nn.Module):
        def __init__(self, L):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.ReLU(),
                nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.ReLU(),
                nn.Conv1d(32, 64, 9, stride=2, padding=4), nn.ReLU(),
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(64, 32, 9, stride=2, padding=4,
                                   output_padding=1), nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 9, stride=2, padding=4,
                                   output_padding=1), nn.ReLU(),
                nn.ConvTranspose1d(16,  1, 9, stride=2, padding=4,
                                   output_padding=1),
            )
        def forward(self, x):
            return self.dec(self.enc(x))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ConvAE(L).to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # 7. Training — MSE only
    # ------------------------------------------------------------------
    print(f"\nTraining on {N_lr} events  |  device: {device}")
    prev_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for batch, _ in loader:
            batch = batch.unsqueeze(1).to(device)   # (B, 1, L)
            recon = model(batch)
            loss  = mse_fn(recon, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * batch.size(0)

        epoch_loss = running / N_lr
        print(f"  Epoch {epoch+1:02d}/{epochs} | loss = {epoch_loss:.4e}")

        if epoch > 5 and abs(prev_loss - epoch_loss) < 1e-6:
            print(f"  -> Plateau at epoch {epoch+1}, stopping early.")
            break
        prev_loss = epoch_loss

    # ------------------------------------------------------------------
    # 8. Reconstruction errors on linear-range events
    # ------------------------------------------------------------------
    model.eval()
    errs_lr = []

    with torch.no_grad():
        for i in range(0, N_lr, 512):
            batch = torch.from_numpy(proc[i:i+512]).unsqueeze(1).to(device)
            recon = model(batch).cpu().numpy().squeeze(1)
            errs_lr.append(((recon - proc[i:i+512]) ** 2).mean(axis=1))

    errs_lr = np.concatenate(errs_lr)   # (N_lr,)

    # ------------------------------------------------------------------
    # 9. Map errors back to full event space
    #    Events outside linear range get nan — not considered for the cut
    # ------------------------------------------------------------------
    errs_all = np.full(N_all, np.nan)
    errs_all[lr_mask] = errs_lr

    # ------------------------------------------------------------------
    # 10. Threshold on linear-range events only
    # ------------------------------------------------------------------
    good_idx_lr   = _gmm_or_percentile_threshold(errs_lr, percent, use_gmm)

    # full mask: must be in linear range AND pass AE cut
    good_idx_full = np.zeros(N_all, dtype=bool)
    good_idx_full[lr_mask] = good_idx_lr

    n_pass = good_idx_full.sum()
    print(f"\nFinal selection: {n_pass} / {N_all} events pass "
          f"({100 * n_pass / N_all:.1f}%)")

    # ------------------------------------------------------------------
    # 11. Save cut to dh
    # ------------------------------------------------------------------
    cut_type  = 'events' if pulse_type == 'event' else 'noise'
    cut_label = (f'autoencoder_cut_{ch}'
                 if pulse_type == 'event'
                 else f'autoencoder_noise_cut_{ch}')

    cut = ai.cuts.LogicalCut(good_idx_full)
    dh.apply_logical_cut(
        cut_flag=cut.get_flag(),
        naming=cut_label,
        channel=ch,
        type=cut_type,
        delete_old=True,
    )

    dh.content()
    return errs_all


# ------------------------------------------------------------------
# GMM / percentile threshold helper
# ------------------------------------------------------------------
def _gmm_or_percentile_threshold(errs, percent, use_gmm):
    N = len(errs)

    if use_gmm:
        try:
            log_errs = np.log(errs + 1e-12).reshape(-1, 1)
            gmm      = GaussianMixture(n_components=2, random_state=42,
                                       max_iter=200).fit(log_errs)
            means    = gmm.means_.ravel()
            stds     = np.sqrt(gmm.covariances_.ravel())
            labels   = gmm.predict(log_errs)

            clean_comp    = int(np.argmin(means))
            noisy_comp    = 1 - clean_comp
            sep           = abs(means[clean_comp] - means[noisy_comp])
            avg_std       = (stds[clean_comp] + stds[noisy_comp]) / 2
            overlap_ratio = avg_std / (sep + 1e-8)

            if overlap_ratio < 1.0:
                good_idx = (labels == clean_comp)
                print(f"\nGMM threshold:"
                      f"\n  clean: mean={means[clean_comp]:.3f}  "
                      f"std={stds[clean_comp]:.3f}  n={good_idx.sum()}"
                      f"\n  noisy: mean={means[noisy_comp]:.3f}  "
                      f"std={stds[noisy_comp]:.3f}  n={N - good_idx.sum()}"
                      f"\n  separation ratio = {1/overlap_ratio:.2f}")
                return good_idx
            else:
                print(f"GMM overlap too high ({overlap_ratio:.2f}), "
                      f"falling back to percentile.")
        except Exception as ex:
            print(f"GMM failed ({ex}), falling back to percentile.")

    if percent is None:
        raise ValueError(
            "GMM failed/disabled and no percent fallback given. "
            "Pass percent=<value>."
        )
    threshold = np.percentile(errs, percent)
    good_idx  = errs < threshold
    print(f"\nPercentile threshold: {percent}th = {threshold:.4e}"
          f"  ->  {good_idx.sum()} / {N} selected")
    return good_idx