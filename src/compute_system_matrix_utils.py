from __future__ import annotations
import numpy as np
from .load_data_utils import ConvexSystemParam
from typing import Tuple
_TWO_PI = 2.0*np.pi

def _clamp_unit(x: float, eps: float = 1e-12) -> float:
    """Clamp to [-1, 1] with a tiny tolerance for floating error."""
    if x > 1 + eps or x < -1 - eps:
        return np.nan
    return min(1.0, max(-1.0, x))

def _union_of_intervals(intervals: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    arr = np.asarray(intervals, dtype=float)
    if arr.size == 0:
        return np.array([[0.0, 0.0]], dtype=float)

    good = np.isfinite(arr).all(axis=1) & (arr[:, 0] <= arr[:, 1] + tol)
    arr = arr[good]
    if arr.size == 0:
        return np.array([[0.0, 0.0]], dtype=float)

    # Sort by starting value of each interval
    arr = arr[np.argsort(arr[:, 0])]
    merged = []
    cur_s, cur_e = arr[0]
    for s, e in arr[1:]:
        if s <= cur_e + tol:
            cur_e = max(cur_e, e)
        else:
            merged.append([cur_s, cur_e])
            cur_s, cur_e = s, e
    merged.append([cur_s, cur_e])
    out = np.asarray(merged, dtype=float)
    return out if len(out) > 0 else np.array([[0.0, 0.0]], dtype=float)

def _thetax(x1: float, x2: float, rho: float, eps: float = 1e-12) -> np.ndarray:
    """
    Returns intervals of θ ∈ [0, 2π] as an (M, 2) array where
    Return either [a, b] or the union of two rows.
    When no valid interval exists, returns [[0, 0]]

    Parameters
    ----------
    x1, x2 : float
        Inputs (will be scaled by rho internally).
    rho : float
        Positive scaling factor.
    """
    # Normalize
    if rho == 0:
        return np.array([[0.0, 0.0]], dtype=float)
    x1n = x1 / rho
    x2n = x2 / rho

    def A(x):
        xc = _clamp_unit(x, eps)
        return np.arccos(xc) if np.isfinite(xc) else np.nan

    # Case 1: x1 > 1 && x2 < -1  → empty
    if (x1n > 1 + eps) and (x2n < -1 - eps):
        return np.array([[0.0, 0.0]], dtype=float)

    # Case 2: -1 <= x1 && x2 <= 1
    if (x1n >= -1 - eps) and (x2n <= 1 + eps):
        a1, b1 = A(x2n), A(x1n)
        a2, b2 = _TWO_PI - A(x1n), _TWO_PI - A(x2n)
        intervals = np.array([[a1, b1], [a2, b2]], dtype=float)
        return _union_of_intervals(intervals)

    # Case 3: x1 < -1 && -1 <= x2 <= 1
    if (x1n < -1 - eps) and (-1 - eps <= x2n <= 1 + eps):
        a = A(x2n)
        intervals = np.array([[a, _TWO_PI - a]], dtype=float)
        return intervals

    # Case 4: -1 <= x1 <= 1 && x2 > 1
    if (-1 - eps <= x1n <= 1 + eps) and (x2n > 1 + eps):
        a = A(x1n)
        intervals = np.array([[0.0, a], [_TWO_PI - a, _TWO_PI]], dtype=float)
        return _union_of_intervals(intervals)

    # Case 5: x1 <= -1 && x2 >= 1  → full [0, 2π]
    if (x1n <= -1 + eps) and (x2n >= 1 - eps):
        return np.array([[0.0, _TWO_PI]], dtype=float)

    # Otherwise return empty
    return np.array([[0.0, 0.0]], dtype=float)

def _thetay(y1: float, y2: float, rho: float) -> np.ndarray:
    """
    Returns a union of θ-intervals (in [0, 2π]) as an (M, 2) array.
    When no valid interval exists, returns [[0, 0]].
    """
    if rho == 0:
        return np.array([[0.0, 0.0]], dtype=float)

    y1n = y1 / rho
    y2n = y2 / rho
    eps = 1e-12

    def S(y):
        yc = _clamp_unit(y, eps)
        return np.arcsin(yc) if np.isfinite(yc) else np.nan

    # Case 1
    if (y1n > 1 + eps) and (y2n < -1 - eps):
        return np.array([[0.0, 0.0]], dtype=float)

    # Case 2: y1 >= 0 && y2 <= 1
    if (y1n >= 0 - eps) and (y2n <= 1 + eps):
        a = S(y1n)
        b = S(y2n)
        intervals = np.array([[a, b],
                              [np.pi - b, np.pi - a]], dtype=float)
        return _union_of_intervals(intervals)

    # Case 3: y1 >= -1 && y2 <= 0
    if (y1n >= -1 - eps) and (y2n <= 0 + eps):
        a = S(y1n)
        b = S(y2n)
        intervals = np.array([[np.pi - b, np.pi - a],
                              [_TWO_PI + a, _TWO_PI + b]], dtype=float)
        return _union_of_intervals(intervals)

    # Case 4: y1 <= 0 && y1 >= -1 && y2 >= 0 && y2 <= 1
    if (y1n <= 0 + eps) and (y1n >= -1 - eps) and (y2n >= 0 - eps) and (y2n <= 1 + eps):
        a1 = S(y2n)
        a2 = S(y1n)
        intervals = np.array([[0.0, a1],
                              [np.pi - a1, np.pi - a2],
                              [_TWO_PI + a2, _TWO_PI]], dtype=float)
        return _union_of_intervals(intervals)

    # Case 5: y1 < -1 && y2 in [0, 1]
    if (y1n < -1 - eps) and (0 - eps <= y2n <= 1 + eps):
        b = S(y2n)
        intervals = np.array([[0.0, b],
                              [np.pi - b, _TWO_PI]], dtype=float)
        return _union_of_intervals(intervals)

    # Case 6: y1 < -1 && y2 in [-1, 0)
    if (y1n < -1 - eps) and (-1 - eps <= y2n < 0 - eps):
        b = S(y2n)
        return np.array([[np.pi - b, _TWO_PI + b]], dtype=float)

    # Case 7: y1 in [0, 1] && y2 > 1
    if (0 - eps <= y1n <= 1 + eps) and (y2n > 1 + eps):
        a = S(y1n)
        return np.array([[a, np.pi - a]], dtype=float)

    # Case 8: y1 in [-1, 0) && y2 > 1
    if (-1 - eps <= y1n < 0 - eps) and (y2n > 1 + eps):
        a = S(y1n)
        intervals = np.array([[0.0, np.pi - a],
                              [_TWO_PI + a, _TWO_PI]], dtype=float)
        return _union_of_intervals(intervals)

    # Case 9: y1 <= -1 && y2 >= 1  → [0, 2π]
    if (y1n <= -1 + eps) and (y2n >= 1 - eps):
        return np.array([[0.0, _TWO_PI]], dtype=float)

    # Else: empty
    return np.array([[0.0, 0.0]], dtype=float)

def intersection_of_intervals(intervals: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Intersect two intervals [a1, a2] and [b1, b2].
    Parameters
    ----------
    intervals : (2, 2) array-like
    Returns
    -------
    (1, 2) ndarray if they overlap, else empty (0, 2) ndarray.
    """
    arr = np.asarray(intervals, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError("intervals must be shape (2, 2)")
    s = max(arr[0, 0], arr[1, 0])
    e = min(arr[0, 1], arr[1, 1])
    if s <= e + tol:
        return np.array([[s, e]], dtype=float)
    return np.empty((0, 2), dtype=float)

def _intersection_of_2_disjoint_sets(A: np.ndarray, B: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    MATLAB: intersection_of_2_disjoint_sets(A, B)
    Returns all pairwise intersections of rows from A and B.
    If no intersections exist, returns empty (0, 2) array.

    Parameters
    ----------
    A, B : (Na, 2), (Nb, 2) arrays of [start, end] intervals.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.size == 0 or B.size == 0:
        return np.empty((0, 2), dtype=float)

    # Broadcast pairwise starts/ends
    # starts shape: (Na, Nb), ends shape: (Na, Nb)
    starts = np.maximum(A[:, 0][:, None], B[:, 0][None, :])
    ends   = np.minimum(A[:, 1][:, None], B[:, 1][None, :])

    # Overlap mask (inclusive)
    mask = (starts <= ends + tol)

    if not np.any(mask):
        return np.empty((0, 2), dtype=float)

    # Gather overlapping pairs
    inters = np.stack([starts[mask], ends[mask]], axis=1)

    # Optional: normalize tiny negative widths to zero
    inters[:, 0] = np.minimum(inters[:, 0], inters[:, 1])

    return inters

def _build_transducer_impulse(Nt: int, info: ConvexSystemParam):
    """
    Build Bt (time-domain transducer impulse) and its 'derivative' Bt_deriv,
    """
    freq = np.linspace(-info.fs/2.0, info.fs/2.0, Nt, dtype=np.float64)
    freq_diff = 1j * freq
    mag = np.abs(freq)

    # Cosine-sum window around 0 with width set by fc_signal
    Bf = 0.42 - 0.5*np.cos(np.pi*mag / info.fc_signal) + 0.08*np.cos(2*np.pi*mag / info.fc_signal)
    # Band-limit
    bw_factor = 2.0
    Bf[(mag > bw_factor*info.fc_signal)] = 0.0
    s = Bf.sum()
    if s != 0:
        Bf = Bf / s

    # Derivative in freq-domain
    Bf_deriv = Bf * freq_diff
    Bf_deriv[(mag > bw_factor*info.fc_signal)] = 0.0

    # IFFT with fftshift/ifftshift symmetry + 1e3 scale like MATLAB
    Bt       = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifftshift(Bf))))       / 1e3
    Bt_deriv = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifftshift(Bf_deriv)))) / 1e3
    return Bt.astype(np.float64), Bt_deriv.astype(np.float64)

def EIR(t: np.ndarray,
        theta_d_deg: np.ndarray,
        rs: np.ndarray,
        info: ConvexSystemParam) -> np.ndarray:
    """
    Parameters
    ----------
    t : (Nt,) array
        Sample times [s].
    theta_d_deg : (Nd,) array
        Detector steering/element angles in DEGREES
    rs : (2,) array-like
        Source coordinates [z_s, x_s] in meters.
    info : probe specifications
        Geometry/acoustics parameters.

    Returns
    -------
    y : (Nt, Nd) array
        System impulse response per detector angle, normalized per-column.
    """
    t = np.asarray(t, dtype=np.float64)
    theta_d_deg = np.atleast_1d(theta_d_deg).astype(np.float64)
    rs = np.asarray(rs, dtype=np.float64)
    Nt = t.size
    Nd = theta_d_deg.size

    y = np.zeros((Nt, Nd), dtype=np.float64)

    # Element angular aperture (radians)
    ele_angle_range = (info.ele_width / info.pitch) * (info.d_theta * np.pi/180.0)

    # Precompute transducer impulse & its 'derivative'
    Bt, Bt_deriv = _build_transducer_impulse(Nt, info)

    # Precompute t * conv(...) factor used later
    t_vec = t.copy()

    # Loop over detector angles
    for j in range(Nd):
        theta = theta_d_deg[j] * np.pi / 180.0   # MATLAB’s code uses degrees → radians here
        # Element lateral/axial spans at this steering
        xd_range = info.ROC * np.array([np.sin(theta - ele_angle_range/2.0),
                                        np.sin(theta + ele_angle_range/2.0)], dtype=np.float64)
        yd_range = np.array([-info.ele_height/2.0, info.ele_height/2.0], dtype=np.float64)
        zd_range = info.ROC * np.array([np.cos(theta - ele_angle_range/2.0),
                                        np.cos(theta + ele_angle_range/2.0)], dtype=np.float64)

        # Time-of-flight window for nonzero support
        tmin = np.sqrt(np.min((rs[0] - zd_range)**2) + np.min((rs[1] - xd_range)**2)) / info.c
        tmax = np.sqrt(np.max((rs[0] - zd_range)**2) + np.max((rs[1] - xd_range)**2) + yd_range[0]**2) / info.c

        # Find indices in Python (0-based)
        # idx_t_start0: first index where t > tmin
        gt = np.nonzero(t > tmin)[0]
        if gt.size == 0:
            continue
        idx_t_start0 = int(gt[0])

        # idx_t_end0: last index where t < tmax
        lt = np.nonzero(t < tmax)[0]
        if lt.size == 0:
            continue
        idx_t_end0 = int(lt[-1])

        if idx_t_end0 < idx_t_start0:
            continue

        # Center axial position for rho
        zd_center = info.ROC * np.cos(theta)

        # Build h only on active support [start:end] to reduce work
        seg_len = idx_t_end0 - idx_t_start0 + 1
        h_seg = np.zeros(seg_len, dtype=np.float64)

        x1 = xd_range[0] - rs[1]
        x2 = xd_range[1] - rs[1]

        # Compute theta measure per time sample
        for k, it in enumerate(range(idx_t_start0, idx_t_end0 + 1)):
            tau = t[it]
            rho_sq = (info.c * tau)**2 - (rs[0] - zd_center)**2
            if rho_sq <= 0:
                # No lateral reach at this time; keep zero
                continue
            rho = np.sqrt(rho_sq)
            theta_x = _thetax(x1, x2, rho)
            theta_y = _thetay(yd_range[0], yd_range[1], rho)
            theta_ = _intersection_of_2_disjoint_sets(theta_x, theta_y)

            if theta_.size > 0:
                theta_sum = np.sum(theta_[:, 1] - theta_[:, 0])
            else:
                theta_sum = 0.0

            h_seg[k] = theta_sum * info.c / _TWO_PI / 1e3

        # Convolution on the active window, with 'same' length like MATLAB
        # Then circular shift
        sir      = np.real(np.convolve(Bt, h_seg, mode='same'))
        sir_der  = t_vec[idx_t_start0:idx_t_end0+1] * np.real(np.convolve(Bt_deriv, h_seg, mode='same'))
        sir_sum  = sir - sir_der
        L_shift = int(np.round((idx_t_start0 - Nt/2.0) + (idx_t_end0 - idx_t_start0)*2.0))

        # Place the shifted segment back into full-length output column
        column = np.zeros(Nt, dtype=np.float64)
        shifted  = np.roll(sir_sum, L_shift)
        column[idx_t_start0:idx_t_end0+1] = shifted

        # Normalize per-column
        m = np.max(np.abs(column))
        if m > 0:
            y[:, j] = column / m
        else:
            y[:, j] = column

    return y

def gen_imaging_matrix(idx_t_end_recon: int,
                       idx_t_start: int,
                       width_radius: float,
                       info: ConvexSystemParam) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    """
    Parameters
    ----------
    idx_t_end_recon : int
        Upper bound index into info.d_sample defining the reconstruction depth (0-based in Python).
    idx_t_start : int
        Lower bound time index (0-based) for the response segment.
    width_radius : float
        Half-width of reconstruction FOV in x (meters).
    info : ProbeInfo
        Probe/system parameters (fs, fc, fc_signal, c, ROC, pixel_d, Nfocus, EleAngle, N_ele, ele_width, ele_height,
        pitch, d_theta, ScanAngle [deg, (N_ele,)], d_sample [m, (Nt,)], t_sample [s, (Nt,)]).

    Returns
    -------
    G : (N_ele*(idx_t_end-idx_t_start), Nz*Nx) float32
        Imaging/system matrix in column-stacked point order, element-time stacked by rows.
    idx_t_end : int
        Computed end index (0-based) of the time window used for G.
    recon_size : (Nz, Nx)
        Reconstruction grid size.
    """
    # --- DEFINE RECONSTRUCTION AREA ---
    d_sample = np.asarray(info.d_sample, dtype=np.float64)
    r_max_tmp = float(d_sample[idx_t_end_recon])  # distance at that sample
    # Grid steps
    delta_z = info.c / info.fc / 2.0 * 3.0
    delta_x = info.c / info.fc / 2.0 * 5.0
    # Define meshgrid for reconstruction
    x_range = np.arange(-width_radius, width_radius + 1e-12, delta_x, dtype=np.float64)
    z_start = info.ROC + (idx_t_start + 1) * info.pixel_d
    z_end   = r_max_tmp + info.ROC
    z_range = np.arange(z_start, z_end + 1e-12, delta_z, dtype=np.float64)
    xx, zz = np.meshgrid(x_range, z_range, indexing='xy')  # xx: (Nz,Nx), zz: (Nz,Nx)
    r_max_data = np.sqrt((r_max_tmp - width_radius)**2 + (r_max_tmp + info.ROC)**2)
    gt = np.flatnonzero(d_sample > r_max_data)
    if gt.size > 0:
        idx_t_end = int(gt[0])
    else:
        idx_t_end = int(info.Nfocus)  # follows your MATLAB fallback

    Nz_cs, Nx_cs = xx.shape
    wavenum = _TWO_PI * info.fc_signal / info.c
    print(f'>>>>>>>> RECON GRID SIZE IS {Nz_cs} x {Nx_cs}')
    recon_size = (Nz_cs, Nx_cs)

    # TRANSDUCER ELEMENT POSITIONS
    ele_angle_radian = np.asarray(info.EleAngle, dtype=np.float64) * np.pi / 180.0  # (N_ele,)
    thetad_radian = np.pi/2.0 - np.abs(ele_angle_radian)

    N_ele = int(info.N_ele)
    sgn_xd = np.sign(np.arange(1, N_ele + 1, dtype=np.float64) - (N_ele/2.0) - 0.5)

    roc_correction = 1.0
    xd_coord = (info.ROC * roc_correction) * sgn_xd * np.cos(thetad_radian)  # (N_ele,)
    zd_coord = (info.ROC * roc_correction) * np.sin(thetad_radian)           # (N_ele,)

    # ALLOCATE G
    Nt_seg = int(idx_t_end - idx_t_start)
    if Nt_seg <= 0:
        raise ValueError("idx_t_end must be > idx_t_start")

    G = np.zeros((N_ele * Nt_seg, Nz_cs * Nx_cs), dtype=np.float32)
    t_seg = d_sample[idx_t_start:idx_t_end] / info.c # length = Nt_seg
    scan_angle_deg = np.asarray(info.ScanAngle, dtype=np.float64)  # (N_ele,)

    # MAIN LOOPS (z, x)
    col = 0
    for idx_z in range(Nz_cs):
        z_coord_row = zz[idx_z, :]  # (Nx_cs,)
        x_coord_row = xx[idx_z, :]
        for idx_x in range(Nx_cs):
            x_coord = float(x_coord_row[idx_x])
            z_coord = float(z_coord_row[idx_x])

            # Distances and delays to each element
            diff_x = x_coord - xd_coord          # (N_ele,)
            diff_z = z_coord - zd_coord          # (N_ele,)
            r_zx   = np.sqrt(diff_x*diff_x + diff_z*diff_z)  # (N_ele,)
            tof    = r_zx / info.c               # (N_ele,) (kept for parity; not re-used)

            # Directional factors
            # cos(theta) between (xd,zd) and (diff_x,diff_z)
            denom = r_zx * (info.ROC * roc_correction) + 1e-18
            dir_vec = (xd_coord*diff_x + zd_coord*diff_z) / denom  # (N_ele,)
            dir_vec = np.clip(dir_vec, -1.0, 1.0)  # numeric safety

            sin_theta = np.sqrt(np.maximum(0.0, 1.0 - dir_vec*dir_vec))  # (N_ele,)

            # Handle sin(θ) ≈ 0 safely: limit of sin(0.5*k*w*sinθ)/(k*w*sinθ) → 0.5
            denom_beam = (wavenum * info.ele_width) * np.where(sin_theta > 1e-12, sin_theta, 1.0)
            core = np.sin(0.5 * wavenum * info.ele_width * sin_theta) / denom_beam
            core = np.where(sin_theta > 1e-12, core, 0.5)

            dir_wt = core * (1.0 + dir_vec)

            # Apply gating: *(sign(dir_vec - 0.5) + 1)/2
            # np.sign: (-1 if <0), (0 if ==0), (1 if >0)
            gate = (np.sign(dir_vec - 0.5) + 1.0) * 0.5
            dir_wt *= gate  # (N_ele,)

            # System impulse response from this point (Nt_seg, N_ele)
            G_r = EIR(t_seg, scan_angle_deg, np.array([z_coord, x_coord], dtype=np.float64), info)

            # Apply geometric & directional weights per element
            w = (dir_wt / np.maximum(r_zx, 1e-18)).reshape(1, -1)  # (1, N_ele)
            G_r = G_r * w  # broadcast along time

            # Store column (MATLAB uses G(:,col) = G_r(:) with column-major)
            G[:, col] = np.ravel(G_r, order='F').astype(np.float32)
            col += 1

        #print(f"Row {idx_z+1}/{Nz_cs} complete ({(idx_z+1)/Nz_cs:.1%})")

    G[np.isnan(G)] = 0.0
    print('>>>>>>>> IMAGING MATRIX FINISH COMPUTING.')
    return G, idx_t_end, recon_size