import numpy as np
from numpy.fft import fft2, ifft2
import math
from typing import List

def _partition_freq_waveatom(H: int) -> List[np.ndarray]:
    if H<16:
        raise ValueError("H must be greater than 16")
    length_hint = int(math.ceil(math.log2(H) / 2.0)) + 1
    bump_samples: List[np.ndarray] = [None] * length_hint  # type: ignore

    bump_samples[0] = np.array([1, 1], dtype=bool)
    bump_samples[1] = np.array([0, 1], dtype=bool)
    bump_samples[2] = np.array([0, 1, 1, 1], dtype=bool)

    cnt, idx , rad = 16 , 3 , 4
    last_filled = 2
    while cnt < H:
        cnt_prev = cnt
        py_i = idx - 1
        if py_i >= len(bump_samples):
            bump_samples.extend([None] * (py_i - len(bump_samples) + 1))
        bump_samples[py_i] = np.concatenate([bump_samples[py_i], np.ones(2, dtype=bool)])
        last_filled = max(last_filled, py_i)

        cnt = cnt + 2 * rad
        idx = idx + 1
        rad = 2 * rad
        trg = min(4 * cnt_prev, H)
        # bumpSamples{idx} = [zeros(1, cnt/rad), ones(1, (trg-cnt)/rad)];
        py_i = idx - 1
        if py_i >= len(bump_samples):
            bump_samples.extend([None] * (py_i - len(bump_samples) + 1))

        zlen = cnt // rad
        olen = (trg - cnt) // rad
        bump_samples[py_i] = np.concatenate([np.zeros(zlen, dtype=bool), np.ones(olen, dtype=bool)])
        last_filled = max(last_filled, py_i)
        cnt = trg
    # Trim to only the entries actually written
    out = [b for b in bump_samples[:last_filled + 1] if b is not None]
    return out

def _bump_function(w) -> np.ndarray:
    g = np.zeros_like(w, dtype=np.float64)
    mask_main = (w < (5.0 * np.pi / 6.0)) & (w > (-7.0 * np.pi / 6.0))
    if not np.any(mask_main):
        return g
    # Shift for sf: v = w - 3π/2
    v = w[mask_main] - (3.0 * np.pi / 2.0)
    av = np.abs(v)
    # Prepare container for sf(v)
    sf_vals = np.zeros_like(v, dtype=np.float64)

    # Band 1: 2π/3 <= |v| <= 4π/3  -> sf = (1/√2) * hf(v/2 + π)
    b1 = (av >= (2.0 * np.pi / 3.0)) & (av <= (4.0 * np.pi / 3.0))
    if np.any(b1):
        t1 = v[b1] * 0.5 + np.pi                 # argument to hf
        # hf inline:
        t1 = (t1 + np.pi) % (2.0 * np.pi) - np.pi  # wrap to (-π, π]
        at1 = np.abs(t1)
        # beta(3*|t|/π - 1)
        bet1 = 0.5 * (1.0 - np.cos(np.pi * (3.0 * at1 / np.pi - 1.0)))
        hf1 = np.sqrt(2.0) * np.cos((np.pi / 2.0) * bet1)
        # overrides
        hf1[at1 <= (np.pi / 3.0)] = np.sqrt(2.0)
        hf1[at1 >= (2.0 * np.pi / 3.0)] = 0.0
        sf_vals[b1] = (1.0 / np.sqrt(2.0)) * hf1

    # Band 2: 4π/3 <= |v| <= 8π/3  -> sf = (1/√2) * hf(v/4)
    b2 = (av >= (4.0 * np.pi / 3.0)) & (av <= (8.0 * np.pi / 3.0))
    if np.any(b2):
        t2 = v[b2] * 0.25                         # argument to hf
        # hf inline:
        t2 = (t2 + np.pi) % (2.0 * np.pi) - np.pi  # wrap to (-π, π]
        at2 = np.abs(t2)
        bet2 = 0.5 * (1.0 - np.cos(np.pi * (3.0 * at2 / np.pi - 1.0)))
        hf2 = np.sqrt(2.0) * np.cos((np.pi / 2.0) * bet2)
        hf2[at2 <= (np.pi / 3.0)] = np.sqrt(2.0)
        hf2[at2 >= (2.0 * np.pi / 3.0)] = 0.0
        sf_vals[b2] = (1.0 / np.sqrt(2.0)) * hf2

    # (|v| <= 2π/3 or |v| > 8π/3) -> sf = 0 already
    # g = |sf|
    g[mask_main] = np.abs(sf_vals)
    return g

def _kf_rt(w , m) -> np.ndarray:
    # alpha_m and eps_m (broadcast with w as needed)
    alpha_m = (np.pi / 2.0) * (m + 0.5)
    eps_m = (-1) ** m
    # bump argument: eps_m * (w - pi*(m+1/2))
    arg = eps_m * (w - np.pi * (m + 0.5))
    # evaluate bump and assemble result
    pref1 = np.exp(-1j * w / 2.0)            # exp(-i w/2)
    pref2 = np.exp(1j * alpha_m)             # exp(+i alpha_m)
    bump_vals = _bump_function(arg)           # real-valued
    r = pref1 * (pref2 * bump_vals)          # complex result
    return np.asarray(r, dtype=np.complex128)

def _kf_lf(w , n) -> np.ndarray:
    an  = (np.pi / 2.0) * (n + 0.5)     # pi/2*(n+1/2)
    en1 = (-1) ** (n + 1)               # (-1)^(n+1)
    arg = en1 * (w + np.pi * (n + 0.5)) # bump argument
    pref1 = np.exp(-1j * w / 2.0)       # exp(-i w/2)
    pref2 = np.exp(-1j * an)            # exp(-i an)
    bump_vals = _bump_function(arg)      # real-valued
    r = pref1 * (pref2 * bump_vals)
    return np.asarray(r, dtype=np.complex128)

def _band_limits(idx_m: int, f_scale: int):
    idx_wc = f_scale * idx_m
    if (idx_m & 1) == 0:
        # even
        wz_start = idx_wc - (2.0 / 3.0) * f_scale
        wz_end   = idx_wc + (4.0 / 3.0) * f_scale
    else:
        # odd
        wz_start = idx_wc - (1.0 / 3.0) * f_scale
        wz_end   = idx_wc + (5.0 / 3.0) * f_scale
    return int(np.ceil(wz_start)), int(np.floor(wz_end))

def _build_axis_cache(Naxis: int, Nm_part: int, f_scale: int, *, is_z_axis: bool):
    cache = []
    small_block_size = 2 * f_scale
    for m in range(Nm_part):
        s, e = _band_limits(m, f_scale)
        # (+) band
        idx_p = np.arange(s, e + 1, dtype=int)
        # (-) band (mirrored)
        idx_n = np.arange(-e, -s + 1, dtype=int)

        # windows
        wscale = (np.pi / f_scale)
        wa_p = (_kf_rt if is_z_axis else _kf_rt)(idx_p * wscale, m)
        wa_n = (_kf_lf if is_z_axis else _kf_lf)(idx_n * wscale, m)

        # wrap indices once for both the small block and the full spectrum
        entry = {
            "rt": {
                "idx": idx_p,
                "rz": np.mod(idx_p, small_block_size),     # indices into c_block axis
                "RS": np.mod(idx_p, Naxis),                # indices into FFT axis
                "wa": wa_p.astype(np.complex128, copy=False),
            },
            "lf": {
                "idx": idx_n,
                "rz": np.mod(idx_n, small_block_size),
                "RS": np.mod(idx_n, Naxis),
                "wa": wa_n.astype(np.complex128, copy=False),
            },
        }
        cache.append(entry)
    return cache

def _accumulate_block(c_block: np.ndarray,
                      z_pack: dict, x_pack: dict,
                      x_spectrum: np.ndarray):
    rz, cx = z_pack["rz"], x_pack["rz"]
    RS, CS = z_pack["RS"], x_pack["CS"]  # 'CS' will be attached by caller
    # separable weight outer product
    W = np.conj(z_pack["wa"])[:, None] * x_pack["wa"][None, :]
    # gather the spectrogram tile once
    sub_spec = x_spectrum[np.ix_(RS, CS)]
    c_block[np.ix_(rz, cx)] += W * sub_spec

def wave_atom_transform2d(x: np.ndarray, tf_type: str):
    """
    Returns:
      - 'ortho' -> List[np.ndarray] length Nj
      - 'directional' -> {'z': List[np.ndarray], 'x': List[np.ndarray]}
      - 'complex' -> {'q1','q2','q3','q4'} each a List[np.ndarray]
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    Nz, Nx = x.shape

    Nmin = min(Nz, Nx)
    wPartList = _partition_freq_waveatom(Nmin // 2)
    Nj = len(wPartList)
    x_spectrum = fft2(x) / np.sqrt(Nz * Nx)

    if tf_type == 'ortho':
        out_scales = [None] * Nj
        for j in range(1, Nj + 1):
            wPart = wPartList[j - 1]
            Nm = len(wPart)
            block_size = 2 ** j           # spatial sub-block edge
            f_scale = 2 ** (j - 1)

            # precompute caches per axis
            # z cache: length Nm; x cache: length Nm
            z_cache = _build_axis_cache(Nz, Nm, f_scale, is_z_axis=True)
            x_cache = _build_axis_cache(Nx, Nm, f_scale, is_z_axis=False)

            # attach 'CS' once on x side (FFT column indices) for both rt and lf
            for m in range(Nm):
                x_cache[m]["rt"]["CS"] = x_cache[m]["rt"]["RS"]
                x_cache[m]["lf"]["CS"] = x_cache[m]["lf"]["RS"]

            out_block = np.zeros((block_size * Nm, block_size * Nm), dtype=np.complex128)
            # work buffer reused
            c_jm = np.zeros((2 * f_scale, 2 * f_scale), dtype=np.complex128)

            for mz in range(Nm):
                # quick skip if both suppressed on this axis and we’ll AND them later
                for mx in range(Nm):
                    if not (wPart[mz] or wPart[mx]):
                        continue

                    i0 = mz * block_size
                    j0 = mx * block_size

                    # reset accumulator (reuse buffer)
                    c_jm.fill(0.0)

                    # (+,+)
                    _accumulate_block(c_jm, z_cache[mz]["rt"], x_cache[mx]["rt"], x_spectrum)
                    # (-,-)
                    _accumulate_block(c_jm, z_cache[mz]["lf"], x_cache[mx]["lf"], x_spectrum)

                    # inverse FFT and scale (same as MATLAB)
                    out_block[i0:i0 + block_size, j0:j0 + block_size] = ifft2(c_jm) * (2 * f_scale)

            out_scales[j - 1] = out_block
        return out_scales

    elif tf_type == 'directional':
        out_z = [None] * Nj
        out_x = [None] * Nj
        for j in range(1, Nj + 1):
            wPart = wPartList[j - 1]
            Nm = len(wPart)
            block_size = 2 ** j
            f_scale = 2 ** (j - 1)

            z_cache = _build_axis_cache(Nz, Nm, f_scale, is_z_axis=True)
            x_cache = _build_axis_cache(Nx, Nm, f_scale, is_z_axis=False)
            for m in range(Nm):
                x_cache[m]["rt"]["CS"] = x_cache[m]["rt"]["RS"]
                x_cache[m]["lf"]["CS"] = x_cache[m]["lf"]["RS"]

            OZ = np.zeros((block_size * Nm, block_size * Nm), dtype=np.complex128)
            OX = np.zeros_like(OZ)

            c = np.zeros((2 * f_scale, 2 * f_scale), dtype=np.complex128)

            for mz in range(Nm):
                for mx in range(Nm):
                    if not (wPart[mz] or wPart[mx]):
                        continue
                    i0 = mz * block_size
                    j0 = mx * block_size

                    # Z-dominant (rt+rt, lf+lf)
                    c.fill(0.0)
                    _accumulate_block(c, z_cache[mz]["rt"], x_cache[mx]["rt"], x_spectrum)
                    _accumulate_block(c, z_cache[mz]["lf"], x_cache[mx]["lf"], x_spectrum)
                    OZ[i0:i0 + block_size, j0:j0 + block_size] = ifft2(c) * (2 * f_scale)

                    # X-dominant (rt+lf, lf+rt)
                    c.fill(0.0)
                    _accumulate_block(c, z_cache[mz]["rt"], x_cache[mx]["lf"], x_spectrum)
                    _accumulate_block(c, z_cache[mz]["lf"], x_cache[mx]["rt"], x_spectrum)
                    OX[i0:i0 + block_size, j0:j0 + block_size] = ifft2(c) * (2 * f_scale)

            out_z[j - 1] = OZ
            out_x[j - 1] = OX
        return {"z": out_z, "x": out_x}

    elif tf_type == 'complex':
        q1 = [None] * Nj
        q2 = [None] * Nj
        q3 = [None] * Nj
        q4 = [None] * Nj
        for j in range(1, Nj + 1):
            wPart = wPartList[j - 1]
            Nm = len(wPart)
            block_size = 2 ** j
            f_scale = 2 ** (j - 1)

            z_cache = _build_axis_cache(Nz, Nm, f_scale, is_z_axis=True)
            x_cache = _build_axis_cache(Nx, Nm, f_scale, is_z_axis=False)
            for m in range(Nm):
                x_cache[m]["rt"]["CS"] = x_cache[m]["rt"]["RS"]
                x_cache[m]["lf"]["CS"] = x_cache[m]["lf"]["RS"]

            O1 = np.zeros((block_size * Nm, block_size * Nm), dtype=np.complex128)
            O2 = np.zeros_like(O1)
            O3 = np.zeros_like(O1)
            O4 = np.zeros_like(O1)

            c = np.zeros((2 * f_scale, 2 * f_scale), dtype=np.complex128)

            for mz in range(Nm):
                for mx in range(Nm):
                    if not (wPart[mz] or wPart[mx]):
                        continue
                    i0 = mz * block_size
                    j0 = mx * block_size

                    # (rt, rt)
                    c.fill(0.0)
                    _accumulate_block(c, z_cache[mz]["rt"], x_cache[mx]["rt"], x_spectrum)
                    O1[i0:i0 + block_size, j0:j0 + block_size] = ifft2(c) * (2 * f_scale)

                    # (lf, lf)
                    c.fill(0.0)
                    _accumulate_block(c, z_cache[mz]["lf"], x_cache[mx]["lf"], x_spectrum)
                    O2[i0:i0 + block_size, j0:j0 + block_size] = ifft2(c) * (2 * f_scale)

                    # (rt, lf)
                    c.fill(0.0)
                    _accumulate_block(c, z_cache[mz]["rt"], x_cache[mx]["lf"], x_spectrum)
                    O3[i0:i0 + block_size, j0:j0 + block_size] = ifft2(c) * (2 * f_scale)

                    # (lf, rt)
                    c.fill(0.0)
                    _accumulate_block(c, z_cache[mz]["lf"], x_cache[mx]["rt"], x_spectrum)
                    O4[i0:i0 + block_size, j0:j0 + block_size] = ifft2(c) * (2 * f_scale)

            q1[j - 1] = O1
            q2[j - 1] = O2
            q3[j - 1] = O3
            q4[j - 1] = O4

        return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}

    else:
        raise ValueError("TRANSFORM TYPE UNDEFINED: expected 'ortho', 'directional', or 'complex'")

def _gather_block(c_block: np.ndarray,
                  z_pack: dict, x_pack: dict) -> np.ndarray:
    return c_block[np.ix_(z_pack["rz"], x_pack["rz"])]

def iwave_atom_transform2d(coeffArray, N, tf_type: str) -> np.ndarray:
    """
    Parameters
    ----------
    coeffArray :
        - 'ortho'       -> List[np.ndarray]               (length Nj)
        - 'directional' -> {'z': List[np.ndarray], 'x': List[np.ndarray]}
        - 'complex'     -> {'q1','q2','q3','q4'} dict of List[np.ndarray]
    N : tuple(int, int)
        (Nz, Nx) of the target reconstruction.
    tf_type : {'ortho','directional','complex'}

    Returns
    -------
    x_recon : (Nz, Nx) complex ndarray
        (Real if the original forward used real inputs and symmetric tiling.)
    """
    Nz, Nx = int(N[0]), int(N[1])
    Nmin = min(Nz, Nx)
    wPartList = _partition_freq_waveatom(Nmin // 2)
    Nj = len(wPartList)
    x_spectrum = np.zeros((Nz, Nx), dtype=np.complex128)

    if tf_type == 'ortho':
        # coeffArray: List[np.ndarray], each shape = ( (2**j)*Nm , (2**j)*Nm )
        for j in range(1, Nj + 1):
            blocks = coeffArray[j - 1]
            wPart = wPartList[j - 1]
            Nm = len(wPart)
            block_size = 2 ** j
            f_scale = 2 ** (j - 1)

            # Precompute caches (same as forward)
            z_cache = _build_axis_cache(Nz, Nm, f_scale, is_z_axis=True)
            x_cache = _build_axis_cache(Nx, Nm, f_scale, is_z_axis=False)
            for m in range(Nm):
                x_cache[m]["rt"]["CS"] = x_cache[m]["rt"]["RS"]
                x_cache[m]["lf"]["CS"] = x_cache[m]["lf"]["RS"]

            # Work buffer for the FFT of one spatial block
            c_fft = np.empty((2 * f_scale, 2 * f_scale), dtype=np.complex128)

            for mz in range(Nm):
                for mx in range(Nm):
                    if not (wPart[mz] or wPart[mx]):
                        continue

                    i0 = mz * block_size
                    j0 = mx * block_size
                    # c_jm in spectral domain (inverse of forward’s ifft2 * (2*f))
                    c_spatial = blocks[i0:i0 + block_size, j0:j0 + block_size]
                    c_fft[:] = fft2(c_spatial) / (2 * f_scale)

                    # (+,+): add W * c_tile into x_spectrum at (RS,CS)
                    sub = _gather_block(c_fft, z_cache[mz]["rt"], x_cache[mx]["rt"])
                    W = (z_cache[mz]["rt"]["wa"])[:, None] * (x_cache[mx]["rt"]["wa"])[None, :]
                    x_spectrum[np.ix_(z_cache[mz]["rt"]["RS"], x_cache[mx]["rt"]["CS"])] += W * sub

                    # (-,-)
                    sub = _gather_block(c_fft, z_cache[mz]["lf"], x_cache[mx]["lf"])
                    W = (z_cache[mz]["lf"]["wa"])[:, None] * (x_cache[mx]["lf"]["wa"])[None, :]
                    x_spectrum[np.ix_(z_cache[mz]["lf"]["RS"], x_cache[mx]["lf"]["CS"])] += W * sub

        x_recon = ifft2(x_spectrum) * np.sqrt(Nz * Nx)
        return x_recon

    elif tf_type == 'directional':
        # coeffArray may be the dict {'z': [...], 'x': [...]} from our forward.
        if isinstance(coeffArray, dict):
            coeffArray_z = coeffArray["z"]
            coeffArray_x = coeffArray["x"]
        else:
            # or an Nj x 2 MATLAB-style cell, emulate columns 0 and 1
            coeffArray_z = [row[0] for row in coeffArray]
            coeffArray_x = [row[1] for row in coeffArray]

        for j in range(1, Nj + 1):
            blocks_z = coeffArray_z[j - 1]
            blocks_x = coeffArray_x[j - 1]
            wPart = wPartList[j - 1]
            Nm = len(wPart)
            block_size = 2 ** j
            f_scale = 2 ** (j - 1)

            z_cache = _build_axis_cache(Nz, Nm, f_scale, is_z_axis=True)
            x_cache = _build_axis_cache(Nx, Nm, f_scale, is_z_axis=False)
            for m in range(Nm):
                x_cache[m]["rt"]["CS"] = x_cache[m]["rt"]["RS"]
                x_cache[m]["lf"]["CS"] = x_cache[m]["lf"]["RS"]

            c_fft = np.empty((2 * f_scale, 2 * f_scale), dtype=np.complex128)

            for mz in range(Nm):
                for mx in range(Nm):
                    if not (wPart[mz] or wPart[mx]):
                        continue
                    i0 = mz * block_size
                    j0 = mx * block_size

                    # Z-dominant block
                    c_spatial = blocks_z[i0:i0 + block_size, j0:j0 + block_size]
                    c_fft[:] = fft2(c_spatial) / (2 * f_scale)

                    sub = _gather_block(c_fft, z_cache[mz]["rt"], x_cache[mx]["rt"])
                    W = (z_cache[mz]["rt"]["wa"])[:, None] * (x_cache[mx]["rt"]["wa"])[None, :]
                    x_spectrum[np.ix_(z_cache[mz]["rt"]["RS"], x_cache[mx]["rt"]["CS"])] += W * sub

                    sub = _gather_block(c_fft, z_cache[mz]["lf"], x_cache[mx]["lf"])
                    W = (z_cache[mz]["lf"]["wa"])[:, None] * (x_cache[mx]["lf"]["wa"])[None, :]
                    x_spectrum[np.ix_(z_cache[mz]["lf"]["RS"], x_cache[mx]["lf"]["CS"])] += W * sub

                    # X-dominant block
                    c_spatial = blocks_x[i0:i0 + block_size, j0:j0 + block_size]
                    c_fft[:] = fft2(c_spatial) / (2 * f_scale)

                    sub = _gather_block(c_fft, z_cache[mz]["rt"], x_cache[mx]["lf"])
                    W = (z_cache[mz]["rt"]["wa"])[:, None] * (x_cache[mx]["lf"]["wa"])[None, :]
                    x_spectrum[np.ix_(z_cache[mz]["rt"]["RS"], x_cache[mx]["lf"]["CS"])] += W * sub

                    sub = _gather_block(c_fft, z_cache[mz]["lf"], x_cache[mx]["rt"])
                    W = (z_cache[mz]["lf"]["wa"])[:, None] * (x_cache[mx]["rt"]["wa"])[None, :]
                    x_spectrum[np.ix_(z_cache[mz]["lf"]["RS"], x_cache[mx]["rt"]["CS"])] += W * sub

        x_recon = ifft2(x_spectrum) * np.sqrt(Nz * Nx)
        return x_recon

    elif tf_type == 'complex':
        # Expect dict from our forward:
        if not isinstance(coeffArray, dict):
            # tolerate MATLAB-style Njx4 cell array -> split columns
            coeffArray_1 = [row[0] for row in coeffArray]
            coeffArray_2 = [row[1] for row in coeffArray]
            coeffArray_3 = [row[2] for row in coeffArray]
            coeffArray_4 = [row[3] for row in coeffArray]
        else:
            coeffArray_1 = coeffArray["q1"]
            coeffArray_2 = coeffArray["q2"]
            coeffArray_3 = coeffArray["q3"]
            coeffArray_4 = coeffArray["q4"]

        for j in range(1, Nj + 1):
            b1 = coeffArray_1[j - 1]
            b2 = coeffArray_2[j - 1]
            b3 = coeffArray_3[j - 1]
            b4 = coeffArray_4[j - 1]
            wPart = wPartList[j - 1]
            Nm = len(wPart)
            block_size = 2 ** j
            f_scale = 2 ** (j - 1)

            z_cache = _build_axis_cache(Nz, Nm, f_scale, is_z_axis=True)
            x_cache = _build_axis_cache(Nx, Nm, f_scale, is_z_axis=False)
            for m in range(Nm):
                x_cache[m]["rt"]["CS"] = x_cache[m]["rt"]["RS"]
                x_cache[m]["lf"]["CS"] = x_cache[m]["lf"]["RS"]

            c_fft = np.empty((2 * f_scale, 2 * f_scale), dtype=np.complex128)

            for mz in range(Nm):
                for mx in range(Nm):
                    if not (wPart[mz] or wPart[mx]):
                        continue
                    i0 = mz * block_size
                    j0 = mx * block_size

                    # (rt, rt)
                    c_spatial = b1[i0:i0 + block_size, j0:j0 + block_size]
                    c_fft[:] = fft2(c_spatial) / (2 * f_scale)
                    sub = _gather_block(c_fft, z_cache[mz]["rt"], x_cache[mx]["rt"])
                    W = (z_cache[mz]["rt"]["wa"])[:, None] * (x_cache[mx]["rt"]["wa"])[None, :]
                    x_spectrum[np.ix_(z_cache[mz]["rt"]["RS"], x_cache[mx]["rt"]["CS"])] += W * sub

                    # (lf, lf)
                    c_spatial = b2[i0:i0 + block_size, j0:j0 + block_size]
                    c_fft[:] = fft2(c_spatial) / (2 * f_scale)
                    sub = _gather_block(c_fft, z_cache[mz]["lf"], x_cache[mx]["lf"])
                    W = (z_cache[mz]["lf"]["wa"])[:, None] * (x_cache[mx]["lf"]["wa"])[None, :]
                    x_spectrum[np.ix_(z_cache[mz]["lf"]["RS"], x_cache[mx]["lf"]["CS"])] += W * sub

                    # (rt, lf)
                    c_spatial = b3[i0:i0 + block_size, j0:j0 + block_size]
                    c_fft[:] = fft2(c_spatial) / (2 * f_scale)
                    sub = _gather_block(c_fft, z_cache[mz]["rt"], x_cache[mx]["lf"])
                    W = (z_cache[mz]["rt"]["wa"])[:, None] * (x_cache[mx]["lf"]["wa"])[None, :]
                    x_spectrum[np.ix_(z_cache[mz]["rt"]["RS"], x_cache[mx]["lf"]["CS"])] += W * sub

                    # (lf, rt)
                    c_spatial = b4[i0:i0 + block_size, j0:j0 + block_size]
                    c_fft[:] = fft2(c_spatial) / (2 * f_scale)
                    sub = _gather_block(c_fft, z_cache[mz]["lf"], x_cache[mx]["rt"])
                    W = (z_cache[mz]["lf"]["wa"])[:, None] * (x_cache[mx]["rt"]["wa"])[None, :]
                    x_spectrum[np.ix_(z_cache[mz]["lf"]["RS"], x_cache[mx]["rt"]["CS"])] += W * sub

        x_recon = ifft2(x_spectrum) * np.sqrt(Nz * Nx)
        return x_recon

    else:
        raise ValueError("TRANSFORM TYPE NOT DEFINED. Choose 'ortho', 'directional', or 'complex'.")