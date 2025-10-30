from __future__ import annotations
import os
from scipy import io as spio
from dataclasses import dataclass, asdict
import numpy as np
from typing import Any, Sequence


def _load_sequence_info():
    mat_file_path = os.path.join('..','artifacts','Sequence.mat')
    mat = spio.loadmat(mat_file_path)
    Roi = mat.get('Roi')
    System = mat.get('System')
    return Roi, System

@dataclass
class ConvexSystemParam:
    c: float
    fs: float
    N_ele: float
    pitch: float
    fc: float
    ROC: float # radius of curvature
    ele_width: float
    ele_height: float
    fc_signal: float = 2.0e6 # default 2.0 MHz for PA
    pixel_d: float = None
    N_sc: int = None
    N_ch: int = None
    Nfocus: int = None
    fc_scaled: float = None
    RxFnum: float = None
    FOV: float = None
    start_angle: float = None
    d_theta_sc: float = None
    d_theta_ele: float = None
    ScanAngle: np.ndarray = None
    EleAngle: np.ndarray = None
    half_rx_ch: float = None
    d_sample: np.ndarray = None

    def __post_init__(self):
        # Derived scalars
        if self.pixel_d is None:
            self.pixel_d = self.c / self.fs / 2
        if self.fc_scaled is None:
            self.fc_scaled = self.fc / self.fs * self.Nfocus / 2
        if self.d_theta_sc is None:
            self.d_theta_sc = self.FOV / (self.N_sc - 1)
        if self.d_theta_ele is None:
            self.d_theta_ele = self.FOV / (self.N_ele - 1)
        if self.half_rx_ch is None:
            self.half_rx_ch = self.N_ch * self.pitch * 0.5

        # Derived arrays
        if self.ScanAngle is None:
            self.ScanAngle = np.linspace(self.start_angle, self.start_angle + (self.N_sc-1)*self.d_theta_sc, self.N_sc)
        if self.EleAngle is None:
            self.EleAngle = np.linspace(-self.FOV/2.0, self.FOV/2.0, self.N_ele)
        if self.d_sample is None:
            self.d_sample = np.arange(self.Nfocus) * self.pixel_d
    
    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

def _load_csystem_param_us(Roi: Sequence[Any], System: Any, *,
                           offset: int = 0, default_Nfocus: int = 3000) -> ConvexSystemParam:
    c = 1540.0  # speed of sound in tissue (m/s)
    transducer = System["Transducer"][0]
    parameters = System["Parameters"][0]

    fs = float(parameters['sampleFreqMHz'][0][0]) * 1e6  # Hz
    ROC = float(transducer['radiusCm'][0][0]) * 1e-2 # m
    pitch = float(transducer['elementPitchCm'][0][0]) / 100.0 # convert cm->m
    fc = float(transducer['frequencyMHz'][0][0]) * 1e6 # convert MHz->Hz
    N_ele = int(transducer['elementCnt'][0][0])
    ele_width = float(transducer['elementWidthCm'][0][0]) / 100.0 # convert cm->m
    ele_height = 6e-3  # m
    pixel_d = c / fs / 2.0  # physical distance per sample ROUND-TRIP (m)
    N_sc = N_ele

    N_ch = int(parameters['receiveNum'][0][0])
    Nfocus = int(default_Nfocus)
    fc_scaled = fc / fs * Nfocus / 2.0
    RxFnum = 1.0
    lateral_length = Roi[0]['lateralLength'][0][0]
    FOV = float(lateral_length) # degrees
    lateral_start = Roi[0]['lateralStart'][0][0]
    start_angle = float(lateral_start)
    d_theta_sc = FOV / (N_sc-1) # separation between scan lines
    ScanAngle = np.linspace(start_angle, start_angle + (N_sc - 1) * d_theta_sc, num = N_sc)
    tmp = -FOV / 2.0
    EleAngle = np.arange(tmp, -tmp + pitch * 0.5, pitch)
    half_rx_ch = N_ch * pitch * 0.5
    n_sample = np.arange(Nfocus, dtype=float) + float(offset)
    d_sample = n_sample * pixel_d

    return ConvexSystemParam(
        c=c,
        fs=fs,
        N_ele=N_ele,
        pitch=pitch,
        fc=fc,
        ROC=ROC,
        ele_width=ele_width,
        ele_height=ele_height,
        pixel_d=pixel_d,
        N_sc=N_sc,
        N_ch=N_ch,
        Nfocus=Nfocus,
        fc_scaled=fc_scaled,
        RxFnum=RxFnum,
        FOV=FOV,
        start_angle=start_angle,
        d_theta_sc=d_theta_sc,
        ScanAngle=ScanAngle,
        EleAngle=EleAngle,
        half_rx_ch=half_rx_ch,
        d_sample=d_sample,
    )

def _load_csystem_param_pa(Roi: Sequence[Any], System: Any, *,
                           offset: int = 0, default_Nfocus: int = 1500) -> ConvexSystemParam:
    c = 1540.0  # speed of sound in tissue (m/s)
    transducer = System["Transducer"][0]
    parameters = System["Parameters"][0]

    fs = float(parameters['sampleFreqMHz'][0][0]) * 1e6  # Hz
    ROC = float(transducer['radiusCm'][0][0]) * 1e-2 # m
    pitch = float(transducer['elementPitchCm'][0][0]) / 100.0 # convert cm->m
    fc = float(transducer['frequencyMHz'][0][0]) * 1e6 # convert MHz->Hz
    N_ele = int(transducer['elementCnt'][0][0])
    ele_width = float(transducer['elementWidthCm'][0][0]) / 100.0 # convert cm->m
    ele_height = 6e-3  # m
    pixel_d = c / fs  # physical distance per sample ONE-WAY-TRIP (m)
    N_sc = N_ele

    N_ch = int(parameters['receiveNum'][0][0])
    Nfocus = int(default_Nfocus)
    fc_scaled = fc / fs * Nfocus / 2.0
    RxFnum = 1.0
    lateral_length = Roi[0]['lateralLength'][0][0]
    FOV = float(lateral_length) # degrees
    lateral_start = Roi[0]['lateralStart'][0][0]
    start_angle = float(lateral_start)
    d_theta_sc = FOV / (N_sc-1) # separation between scan lines
    ScanAngle = np.linspace(start_angle, start_angle + (N_sc - 1) * d_theta_sc, num = N_sc)
    tmp = -FOV / 2.0
    EleAngle = np.arange(tmp, -tmp + pitch * 0.5, pitch)
    half_rx_ch = N_ch * pitch * 0.5
    n_sample = np.arange(Nfocus, dtype=float) + float(offset)
    d_sample = n_sample * pixel_d

    return ConvexSystemParam(
        c=c,
        fs=fs,
        N_ele=N_ele,
        pitch=pitch,
        fc=fc,
        ROC=ROC,
        ele_width=ele_width,
        ele_height=ele_height,
        pixel_d=pixel_d,
        N_sc=N_sc,
        N_ch=N_ch,
        Nfocus=Nfocus,
        fc_scaled=fc_scaled,
        RxFnum=RxFnum,
        FOV=FOV,
        start_angle=start_angle,
        d_theta_sc=d_theta_sc,
        ScanAngle=ScanAngle,
        EleAngle=EleAngle,
        half_rx_ch=half_rx_ch,
        d_sample=d_sample,
    )

def convex_us_param():
    roi_us , sys_us = _load_sequence_info()
    return _load_csystem_param_us(roi_us , sys_us)

def convex_pa_param():
    roi_us , sys_us = _load_sequence_info()
    return _load_csystem_param_pa(roi_us , sys_us)

def list_subfolders(folder_name: str):
    all_items = os.listdir(folder_name)
    all_folders = [os.path.join(folder_name, f) for f in all_items if os.path.isdir(os.path.join(folder_name, f))]
    return all_folders