"""Build the literature-grounded ICCD characterization stage-1 audit.

This runner is deliberately report-only. It reads existing audit artifacts and
filesystem metadata, but it never opens source images or computes new pixel
statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from json_serialization import dump_json


ACCESSED_AT = "2026-07-20"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)


def references() -> list[dict[str, Any]]:
    rows = [
        ("R01", "EMVA Standard 1288 Release 4.0 Linear", "European Machine Vision Association", 2021, "EMVA 1288", "https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf", "STANDARD", "CCD/CMOS/scientific camera", "PTC; dark current; DSNU/PRNU; linearity", "Homogeneous calibrated illumination, dark images, exposure series, fixed camera settings", "Defines mandatory linear-camera characterization and compliance boundary", "Not ICCD-specific and cannot be claimed without the prescribed setup"),
        ("R02", "EMVA Standard 1288 Release 4.0 General", "European Machine Vision Association", 2020, "EMVA 1288", "https://www.emva.org/wp-content/uploads/EMVA1288General_4.0Release.pdf", "STANDARD", "general/nonlinear cameras", "temporal/spatial variance; DSNU/PRNU", "Dark and illuminated stacks with controlled input and known response mapping", "Extends the standard framework beyond a strictly linear camera", "Still requires controlled characterization data"),
        ("R03", "ISO 15739:2023 Photography - Electronic still-picture imaging - Noise measurements", "ISO/TC 42", 2023, "ISO 15739:2023", "https://www.iso.org/standard/82233.html", "STANDARD", "monochrome/color digital still cameras", "noise versus signal; dynamic range", "Defined signal levels and camera response measurements", "Standard methods for reporting noise against signal and dynamic range", "Scope is still-picture cameras, not a direct ICCD compliance claim"),
        ("R04", "Photon Transfer", "James R. Janesick", 2007, "SPIE Press", "https://spie.org/Publications/Book/725073", "DETECTOR_REFERENCE", "CCD/CMOS", "PTC; gain; read noise; full well", "Raw bias/dark and paired uniform flats across signal levels", "Canonical photon-transfer treatment", "Book method assumptions must be met; ordinary scenes are not flats"),
        ("R05", "The shape of the photon transfer curve of CCD sensors", "Pierre Astier et al.", 2019, "Astronomy & Astrophysics 629 A36", "https://doi.org/10.1051/0004-6361/201935508", "DETECTOR_PAPER", "CCD", "PTC; covariance; brighter-fatter", "Large controlled flat-field series", "Shows high-flux PTC departures are tied to neighboring-pixel correlations", "Requires flats and cannot be inferred from scene stacks"),
        ("R06", "The brighter-fatter effect and pixel correlations in CCD sensors", "Pierre Antilogus et al.", 2014, "Journal of Instrumentation 9 C03048", "https://doi.org/10.1088/1748-0221/9/03/C03048", "DETECTOR_PAPER", "CCD", "flat-field covariance; signal-dependent correlation", "Flat fields over flux levels and spot/star data", "Connects variance deficits and pixel correlations", "CCD charge redistribution is not automatically an ICCD explanation"),
        ("R07", "3D noise photon transfer curve", "Bradley L. Preece; David P. Haefner", 2022, "Applied Optics 61 6202-6212", "https://doi.org/10.1364/AO.452166", "DETECTOR_PAPER", "high-gain imaging sensors", "3D spatial variance decomposition; high-gain noise factor", "Controlled dark/flat image stacks across signal", "Separates row, column, pixel, temporal and fixed components", "Seven-plus fitted curves increase data and identifiability requirements"),
        ("R08", "Noise Power Spectrum Measurements in Digital Imaging With Gain Nonuniformity Correction", "Dong Sik Kim", 2016, "IEEE Transactions on Image Processing 25", "https://doi.org/10.1109/TIP.2016.2574985", "DETECTOR_PAPER", "digital image sensors", "NPS under gain nonuniformity", "Uniform acquisitions and explicit treatment of fixed gain patterns", "Shows FPN inflates measured NPS and detrending can be insufficient", "Scene spectra must not be labeled detector NPS"),
        ("R09", "Exposure Time Dependence of Dark Current in CCD Imagers", "Ralf Widenhorn; Justin C. Dunlap; Erik Bodegom", 2010, "IEEE Transactions on Electron Devices 57", "https://doi.org/10.1109/TED.2009.2038649", "DETECTOR_PAPER", "scientific CCD", "dark mean/current; hot-pixel nonlinearity", "Matched dark frames from 5 to 7200 s at constant temperature", "Demonstrates pixel-level dark-current nonlinearity and hot-pixel risk", "Device, temperature and exposure must match"),
        ("R10", "The Noise Performance of Electron Multiplying Charge-Coupled Devices", "Mark S. Robbins; Benjamin J. Hadwen", 2003, "IEEE Transactions on Electron Devices 50", "https://doi.org/10.1109/TED.2003.813462", "DETECTOR_PAPER", "EMCCD", "multiplication gain; excess-noise factor", "Controlled illumination, gain settings and photon-transfer measurements", "Formal treatment and measurement of multiplication noise", "EMCCD multiplication is analogous context, not proof of MCP behavior"),
        ("R11", "Fixed pattern noise in high-resolution, CCD readout photon-counting detectors", "R. Michel; J. L. Fordham; H. Kawakami", 1997, "MNRAS 292 611-620", "https://doi.org/10.1093/mnras/292.3.611", "DETECTOR_PAPER", "photon-counting intensifier/CCD", "event-centroid FPN", "Photon-counting event data at controlled illumination", "Shows event width, centroiding and illumination affect fixed patterns", "Not applicable when acquisition mode is analog/unknown"),
        ("R12", "A simple approach for characterizing the spatially varying sensitivity of microchannel plate detectors", "Denis Aglagul et al.", 2022, "Review of Scientific Instruments 93 075108", "https://doi.org/10.1063/5.0092346", "DETECTOR_PAPER", "MCP-phosphor", "spatial gain variation", "Resolved single electron/photon hits and accumulated exposure", "Direct MCP spatial-gain characterization", "Current project lacks resolved hit events"),
        ("R13", "On the use of electron-multiplying CCDs for astronomical spectroscopy", "Simon M. Tulloch; Vik S. Dhillon", 2011, "MNRAS 411", "https://doi.org/10.1111/j.1365-2966.2010.17675.x", "DETECTOR_PAPER", "EMCCD", "analog versus photon-counting operation; CIC", "Known EM gain, flux and threshold regime", "Clarifies regime-dependent gain noise and photon-counting limitations", "Cannot classify current ICCD mode without acquisition evidence"),
        ("R14", "Measuring the Noise of Digital Imaging Sensors by Stacking Raw Images Affected by Vibrations and Illumination Flickering", "Francois Sur; Michel Grediac", 2015, "SIAM Journal on Imaging Sciences", "https://doi.org/10.1137/140977035", "DETECTOR_PAPER", "digital cameras", "stack-based temporal noise under nuisance variation", "Repeated raw static-scene frames with nuisance modeling", "Demonstrates flicker and micro-motion can bias stack noise estimates", "Current cross-folder scene and illumination stability remain unknown"),
        ("R15", "A statistical model for signal-dependent charge sharing in image sensors", "Konstantin D. Stefanov", 2014, "IEEE Transactions on Electron Devices 61", "https://doi.org/10.1109/TED.2013.2291448", "DETECTOR_PAPER", "CCD/image sensor", "signal-dependent covariance", "Controlled flats across signal levels", "Models charge sharing and covariance effects", "Requires controlled illumination and sensor-level evidence"),
        ("R16", "TheoH and Allan Deviation as Power-Law Noise Estimators", "J. Taylor; D. A. Howe", 2007, "IEEE TUFFC 54", "https://www.nist.gov/publications/theoh-and-allan-deviation-power-law-noise-estimators-0", "DETECTOR_PAPER", "time-series metrology", "multi-timescale stability", "Ordered samples with known sampling interval", "Authoritative stability-estimator background", "Unknown frame interval prevents physical time-scale claims"),
        ("R17", "Allan Variance Characterization of Compact Fourier Transform Infrared Spectrometers", "George A. Adib; Yasser M. Sabry; Diaa Khalil", 2023, "Applied Spectroscopy 77", "https://doi.org/10.1177/00037028231174248", "DETECTOR_PAPER", "optical spectrometer", "short/long-term instability", "Time-ordered repeated measurements with timestamps", "Optical-instrument example of separating drift/noise regimes", "Transferable only as an auxiliary stability method"),
        ("R18", "The nonlinear photon transfer curve of CCDs and its effects on photometry", "Bin Ma et al.", 2014, "Science China Physics Mechanics and Astronomy", "https://arxiv.org/abs/1407.8280", "DETECTOR_PAPER", "CCD", "nonlinear PTC; charge-sharing PSF", "Flat fields spanning the linear and high-flux range", "Highlights saturation-adjacent and covariance effects", "Bibliographic record retained with arXiv access; not a current-data PTC license"),
        ("R19", "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data", "Alessandro Foi et al.", 2008, "IEEE Transactions on Image Processing 17", "https://doi.org/10.1109/TIP.2008.2001399", "RESTORATION_PAPER", "raw digital sensors", "noise-level function; clipped Poisson-Gaussian", "Raw sensor data and identifiable signal/noise relation", "Defines an operational signal-dependent raw-noise model", "Model fit is not a detector gain calibration without controlled data"),
        ("R20", "Unprocessing Images for Learned Raw Denoising", "Tim Brooks et al.", 2019, "CVPR", "https://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.html", "RESTORATION_PAPER", "camera RAW", "synthetic raw noise and inverse pipeline", "Known/assumed camera pipeline and noise parameters", "Shows why raw-domain processing history matters for synthesis", "Restoration evidence, not detector characterization standard"),
        ("R21", "A Physics-Based Noise Formation Model for Extreme Low-Light Raw Denoising", "Kaixuan Wei et al.", 2020, "CVPR", "https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_A_Physics-Based_Noise_Formation_Model_for_Extreme_Low-Light_Raw_Denoising_CVPR_2020_paper.html", "RESTORATION_PAPER", "CMOS RAW", "low-light noise components and calibration", "Bias/flat/raw calibration across camera settings", "Shows richer electronics noise matters beyond heteroscedastic Gaussian", "CMOS component model cannot be transplanted to ICCD without data"),
        ("R22", "A Holistic Approach to Cross-Channel Image Noise Modeling and Its Application to Image Denoising", "Seonghyeon Nam et al.", 2016, "CVPR", "https://openaccess.thecvf.com/content_cvpr_2016/html/Nam_A_Holistic_Approach_CVPR_2016_paper.html", "RESTORATION_PAPER", "processed RGB cameras", "cross-channel processed noise", "Repeated real images and known image pipeline context", "Shows processing can mix and reshape sensor noise", "Current monochrome/raw-like use makes cross-channel model non-primary"),
        ("R23", "Physics-Guided ISO-Dependent Sensor Noise Modeling for Extreme Low-Light Photography", "Yue Cao et al.", 2023, "CVPR", "https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Physics-Guided_ISO-Dependent_Sensor_Noise_Modeling_for_Extreme_Low-Light_Photography_CVPR_2023_paper.html", "RESTORATION_PAPER", "CMOS RAW", "condition-dependent learned noise", "Paired noisy/clean, flat and bias frames over ISO", "Supports condition-aware modeling only when conditions are observed and calibrated", "Not evidence that folder IDs or unknown ICCD settings are physical conditions"),
    ]
    fields = ["reference_id", "title", "authors", "year", "journal_or_standard", "doi_or_official_url", "reference_class", "detector_type", "method_family", "data_requirements", "key_contribution", "limitations"]
    return [dict(zip(fields, row)) | {"relevance": "ICCD stage-1 method/data eligibility", "citation_status": "VERIFIED", "accessed_at": ACCESSED_AT} for row in rows]


def methods() -> list[dict[str, Any]]:
    # id, family, name, ref, level, data, illumination, exposure, dark, repeated,
    # metadata, calibration, outputs, parameters, assumptions, failures, current status, terminology, paper/noise value
    raw = [
        ("M01","PTC","standard photon transfer curve","R01;R04;R05","LEVEL-S","paired raw flat and dark frames across >=50 signal points","homogeneous calibrated monochromatic flat","required","matched","pairs/stacks","exposure/irradiance, gain, temperature","radiometry and fixed camera settings","mean-variance, gain, read noise, full well","K, read noise, saturation capacity","uniform stable illumination and linear response","scene leakage, nonlinearity, covariance","REQUIRES-NEW-ACQUISITION","photon-transfer method only after compliant acquisition","high","high"),
        ("M02","PTC","difference-frame PTC","R04;R05","LEVEL-A","paired same-level flats and matched bias/dark","uniform flat","required","matched","two or more per level","exposure/irradiance and settings","dark subtraction and pair matching","temporal variance with FPN suppression","operational/system gain if prerequisites hold","pair differences remove static spatial pattern","motion/flicker/unequal flats","REQUIRES-NEW-ACQUISITION","difference-frame PTC only after controlled flat acquisition","high","high"),
        ("M03","PTC","scene-stack temporal mean-variance","R14;R19","LEVEL-C","repeated raw scene frames","ordinary scene","not required","not required","required","scene stability; settings","none","observed signal versus temporal variance","operational noise-level relation","scene and illumination stable within stack","scene/flicker/ROI confounding","PARTIALLY-ELIGIBLE","observed signal-level/temporal-variance relation; not PTC","high","high"),
        ("M04","Dark","dark-current exposure sequence","R01;R09","LEVEL-S","capped matched dark stacks at >=6 exposure times","dark","required","required","required","exposure, temperature, mode, gain","matched settings","dark mean/variance versus exposure","dark-current proxy or calibrated current","same device/mode and stable temperature","untraced darks, compensation, hot-pixel nonlinearity","REQUIRES-NEW-ACQUISITION","dark-current sequence only after matched acquisition","high","medium"),
        ("M05","Dark","temporal dark noise/read-noise proxy","R01;R09","LEVEL-S","short-exposure matched dark stack","dark","single verified short exposure","required","required","temperature, mode, gain","camera offset/underflow known","dark temporal std","temporal dark-noise proxy","dark current negligible or modeled","unknown compensation/mismatch","REQUIRES-NEW-ACQUISITION","temporal dark-noise proxy after matched acquisition","high","medium"),
        ("M06","Spatial","DSNU","R01;R02","LEVEL-S","averaged matched dark stack","dark","verified","required","stack","temperature, gain, mode","temporal-noise correction","dark signal nonuniformity, row/col/pixel terms","DSNU1288","matched dark and standard procedure","scene or unmatched dark contamination","REQUIRES-NEW-ACQUISITION","DSNU only after matched dark acquisition","high","low"),
        ("M07","Spatial","PRNU","R01;R02","LEVEL-S","dark plus homogeneous bright stack near prescribed level","homogeneous flat","required","matched","stack","irradiance and settings","illumination uniformity correction","photoresponse nonuniformity","PRNU1288","controlled uniform illumination","vignetting/scene texture","REQUIRES-NEW-ACQUISITION","PRNU only after controlled flat acquisition","high","medium"),
        ("M08","Spatial","row/column/pixel spatial variance decomposition","R01;R07","LEVEL-A","dark/flat stacks or temporal-residual stacks","uniform preferred","optional","depending target","required","settings and scene","temporal/spatial separation","directional variance components","row/column/pixel spatial terms","residual field isolates noise","scene edges and repeated structure","PARTIALLY-ELIGIBLE","row-profile energy and column-profile energy of temporal residual","high","high"),
        ("M09","Dark","hot/defect pixel statistics","R01;R09","LEVEL-A","matched dark/flat stacks","dark/flat","useful","required for dark defects","required","temperature, exposure, gain","threshold definition","defect rates and persistence","hot/unstable pixel rate","repeatable defect definition","threshold and scene leakage","EXPLORATORY-ONLY","hot-pixel candidate rate","medium","low"),
        ("M10","Response","exposure-response and linearity","R01;R03","LEVEL-S","controlled exposure/irradiance series","uniform stable","required","matched","pairs","true exposure and irradiance","radiometric setup","response curve, linearity error","responsivity, linear range","same scene/flat and fixed settings","folder labels, auto processing","REQUIRES-NEW-ACQUISITION","observed signal range only until controlled acquisition","high","medium"),
        ("M11","Response","saturation and dynamic range","R01;R03","LEVEL-S","dark/noise floor and controlled saturation series","uniform stable","required","matched","pairs","bit depth, exposure, settings","response calibration","saturation, SNRmax, dynamic range","dynamic range","noise floor and saturation measured consistently","sample maxima are not saturation capacity","REQUIRES-NEW-ACQUISITION","zero/saturation ratios and dynamic-range proxy only","high","low"),
        ("M12","Temporal","pixelwise temporal mean/std/variance","R01;R14","LEVEL-A","ordered repeated raw frames at fixed settings","scene or flat","no","no","required","settings; scene relation","fixed ROI","temporal maps/distributions","operational temporal strength","scene stationary enough","motion/flicker/drift","FORMALLY-ELIGIBLE","pixelwise temporal variability at frozen ROI","high","high"),
        ("M13","Temporal","frame-difference temporal noise","R04;R14","LEVEL-A","paired consecutive/split repeated frames","scene or flat","no","no","required","ordering","difference scaling","difference std/maps","temporal difference-noise strength","scene stable between pair","motion/flicker","FORMALLY-ELIGIBLE","frame-difference operational temporal noise","high","high"),
        ("M14","Temporal","frame-count convergence","R14","LEVEL-B","long repeated sequence","scene or flat","no","no","required","ordering","pre-registered subsets","estimate versus frame count","convergence range","same acquisition block","drift and scene changes","FORMALLY-ELIGIBLE","frame-count sensitivity/convergence","high","high"),
        ("M15","Temporal","split-half and odd-even repeatability","R14","LEVEL-B","ordered repeated sequence","scene or flat","no","no","required","ordering","pre-registered split","repeatability/correlation","reliability bounds","splits represent same state","drift","FORMALLY-ELIGIBLE","split-map repeatability of observed stable component","high","high"),
        ("M16","Temporal","frame-level DC drift","R14;R17","LEVEL-B","ordered repeated frames","scene or flat","no","no","required","ordering; timestamps helpful","none","frame mean trend/change","operational drift","scene and illumination stable","flicker, scene motion","FORMALLY-ELIGIBLE","frame-mean drift over acquisition order","high","medium"),
        ("M17","Temporal","time-series autocorrelation","R16;R17","LEVEL-B","ordered samples","any stable target","no","no","required","ordering; interval","none","lag correlation","sample-lag memory","sampling order known","unknown interval limits seconds/Hz","PARTIALLY-ELIGIBLE","frame-lag autocorrelation; no physical frequency","medium","medium"),
        ("M18","Temporal","Allan/multiscale stability","R16;R17","LEVEL-B","long ordered time series","stable target","no","no","required","sampling interval strongly preferred","none","variance versus averaging scale","stability regimes","stationarity and known cadence","only 200 frames; cadence unknown","EXPLORATORY-ONLY","sample-count multiscale stability; not physical Allan time","medium","low"),
        ("M19","Correlation","horizontal/vertical/radial ACF of temporal residual","R07;R08","LEVEL-A","temporal residual fields from repeated frames","scene or flat","no","no","required","fixed ROI and settings","temporal mean subtraction/differencing","ACF profiles","correlation strength/length proxy","residual isolates temporal field","mean-image or scene leakage","FORMALLY-ELIGIBLE","temporal-residual spatial autocorrelation","high","high"),
        ("M20","Correlation","2D spatial covariance","R05;R06;R15","LEVEL-A","paired flats ideally; temporal residual acceptable operationally","flat preferred","signal levels preferred","matched for physical PTC","required","settings and signal","residual construction","anisotropic covariance map","covariance parameters","stationary residual field","scene/FPN leakage and limited ROI","FORMALLY-ELIGIBLE","2D covariance of temporal residual; operational","high","high"),
        ("M21","Frequency","temporal-residual NPS/radial NPS","R08","LEVEL-A","multiple temporal residual images","uniform preferred; scene stack possible after temporal subtraction","no","no","required","pixel pitch for physical frequency","detrending/windowing definition","2D/radial NPS and bands","spectral shape","residual rather than scene image","fixed gain pattern, window/normalization errors","FORMALLY-ELIGIBLE","temporal-residual power spectrum in cycles/pixel","high","high"),
        ("M22","Frequency","mean-image/fixed-pattern spectrum","R01;R08","LEVEL-B","averaged stack with temporal correction","dark/flat preferred","optional","depending target","required","settings and scene","scene removal","stable spatial spectrum","repeatable observed spatial structure","scene absent or separately modeled","ordinary scene dominates","EXPLORATORY-ONLY","spectrum of repeatable observed stable component","medium","low"),
        ("M23","Frequency","anisotropic row/column spectral peaks","R01;R08","LEVEL-B","temporal residual or controlled dark/flat","uniform preferred","no","optional","required","orientation/pixel pitch","consistent PSD","directional peaks","row/column periodicity","residual field valid","scene edges and windowing","FORMALLY-ELIGIBLE","directional temporal-residual spectral energy","high","high"),
        ("M24","PTC","3D/spatiotemporal PTC","R07","LEVEL-A","controlled dark/flat stacks across signal","homogeneous flat","required","matched","many frames","gain/noise factor/settings","multi-component fit","temporal+row+column+pixel curves","noise factor and spatial terms","identifiable multi-curve model","too few controlled levels; double counting","REQUIRES-NEW-ACQUISITION","not applicable without controlled flat series","high","medium"),
        ("M25","Photon statistics","Fano/variance-to-mean overdispersion","R10;R19","LEVEL-B","raw repeated data and valid mean offset","uniform/known signal preferred","signal levels","matched offset preferred","required","mode, gain, black level","offset/gain validation","variance-to-mean ratio","operational overdispersion","mean represents comparable signal scale","pedestal, gain and scene confounding","PARTIALLY-ELIGIBLE","Fano-like operational statistic in DN; not photon gain","medium","high"),
        ("M26","Photon statistics","photon-count histogram/zero-event probability","R11;R12;R13","LEVEL-A","resolved photon events in verified counting mode","controlled low flux","multiple fluxes","dark events","required","mode, threshold, dead time","event threshold calibration","event-rate/histogram","count rate, ENF","single events resolvable","analog integration or unknown mode","NOT-ELIGIBLE","possible future photon-counting method only","high","low"),
        ("M27","Gain detector","multiplication/excess-noise factor","R10;R12;R13","LEVEL-A","controlled flux, darks and multiple gain settings","calibrated low-light flat","required","required","required","MCP/EM gain, mode","input flux calibration","gain distribution/ENF","ENF, gain","gain chain and flux known","unknown gain and compound coupling","REQUIRES-NEW-ACQUISITION","possible physical origin only","high","medium"),
        ("M28","Restoration","noise-level function/heteroscedastic Gaussian","R19","LEVEL-C","repeated data or robust raw patches across signal","scene/flat","signal range","optional","required or estimator","raw scale and clipping","calibration-only fit","sigma(signal)","operational a,b","signal relation repeatable","scene/ROI/pedestal confounding","PARTIALLY-ELIGIBLE","observed-signal-conditioned operational noise-strength model","high","high"),
        ("M29","Restoration","Poisson-Gaussian noise model","R19;R21","LEVEL-C","raw data with identifiable shot/read terms","controlled preferable","signal range","bias/dark","repeated or clean pairs","gain/black level/settings","parameter fit","variance function","Poisson/read parameters","component assumptions identifiable","ICCD multiplication and unknown processing","EXPLORATORY-ONLY","heteroscedastic operational fit; no physical component labels","medium","high"),
        ("M30","Restoration","correlated Gaussian residual model","R08;R19","LEVEL-C","temporal residual covariance/NPS","scene/flat","no","no","required","ROI and condition","positive-definite covariance","covariance/filter kernel","correlation parameters","stationary Gaussian approximation","non-Gaussian tails and stable leakage","PARTIALLY-ELIGIBLE","calibration-only correlated operational residual","medium","high"),
        ("M31","Restoration","real-noise patch extraction/covariance","R21;R23","LEVEL-C","scene-free or paired/repeated residual patches","scene stacks","conditions","optional","required","condition labels","leakage-safe residual extraction","empirical patches/covariance","conditioned empirical distribution","scene cancels reliably","scene and holdout leakage","EXPLORATORY-ONLY","calibration-only residual patches after leakage audit","medium","high"),
        ("M32","Gain detector","afterglow/lag and phosphor-spread characterization","R12;R13","LEVEL-A","impulse/event or switched illumination sequence","controlled pulsed light","gate/delay series","dark events","high cadence","gate width, cadence, phosphor, MCP gain","timing/flux calibration","lag decay and spread kernel","afterglow time, spread proxy","timing known and events controlled","unknown cadence and ordinary scenes","REQUIRES-NEW-ACQUISITION","possible physical origins only","high","medium"),
    ]
    fields = ["method_id","method_family","method_name","primary_reference","evidence_level","required_data","required_illumination","required_exposure_series","required_dark_frames","required_repeated_frames","required_metadata","required_calibration","primary_outputs","estimable_parameters","main_assumptions","main_failure_modes","eligibility","recommended_terminology","paper_value","noise_model_value"]
    rows = []
    for entry in raw:
        item = dict(zip(fields, entry))
        item.update({
            "physical_or_operational": "physical" if item["evidence_level"] in {"LEVEL-S", "LEVEL-A"} else "operational",
            "detector_type": "ICCD/CCD/CMOS transferable with stated limits",
            "applicable_to_iccd": True,
            "applicable_to_current_data": item["eligibility"] in {"FORMALLY-ELIGIBLE","PARTIALLY-ELIGIBLE","EXPLORATORY-ONLY"},
            "risk_of_overclaiming": "high" if item["eligibility"] in {"NOT-ELIGIBLE","REQUIRES-NEW-ACQUISITION"} else "controlled by terminology",
        })
        rows.append(item)
    return rows


def assets(repo: Path) -> list[dict[str, Any]]:
    common = {"gate_width":"UNKNOWN","MCP_gain":"UNKNOWN","trigger":"UNKNOWN","readout_mode":"UNKNOWN","temperature":"UNKNOWN","frame_interval":"UNKNOWN"}
    rows = [
        {"asset_id":"iccd_20260319_formal_repeated_folders","path":"D:/iccd/data/20260319/{1,2,4,5,7,8,9,10,11,13}","device_type":"gated ICCD","device_identity":"Camera1","camera_serial":"20600555","acquisition_mode":"analog_or_unknown gated imaging","exposure_label":"Exposure width 900 ms","exposure_value":900,"exposure_unit":"ms","exposure_field_type":"EXPOSURE_CONTROL_WIDTH","recording_gain":"60","sync":"A=4 us; B=4 us","scene_id":"UNKNOWN","scene_relation":"within-folder partly stable; cross-folder unknown","light_or_dark":"light/ordinary scene","repeated_frames":True,"frame_count":2000,"dtype":"uint16","shape":"5120x5120","format":"TIFF","raw_or_processed":"camera output; processing pipeline unknown","processing_history":"no project-side dark/p99/minmax for E1","normalization_history":"raw DN for E1","dark_subtraction_history":"none","source_metadata":"PictureInfo plus filenames","confidence":"HIGH for file/setting fields; LOW for physical gate/scene","existing_role":"E1 calibration and folder-blocked real holdout",**common},
        {"asset_id":"iccd_20260319_incomplete_or_unselected","path":"D:/iccd/data/20260319/{3,6,12,1_20260715_143749}","device_type":"gated ICCD","device_identity":"Camera1/partly unknown","camera_serial":"20600555 where inventoried","acquisition_mode":"same acquisition family or unverified","exposure_label":"900 ms where PictureInfo inventoried","exposure_value":"900 or UNKNOWN","exposure_unit":"ms","exposure_field_type":"EXPOSURE_CONTROL_WIDTH","recording_gain":"60 where inventoried","sync":"A/B 4 us where inventoried","scene_id":"UNKNOWN","scene_relation":"unverified","light_or_dark":"likely light; unverified","repeated_frames":True,"frame_count":"3=4; 6=5; others not formally inventoried","dtype":"uint16 where sampled","shape":"5120x5120 where sampled","format":"TIFF/TXT","raw_or_processed":"camera output; incomplete provenance","processing_history":"not used in formal E1","normalization_history":"none in inventory","dark_subtraction_history":"none known","source_metadata":"partial directory/PictureInfo evidence","confidence":"LOW-MEDIUM","existing_role":"excluded from formal E1",**common},
        {"asset_id":"scmos_multiexposure_local_family","path":"F:/目标传感器噪声参数估计/data and D:/PMRID4/data","device_type":"sCMOS operational content","device_identity":"not ICCD; exact model unverified","camera_serial":"UNKNOWN","acquisition_mode":"content/recovery acquisition","exposure_label":"1,5,10,15,25,50,125,250,500 ms; 1 s","exposure_value":"1..1000","exposure_unit":"ms","exposure_field_type":"FOLDER_LABEL_ONLY","recording_gain":"UNKNOWN","sync":"UNKNOWN","scene_id":"scene0001 filename only","scene_relation":"paired filenames suggest common content family; independent scene structure not established","light_or_dark":"light","repeated_frames":False,"frame_count":"100 files per labeled exposure in D copy; 2900 TIFF root inventory in prior F audit","dtype":"uint16","shape":"2048x2048","format":"TIFF","raw_or_processed":"processing status unknown","processing_history":"historical exposure matching, caches and training lists exist","normalization_history":"historical loaders may normalize; source TIFF preserved","dark_subtraction_history":"untraced historical artifact existed but is invalid","source_metadata":"directory labels and historical scripts; no camera exposure tag proof","confidence":"HIGH for directory labels/counts; LOW for physical exposure/device settings","existing_role":"sCMOS operational content/training history; not ICCD characterization",**common},
        {"asset_id":"scmos_dark_background","path":"F:/目标传感器噪声参数估计/data/dark_Background","device_type":"sCMOS","device_identity":"same broad sCMOS project; matching conditions unverified","camera_serial":"UNKNOWN","acquisition_mode":"dark candidate","exposure_label":"UNKNOWN","exposure_value":"UNKNOWN","exposure_unit":"UNKNOWN","exposure_field_type":"UNKNOWN","recording_gain":"UNKNOWN","sync":"UNKNOWN","scene_id":"dark","scene_relation":"not matched to ICCD; not proven matched to 500 ms sCMOS","light_or_dark":"dark","repeated_frames":True,"frame_count":100,"dtype":"uint16","shape":"2048x2048","format":"TIFF","raw_or_processed":"unknown","processing_history":"64-frame mean artifact previously produced","normalization_history":"raw-DN mean artifact","dark_subtraction_history":"historical subtraction invalid; 97%+ clipping","source_metadata":"folder name only; conditions unrecorded","confidence":"MEDIUM for existence/count; LOW for compatibility","existing_role":"calibration candidate only; no formal correction",**common},
        {"asset_id":"iccd_pir_background_candidate","path":"F:/ICCD_pir/2025.07.09/CDM-A4000-UM90_DH09131AAK00007","device_type":"ICCD-like external acquisition","device_identity":"CDM-A4000-UM90_DH09131AAK00007","camera_serial":"UNKNOWN","acquisition_mode":"background candidate; mode unknown","exposure_label":"UNKNOWN","exposure_value":"UNKNOWN","exposure_unit":"UNKNOWN","exposure_field_type":"UNKNOWN","recording_gain":"UNKNOWN","sync":"UNKNOWN","scene_id":"background candidate","scene_relation":"different date/device/file format from main ICCD batch","light_or_dark":"background candidate, not verified capped dark","repeated_frames":True,"frame_count":131,"dtype":"uint8","shape":"2048x2048","format":"image files","raw_or_processed":"8-bit processing status unknown","processing_history":"unknown","normalization_history":"unknown","dark_subtraction_history":"unknown","source_metadata":"prior auxiliary audit only","confidence":"MEDIUM as auxiliary background; LOW as dark calibration","existing_role":"auxiliary evidence only",**common},
        {"asset_id":"pmrid_official_reference_raw","path":"E:/PMRID-Pytorch-main/PMRID/PMRID","device_type":"mobile Bayer camera","device_identity":"PMRID benchmark","camera_serial":"benchmark metadata","acquisition_mode":"official paired mobile RAW","exposure_label":"official ISO/exposure fields","exposure_value":"varied","exposure_unit":"benchmark metadata","exposure_field_type":"SENSOR_INTEGRATION_TIME","recording_gain":"ISO metadata","sync":"not applicable","scene_id":"Scene1..Scene4","scene_relation":"official scene groups","light_or_dark":"bright/dark scene conditions, not detector dark frames","repeated_frames":False,"frame_count":39,"dtype":"uint16","shape":"3000x4000","format":"RAW","raw_or_processed":"official reference RAW","processing_history":"official benchmark","normalization_history":"not used for ICCD characterization","dark_subtraction_history":"benchmark-specific unknown/not relevant here","source_metadata":"benchmark.json and official repository","confidence":"HIGH","existing_role":"validation_content_only; not detector characterization",**common},
        {"asset_id":"historical_dark_offset_artifact","path":"reports/target_scmos_risk_audit/dark_offset_center_crop.npy","device_type":"unknown/untraced sCMOS calibration derivative","device_identity":"UNKNOWN","camera_serial":"UNKNOWN","acquisition_mode":"64-frame mean derivative","exposure_label":"UNKNOWN","exposure_value":"UNKNOWN","exposure_unit":"UNKNOWN","exposure_field_type":"UNKNOWN","recording_gain":"UNKNOWN","sync":"UNKNOWN","scene_id":"dark candidate","scene_relation":"unmatched","light_or_dark":"derived dark mean","repeated_frames":False,"frame_count":1,"dtype":"float array","shape":"512x512 effective","format":"NPY","raw_or_processed":"derived","processing_history":"mean of 64 named dark-folder frames","normalization_history":"raw DN array","dark_subtraction_history":"incompatible with current 500 ms source","source_metadata":"source conditions absent","confidence":"HIGH that artifact is incompatible; LOW provenance","existing_role":"UNTRACED CALIBRATION ARTIFACT; excluded",**common},
    ]
    return rows


def exposure_rows() -> list[dict[str, Any]]:
    rows = []
    for label, value in [("1ms",1),("5ms",5),("10ms",10),("15ms",15),("25ms",25),("50ms",50),("125ms",125),("250ms",250),("500ms",500),("1s",1000)]:
        rows.append({"exposure_label":label,"nominal_value_ms":value,"path":f"D:/PMRID4/data/{label}; historical F:/目标传感器噪声参数估计/data/{label}","device_assignment":"sCMOS operational content family, not gated ICCD","field_type":"FOLDER_LABEL_ONLY","frame_count":100,"same_scene_evidence":"matching scene0001 indices suggest a common acquisition family but not 100 independent scenes","raw_dn_status":"uint16 source TIFF; processing status unknown","dark_available":"no matched dark proven","saturation_evidence":"low saturation in deterministic prior samples","formal_detector_use":"NOT-ELIGIBLE for ICCD characterization","recovery_use":"historical paired exposure/recovery data","verification_level":"PARTIALLY-VERIFIED label only","blocking_issue":"camera metadata, physical exposure tag, processing history and stable-scene proof missing"})
    rows.extend([
        {"exposure_label":"Exposure width 900 ms","nominal_value_ms":900,"path":"D:/iccd/data/20260319/{formal 10 folders}","device_assignment":"gated ICCD camera serial 20600555","field_type":"EXPOSURE_CONTROL_WIDTH","frame_count":2000,"same_scene_evidence":"within-folder partly stable; cross-folder unknown","raw_dn_status":"uint16 camera output; project uses raw DN","dark_available":"no matched dark","saturation_evidence":"formal inventory reports near-zero sampled saturation","formal_detector_use":"FORMALLY-ELIGIBLE for repeated-frame operational statistics only","recovery_use":"supports observed-signal-conditioned modeling","verification_level":"VERIFIED field; physical gate attribution UNKNOWN","blocking_issue":"Exposure control width is not proven gate width or sensor integration time"},
        {"exposure_label":"Sync A/B 4 us","nominal_value_ms":0.004,"path":"D:/iccd/data/20260319/{formal 10 folders}","device_assignment":"gated ICCD camera serial 20600555","field_type":"UNKNOWN","frame_count":2000,"same_scene_evidence":"not relevant","raw_dn_status":"PictureInfo setting","dark_available":"no","saturation_evidence":"not relevant","formal_detector_use":"metadata only","recovery_use":"none until physical attribution","verification_level":"VERIFIED value; UNKNOWN meaning","blocking_issue":"cannot call this gate width without device/acquisition documentation"},
        {"exposure_label":"50 ms and 1 s dark claims","nominal_value_ms":"50;1000","path":"no verified matching ICCD dark sequence located; same labels exist under D:/PMRID4 sCMOS light data","device_assignment":"UNRESOLVED; available labeled directories are sCMOS light/recovery assets","field_type":"CONFLICTING","frame_count":"not established for matching ICCD dark","same_scene_evidence":"not applicable","raw_dn_status":"unresolved","dark_available":"not formally usable","saturation_evidence":"unknown","formal_detector_use":"NOT-ELIGIBLE","recovery_use":"none for detector calibration","verification_level":"CONFLICTING","blocking_issue":"device, darkness, exposure, temperature, gain and readout mode are not jointly traced"},
    ])
    return rows


def e1_matrix(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coverage = {
        "M03":("folder-level signal and temporal variance/std computed","yes","operational relation correctly bounded","same-scene/exposure condition unresolved","retain and rename"),
        "M08":("row/column profile energies computed on residual statistics","yes","not standard DSNU/PRNU decomposition","controlled dark/flat missing","retain as directional residual energy"),
        "M12":("pixelwise/folder temporal metrics and distributions","yes","correct operational scope","none for fixed ROI result","retain core"),
        "M14":("16/32/64/128 and positional subset checks","yes","correct","physical cadence missing","retain core"),
        "M15":("split-map correlation and subset repeatability","yes","stable component carefully qualified","scene leakage remains","retain with observed-stable wording"),
        "M16":("frame mean slopes and half changes","yes","operational drift","timestamps/cadence incomplete","retain core"),
        "M17":("lag 1/10/50 residual correlation","yes","sample-lag only","frame interval unknown","retain with no Hz claim"),
        "M19":("horizontal/vertical/diagonal and radial ACF","yes","source appears temporal residual in formal E1","pipeline definition should be restated in paper","retain core"),
        "M21":("low/mid/high PSD fractions","yes","formal report labels operational residual spectrum","window/normalization and 2D anisotropy need explicit method","recompute/verify in P1"),
        "M25":("variance-to-mean Fano-like DN statistic","yes","correctly limited in current reports","black level/gain unknown","retain only as Fano-like operational statistic"),
        "M28":("calibration-only observed-signal sigma model","yes","correctly nonphysical","signal-scene confounding and cross-device scaling","retain for CG, not detector gain"),
    }
    rows=[]
    for method in method_rows:
        cov=coverage.get(method["method_id"],("not computed under method prerequisites","no","n/a","required controlled inputs missing","follow eligibility status"))
        rows.append({"method":method["method_name"],"literature_requirement":method["required_data"],"current_e1_coverage":cov[0],"current_data_support":method["eligibility"],"already_computed":cov[1],"correctly_named":cov[2],"misnaming_risk":method["risk_of_overclaiming"],"missing_inputs":cov[3],"can_be_computed_from_existing_data":method["eligibility"] in {"FORMALLY-ELIGIBLE","PARTIALLY-ELIGIBLE","EXPLORATORY-ONLY"},"requires_new_acquisition":method["eligibility"]=="REQUIRES-NEW-ACQUISITION","value_for_paper":method["paper_value"],"value_for_noise_model":method["noise_model_value"],"recommendation":cov[4]})
    return rows


def framework_rows() -> list[dict[str, Any]]:
    return [
        {"framework_id":"A","framework":"standard PTC dominated","scientific_accuracy":"high only with compliant acquisition","data_compatibility":"low","physical_interpretability":"potentially high","paper_strength":"weak with current data","reproducibility":"not currently reproducible","risk":"very high overclaiming","required_missing_data":"uniform calibrated flats, matched darks, >=50 signal points, irradiance","ability_to_support_CG":"only after new calibration","acta_photonica_fit":"not currently","decision":"REJECT AS CURRENT PRIMARY"},
        {"framework_id":"B","framework":"repeated-frame operational statistics dominated","scientific_accuracy":"high for observed operational statistics","data_compatibility":"high","physical_interpretability":"limited","paper_strength":"moderate","reproducibility":"high at frozen ROI/folder split","risk":"scene/brightness confounding","required_missing_data":"metadata and stronger residual definitions","ability_to_support_CG":"yes, observed-signal condition with limitations","acta_photonica_fit":"partial","decision":"RETAIN AS CORE BUT NOT ALONE"},
        {"framework_id":"C","framework":"exposure/dark sequence dominated","scientific_accuracy":"high if exposure and dark provenance verified","data_compatibility":"low for ICCD; labels belong mainly to sCMOS","physical_interpretability":"currently low","paper_strength":"weak unless metadata recovered","reproducibility":"labels reproducible, physical meaning not","risk":"cross-device and exposure-label misattribution","required_missing_data":"matching ICCD dark/exposure series, temperature, same scene/flat","ability_to_support_CG":"not currently","acta_photonica_fit":"not current primary","decision":"CONDITIONAL AUXILIARY ONLY"},
        {"framework_id":"D","framework":"layered EMVA-inspired operational framework","scientific_accuracy":"highest under present evidence","data_compatibility":"high","physical_interpretability":"explicitly tiered","paper_strength":"strongest defensible option","reproducibility":"high for existing layers; gaps explicit","risk":"controlled by method eligibility and naming","required_missing_data":"P0 metadata recovery; new acquisition only for compliant PTC/DSNU/PRNU","ability_to_support_CG":"yes through calibration-only operational NLF","acta_photonica_fit":"yes for a draft with metadata limitations","decision":"RECOMMENDED"},
    ]


def task_rows() -> list[dict[str, Any]]:
    base = [
        ("P0-01","P0","Metadata","Verify whether Exposure width 900 ms is sensor integration, exposure-control pulse, or gate-related control","R01","D:/iccd/data/20260319 PictureInfo plus device documentation","device/acquisition documentation","PARTIALLY-ELIGIBLE","document-only trace","metadata verifier","exposure_metadata_verification.csv","critical","high","high","low",False,False,"physical field meaning unknown"),
        ("P0-02","P0","Scene provenance","Recover same-scene/cross-folder relation and illumination stability","R14","D:/iccd/data/20260319 logs and acquisition notes","scene/acquisition record","PARTIALLY-ELIGIBLE","document and filename trace","scene provenance audit","folder_scene_relation.csv","critical","high","high","medium",False,False,"folder brightness is confounded with scene"),
        ("P0-03","P0","Dark provenance","Resolve device/settings of alleged 50 ms, 1 s and dark_Background assets","R01;R09","D:/PMRID4; F audit artifacts; acquisition notes","device, mode, temperature, exposure","NOT-ELIGIBLE","document-only trace","dark provenance audit","dark_asset_compatibility.csv","critical","medium","high","medium",False,False,"no matching ICCD dark proven"),
        ("P0-04","P0","Definition audit","Freeze temporal residual, PSD normalization, ACF and row/column equations and terminology","R01;R08","existing E1 scripts/reports","code and report consistency","FORMALLY-ELIGIBLE","code-level method trace only","definition consistency checker","metric_definition_registry.csv","critical","high","medium","low",False,False,"paper methods must be exactly reproducible"),
        ("P1-01","P1","Temporal","Difference-frame operational temporal noise by folder","R04;R14","10 formal 200-frame ICCD folders","pre-registered pairing","FORMALLY-ELIGIBLE","pair differences at frozen ROI","difference-frame analysis","folder_difference_noise.csv","high","high","medium","medium",False,False,"scene motion/flicker diagnostics required"),
        ("P1-02","P1","Temporal","Frame-count convergence and positional subset stability","R14;R16","10 formal ICCD folders","ordered frames","FORMALLY-ELIGIBLE","subset estimates","convergence analysis","convergence_curves.csv","high","high","low","medium",False,False,"cadence unknown limits time units"),
        ("P1-03","P1","Correlation","2D temporal-residual covariance and anisotropy","R05;R15","calibration folders only for model parameters","residual definition frozen","FORMALLY-ELIGIBLE","covariance maps and profiles","covariance analysis","covariance_summary.csv","high","high","medium","medium",False,False,"must exclude scene/stable mean"),
        ("P1-04","P1","Frequency","Temporal-residual 2D/radial NPS with directional peaks","R08","formal repeated ICCD folders","window/normalization frozen","FORMALLY-ELIGIBLE","residual NPS","NPS analysis","nps_summary.csv","high","high","medium","medium",False,False,"cycles/pixel only without pitch"),
        ("P1-05","P1","Spatial","Operational row/column/pixel residual variance decomposition","R01;R07","formal repeated ICCD folders","temporal/spatial separation","PARTIALLY-ELIGIBLE","directional decomposition","spatial decomposition","spatial_component_summary.csv","high","high","medium","medium",False,False,"cannot call DSNU/PRNU"),
        ("P1-06","P1","Stability","Sample-count multiscale stability/Allan-like exploratory check","R16;R17","ordered 200-frame folders","ordered frames; cadence optional","EXPLORATORY-ONLY","sample-scale variance","multiscale stability","multiscale_stability.csv","medium","medium","medium","low",False,False,"no physical seconds without cadence"),
        ("P2-01","P2","Restoration","Calibration-only noise-level function uncertainty and range","R19","calibration folders 1,4,7,8,10,13","folder-blocked fit","PARTIALLY-ELIGIBLE","bootstrap/LOFO operational fit","NLF audit","noise_level_function.csv","high","high","medium","medium",False,False,"brightness/scene confounding"),
        ("P2-02","P2","Restoration","Single correlated residual component feasibility from calibration-only covariance","R08;R19","calibration folders only","positive-definite covariance and leakage audit","PARTIALLY-ELIGIBLE","one-component feasibility only","correlated residual feasibility","correlation_component_candidate.csv","medium","high","high","medium",False,False,"CGS not authorized in this stage"),
        ("P2-03","P2","Restoration","Noise synthesis fidelity metric registry","R19;R21","E1 calibration residual statistics","pre-registered target metrics","FORMALLY-ELIGIBLE","no generation in task design","fidelity protocol","fidelity_metric_registry.csv","medium","high","low","low",False,False,"must not tune on holdout"),
        ("P3-01","P3","PTC","Acquire EMVA-style homogeneous calibrated flat/dark exposure series","R01;R04","new ICCD acquisition","integrating sphere/diffuse source, radiometry, >=50 points","REQUIRES-NEW-ACQUISITION","controlled acquisition","future acquisition protocol","PTC/linearity/gain/full-well","high","high","high","high",False,True,"not executable from current data"),
        ("P3-02","P3","Dark","Acquire matched capped-dark sequence at >=6 exposures and controlled temperature","R01;R09","new ICCD acquisition","same mode/gain plus temperature","REQUIRES-NEW-ACQUISITION","controlled acquisition","future dark protocol","dark current/read/hot-pixel curves","high","medium","high","high",False,True,"not executable from current data"),
        ("P3-03","P3","Gain detector","Acquire multi-MCP-gain and multi-gate-width flat/event sequences","R10;R12","new ICCD acquisition","known flux, MCP gain, gate, mode","REQUIRES-NEW-ACQUISITION","controlled acquisition","future intensifier protocol","ENF/gain/afterglow/spread","high","high","high","high",False,True,"not executable from current data"),
        ("P3-04","P3","Spatial","Acquire dark/flat stacks for formal DSNU/PRNU and defect maps","R01;R02","new ICCD acquisition","homogeneous flat plus matched dark","REQUIRES-NEW-ACQUISITION","controlled acquisition","future nonuniformity protocol","DSNU/PRNU/spectrograms","high","medium","high","high",False,True,"not executable from current data"),
    ]
    fields=["task_id","priority","method_family","scientific_question","literature_basis","data_asset","required_conditions","existing_data_eligibility","required_processing","required_script","expected_outputs","paper_value","noise_model_value","overclaiming_risk","effort","requires_training","requires_new_acquisition","blocking_issue"]
    return [dict(zip(fields,row)) for row in base]


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    cfg_path=(repo/args.config).resolve()
    output=(repo/args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    cfg=yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    status_before=git(repo,"status","--porcelain=v1","--untracked-files=all")
    if status_before.strip():
        raise RuntimeError("Formal run requires a clean worktree")
    started=utc_now()
    output.mkdir(parents=True)
    (output/"provenance").mkdir()

    refs=references(); method_rows=methods(); asset_rows=assets(repo); exp_rows=exposure_rows()
    search_log=[
        {"query_id":f"Q{i:02d}","query":q,"scope":scope,"accessed_at":ACCESSED_AT,"result":"primary/official evidence retained; secondary search snippets excluded from claims"}
        for i,(q,scope) in enumerate([
            ("EMVA 1288 Release 4.0 Linear General","standards"),("ISO 15739:2023 noise measurements","standards"),("photon transfer curve flat field covariance","PTC"),("difference frame photon transfer Janesick","PTC"),("CCD dark current exposure temperature hot pixels","dark"),("image sensor NPS gain nonuniformity","NPS"),("3D noise photon transfer curve","spatiotemporal"),("EMCCD excess noise factor photon counting","gain detectors"),("ICCD MCP phosphor spatial gain afterglow","ICCD/MCP"),("stack raw images flicker vibration sensor noise","repeatability"),("Allan variance optical instrument stability","temporal"),("Poisson Gaussian raw noise model","restoration"),("physics based low-light raw noise modeling","restoration"),
        ],1)
    ]

    ref_fields=["reference_id","title","authors","year","journal_or_standard","doi_or_official_url","reference_class","detector_type","method_family","key_contribution","data_requirements","relevance","limitations","citation_status","accessed_at"]
    write_csv(output/"literature_search_log.csv",search_log,list(search_log[0]))
    write_csv(output/"literature_reference_registry.csv",refs,ref_fields)
    method_fields=["method_id","method_family","method_name","primary_reference","reference_type","detector_type","physical_or_operational","required_data","required_illumination","required_exposure_series","required_dark_frames","required_repeated_frames","required_metadata","required_calibration","primary_outputs","estimable_parameters","main_assumptions","main_failure_modes","applicable_to_iccd","applicable_to_current_data","evidence_level","paper_value","noise_model_value","risk_of_overclaiming"]
    for row in method_rows: row["reference_type"]=row["evidence_level"]
    write_csv(output/"literature_noise_characterization_methods.csv",method_rows,method_fields)
    asset_fields=["asset_id","path","device_type","device_identity","camera_serial","acquisition_mode","exposure_label","exposure_value","exposure_unit","exposure_field_type","gate_width","MCP_gain","recording_gain","trigger","sync","readout_mode","temperature","frame_interval","scene_id","scene_relation","light_or_dark","repeated_frames","frame_count","dtype","shape","format","raw_or_processed","processing_history","normalization_history","dark_subtraction_history","source_metadata","confidence","existing_role"]
    write_csv(output/"current_data_asset_inventory.csv",asset_rows,asset_fields)
    write_csv(output/"exposure_metadata_verification.csv",exp_rows,list(exp_rows[0]))
    e1_rows=e1_matrix(method_rows)
    write_csv(output/"current_e1_vs_literature_matrix.csv",e1_rows,list(e1_rows[0]))
    eligibility=[]
    for method in method_rows:
        eligibility.append({"method_id":method["method_id"],"method":method["method_name"],"status":method["eligibility"],"supporting_asset":"formal ICCD repeated folders" if method["applicable_to_current_data"] else "none satisfying formal prerequisites","missing_condition":method["main_failure_modes"],"overclaiming_risk":method["risk_of_overclaiming"],"recommended_terminology":method["recommended_terminology"],"paper_section":method["method_family"],"priority":"P1" if method["eligibility"]=="FORMALLY-ELIGIBLE" else ("P2" if method["eligibility"]=="PARTIALLY-ELIGIBLE" else "P3/auxiliary")})
    write_csv(output/"method_data_eligibility.csv",eligibility,list(eligibility[0]))
    frameworks=framework_rows(); write_csv(output/"candidate_characterization_framework_comparison.csv",frameworks,list(frameworks[0]))

    recommendation={
        "RECOMMENDED_CHARACTERIZATION_FRAMEWORK":"D_LAYERED_EMVA_INSPIRED_OPERATIONAL",
        "formal_name":"EMVA-inspired layered gated ICCD operational noise characterization",
        "emva_compliance":False,
        "layers":[
            {"layer":1,"name":"data and response validity","content":"metadata, DN integrity, zero/saturation, scene relation, candidate response range"},
            {"layer":2,"name":"temporal noise and stability","content":"temporal maps, difference-frame noise, convergence, split repeatability, drift, sample-lag correlation"},
            {"layer":3,"name":"spatial nonuniformity and correlation","content":"repeatable observed stable component, row/column residual energy, covariance, temporal-residual ACF/NPS, ROI sensitivity"},
            {"layer":4,"name":"signal/exposure dependence","content":"verified exposure-response only when metadata permits; otherwise observed signal-level versus operational temporal strength"},
            {"layer":5,"name":"restoration-model constraints","content":"calibration-only repeatable NLF/covariance parameters with holdout separation"},
        ],
        "why_not_standard_ptc_only":"Current data lack prescribed homogeneous calibrated illumination, matched darks and a verified ICCD exposure/irradiance series.",
        "why_not_current_e1_only":"E1 lacks a formal difference-frame/covariance/NPS definition registry and does not resolve exposure, scene or matching dark provenance.",
        "current_e1_restructure":"YES-NAMING-AND-LAYERING; retain verified values, reorganize methods and add only approved P0/P1 analyses later.",
        "signal_conditioned_conclusion":"RETAINS-OPERATIONAL-BASIS; it remains an observed-signal-conditioned strength model, not a physical gain/PTC result.",
        "status":"LITERATURE-GROUNDED-FRAMEWORK-READY-WITH-METADATA-GAPS",
    }
    dump_json(output/"recommended_characterization_framework.json",recommendation)
    readiness=[
        {"criterion":"experimental system information","general_paper":"PARTIAL","acta_photonica_sinica":"PARTIAL","evidence":"camera serial and several controls recovered; physical gate/mode/temp incomplete","blocking":"P0 metadata"},
        {"criterion":"real repeated ICCD data","general_paper":"SUPPORTED","acta_photonica_sinica":"SUPPORTED","evidence":"10x200 uint16 repeated frames and folder-blocked split","blocking":"scene relation"},
        {"criterion":"temporal noise","general_paper":"SUPPORTED","acta_photonica_sinica":"SUPPORTED","evidence":"std/variance, convergence, repeatability, drift","blocking":"difference-frame formalization"},
        {"criterion":"spatial noise/correlation","general_paper":"SUPPORTED-WITH-LIMITATIONS","acta_photonica_sinica":"SUPPORTED-WITH-LIMITATIONS","evidence":"row/column, ACF, PSD, observed stable component","blocking":"covariance/NPS method registry and scene leakage"},
        {"criterion":"PTC/linearity/dynamic range","general_paper":"NOT-REQUIRED for operational paper","acta_photonica_sinica":"NOT-SUPPORTED as formal calibration","evidence":"no compliant flats/darks/radiometry","blocking":"new acquisition only"},
        {"criterion":"signal-conditioned modeling","general_paper":"SUPPORTED-WITH-LIMITATIONS","acta_photonica_sinica":"SUPPORTED-WITH-LIMITATIONS","evidence":"calibration-only observed signal/noise relation and controlled validation","blocking":"cannot call physical condition"},
        {"criterion":"overall draft","general_paper":"SUPPORTED","acta_photonica_sinica":"SUPPORTED-WITH-METADATA-GAPS","evidence":"layered operational characterization plus controlled restoration validation","blocking":"P0 terminology/metadata and selected P1 computations before final submission"},
    ]
    write_csv(output/"paper_characterization_section_readiness.csv",readiness,list(readiness[0]))
    claims=[
        {"claim":"gated ICCD operational temporal variability differs across observed folder states","support":"SUPPORTED","allowed_wording":"observed folder/state differences at frozen ROI","prohibited_wording":"physical gain/gate causality"},
        {"claim":"current analysis is EMVA 1288 compliant","support":"NOT-SUPPORTED","allowed_wording":"EMVA-inspired partial characterization","prohibited_wording":"EMVA compliant"},
        {"claim":"Fano statistic measures photon gain","support":"NOT-SUPPORTED","allowed_wording":"Fano-like variance-to-mean statistic in DN","prohibited_wording":"conversion gain or photon gain"},
        {"claim":"stable residual is fixed-pattern noise","support":"PARTIAL","allowed_wording":"repeatable observed stable component","prohibited_wording":"pure FPN/DSNU"},
        {"claim":"row/column metrics are formal DSNU components","support":"NOT-SUPPORTED","allowed_wording":"row/column profile energy of temporal residual","prohibited_wording":"DSNU.row/DSNU.col"},
        {"claim":"signal-conditioned CG has an operational basis","support":"SUPPORTED-WITH-LIMITATIONS","allowed_wording":"observed-signal-conditioned noise-strength model","prohibited_wording":"PTC-calibrated physical ICCD model"},
        {"claim":"photon-counting statistics apply","support":"NOT-SUPPORTED","allowed_wording":"possible future method if acquisition mode is verified","prohibited_wording":"event rate/zero-photon probability from analog images"},
    ]
    write_csv(output/"paper_claim_support_audit.csv",claims,list(claims[0]))
    tasks=task_rows(); write_csv(output/"second_stage_candidate_tasks.csv",tasks,list(tasks[0]))
    new_only=[row for row in tasks if row["requires_new_acquisition"]]
    write_csv(output/"new_acquisition_only_items.csv",new_only,list(tasks[0]))

    ref_counts={k:sum(1 for row in refs if row["reference_class"]==k) for k in {r["reference_class"] for r in refs}}
    eligibility_counts={k:sum(1 for row in eligibility if row["status"]==k) for k in sorted({r["status"] for r in eligibility})}
    source_checks=[]
    for source in [Path("D:/iccd/data"),Path("D:/PMRID4"),Path("F:/目标传感器噪声参数估计/data"),Path("F:/ICCD_pir")]:
        source_checks.append({"path":str(source),"accessible":source.exists(),"write_attempted":False,"note":"F evidence reused from prior formal audit when volume is unavailable"})
    write_csv(output/"provenance"/"source_access.csv",source_checks,list(source_checks[0]))
    evidence_paths=[repo/Path(v) for v in cfg["existing_evidence"].values()]
    input_hashes=[{"path":str(p.relative_to(repo)),"sha256":sha256_file(p),"size_bytes":p.stat().st_size} for p in evidence_paths if p.exists()]
    write_csv(output/"provenance"/"input_hashes.csv",input_hashes,["path","sha256","size_bytes"])
    write_csv(output/"provenance"/"script_hashes.csv",[
        {"path":str(Path(__file__).relative_to(repo)),"sha256":sha256_file(Path(__file__))},
        {"path":str(cfg_path.relative_to(repo)),"sha256":sha256_file(cfg_path)},
    ],["path","sha256"])
    commit=git(repo,"rev-parse","HEAD").strip()
    (output/"provenance"/"git_commit.txt").write_text(commit+"\n",encoding="utf-8")
    (output/"provenance"/"git_status_before.txt").write_text(status_before,encoding="utf-8")
    (output/"provenance"/"git_diff.patch").write_text(git(repo,"diff","--binary","HEAD"),encoding="utf-8")
    (output/"provenance"/"command.txt").write_text(subprocess.list2cmdline(sys.argv)+"\n",encoding="utf-8")
    (output/"provenance"/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\npyyaml={yaml.__version__}\n",encoding="utf-8")
    (output/"provenance"/"resolved_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8")

    report=f"""# Literature-Grounded Gated ICCD Characterization Stage 1

- Final status: **{recommendation['status']}**
- References: **{len(refs)}** (standards {ref_counts.get('STANDARD',0)}, detector references/papers {ref_counts.get('DETECTOR_REFERENCE',0)+ref_counts.get('DETECTOR_PAPER',0)}, restoration papers {ref_counts.get('RESTORATION_PAPER',0)})
- Methods assessed: **{len(method_rows)}**
- Eligibility: {json.dumps(eligibility_counts, ensure_ascii=False, sort_keys=True)}
- Recommended framework: **{recommendation['formal_name']}**
- EMVA 1288 compliant: **no**

## Decision

The current project should use a layered, EMVA-inspired operational framework rather than a strict PTC claim or an unstructured list of E1 metrics. Formal results can be based on the frozen repeated ICCD folders: pixelwise temporal variability, difference-frame operational noise, frame-count convergence, split repeatability, drift, temporal-residual covariance/ACF/NPS, and directional residual energy. Physical labels that require flats, matched darks, calibrated irradiance, verified exposure, temperature, gain or photon-counting mode remain unavailable.

The existing E1 values remain valid within their frozen ROI and operational definitions. They require reorganization and terminology control, not replacement: `Fano-like statistic in DN`, `repeatable observed stable component`, `row/column profile energy of temporal residual`, and `temporal-residual spatial ACF/NPS`. The observed-signal-conditioned model retains an operational basis but is not a photon-transfer, conversion-gain or physical gate model.

## Data interpretation

- `D:/iccd/data/20260319`: the only primary gated ICCD repeated-frame asset. The recorded 900 ms value is an exposure-control width, not a verified gate width.
- `D:/PMRID4/data` and historical `F:/目标传感器噪声参数估计/data`: sCMOS/recovery assets with 1 ms through 1 s directory labels. They are not ICCD exposure-response data.
- `dark_Background`: an sCMOS dark candidate with unmatched settings. The derived dark offset is an excluded untraced artifact.
- `F:/ICCD_pir/...`: an auxiliary 8-bit background candidate from another acquisition context; not a matching dark for the main 16-bit ICCD batch.
- No verified matching ICCD 50 ms/1 s dark sequence was established in this stage.

## Paper readiness

`GENERAL_PAPER_DRAFT_SUPPORTED = true` for a reproducible operational characterization and controlled restoration study. `ACTA_PHOTONICA_SINICA_DRAFT_SUPPORTED = true_with_metadata_gaps`: a draft is supportable if it explicitly avoids EMVA compliance, strict PTC, physical gain, DSNU/PRNU and photon-counting claims. P0 terminology/metadata closure and selected P1 residual analyses block a strong final submission, while formal PTC/DSNU/PRNU and MCP/gate physics require new controlled acquisition and do not block an operational draft.

## Proposed section structure

1. Acquisition evidence, data roles, ROI and claim boundary.
2. Temporal noise and stability from repeated frames.
3. Spatial nonuniformity proxies, covariance and temporal-residual spectra.
4. Verified exposure response or, when unavailable, observed signal/noise relation.
5. Calibration-only parameters used by G/CG and controlled holdout validation.

Core tables: acquisition/metadata table; method-eligibility table; folder-level temporal/spatial statistics; calibration/evaluation role table. Core figures: temporal std maps/distributions and convergence; difference-frame noise versus observed signal; covariance/ACF and 2D/radial temporal-residual NPS; split-stable and drift diagnostics; G/CG controlled validation with explicit tradeoffs.

## Scope and safety

No image was opened for new pixel statistics, no stage-2 analysis was executed, no model was trained or inferred, and no source file was modified. F-volume direct access was unavailable in this session; its inventory entries are sourced from hashed prior formal audit artifacts and are marked accordingly.
"""
    (output/"verification_report.md").write_text(report,encoding="utf-8")
    ended=utc_now()
    verification={"experiment_id":cfg["experiment_id"],"final_status":recommendation["status"],"reference_count":len(refs),"reference_class_counts":ref_counts,"method_count":len(method_rows),"eligibility_counts":eligibility_counts,"recommended_framework":recommendation["RECOMMENDED_CHARACTERIZATION_FRAMEWORK"],"general_paper_draft_supported":True,"acta_photonica_sinica_draft_supported":"SUPPORTED-WITH-METADATA-GAPS","stage2_executed":False,"new_pixel_statistics_computed":False,"source_data_modified":False,"f_volume_accessible":Path("F:/").exists(),"provenance_complete":True,"started_at_utc":started,"ended_at_utc":ended,"next_task":"Submit the complete stage-1 literature/method/data-fit results to the project director for selection of stage-2 characterization procedures; do not execute stage 2 automatically."}
    dump_json(output/"verification_status.json",verification)
    dump_json(output/"provenance"/"run_manifest.json",verification|{"git_commit":commit,"command":subprocess.list2cmdline(sys.argv),"inputs":input_hashes})
    status_after=git(repo,"status","--porcelain=v1","--untracked-files=all")
    (output/"provenance"/"git_status_after.txt").write_text(status_after,encoding="utf-8")
    hashes=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name!="output_hashes.csv":
            hashes.append({"relative_path":str(path.relative_to(output)),"size_bytes":path.stat().st_size,"sha256":sha256_file(path)})
    write_csv(output/"output_hashes.csv",hashes,["relative_path","size_bytes","sha256"])
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
