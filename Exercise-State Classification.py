from __future__ import annotations
import argparse
import re
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def subject_name_from_path(excel_path: str | Path) -> str:
    return Path(excel_path).stem

def compute_class_weights(y_int: np.ndarray, n_class: int) -> np.ndarray:
    n = len(y_int)
    counts = np.bincount(y_int, minlength=n_class).astype(float)
    counts[counts == 0] = 1.0
    w_class = n / (n_class * counts)
    return w_class[y_int]

def load_and_clean_sheet(excel_path: str | Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if '时间' not in df.columns:
        raise ValueError(f'[{excel_path}] sheet={sheet_name} missing 时间 column. cols={list(df.columns)}')
    df = df.rename(columns={'时间': 'time'})
    stage_col = None
    for cand in ['状态', '阶段', 'stage_raw', 'stage', 'label']:
        if cand in df.columns:
            stage_col = cand
            break
    if stage_col is None:
        for c in df.columns:
            if c == 'time':
                continue
            if pd.api.types.is_object_dtype(df[c]):
                txt = ' '.join(df[c].dropna().astype(str).head(100).tolist())
                if any((k in txt for k in ['静息', '热身', '递增', '恢复', 'RPE'])):
                    stage_col = c
                    break
    if stage_col is not None:
        df = df.rename(columns={stage_col: 'stage_raw'})
    value_col = None
    if sheet_name in df.columns:
        value_col = sheet_name
    else:
        unnamed_candidates = [c for c in df.columns if str(c).startswith('Unnamed') and c != 'stage_raw']
        for c in unnamed_candidates:
            if pd.api.types.is_numeric_dtype(df[c]):
                value_col = c
                break
    if value_col is None:
        for c in df.columns:
            if c in ['time', 'stage_raw']:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                value_col = c
                break
    if value_col is None and len(df.columns) >= 2:
        value_col = df.columns[1]
    if value_col is None:
        raise ValueError(f'[{excel_path}] sheet={sheet_name} cannot find numeric value column.')
    df = df.rename(columns={value_col: sheet_name})
    keep = ['time', sheet_name]
    if 'stage_raw' in df.columns:
        keep.append('stage_raw')
    df = df[keep].copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time']).copy()
    df[sheet_name] = pd.to_numeric(df[sheet_name], errors='coerce')
    return df

def extract_stage_4class(stage_raw: str) -> str:
    if stage_raw is None or (isinstance(stage_raw, float) and np.isnan(stage_raw)):
        return 'Unknown'
    s = str(stage_raw).strip()
    if '静息' in s:
        return 'Rest'
    if '热身' in s:
        return 'Warmup'
    if '递增' in s:
        return 'Exercise'
    if '恢复' in s:
        return 'Recovery'
    return 'Unknown'

def parse_stage_key(stage_raw: str) -> str:
    if stage_raw is None:
        return 'Unknown'
    s = str(stage_raw).strip()
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = ' '.join(s.split())
    return s
_RPE_RE = re.compile('RPE\\s*=\\s*([0-9]+(?:\\.[0-9]+)?)', re.IGNORECASE)

def parse_rpe(stage_raw_or_key: Optional[str]) -> float:
    if stage_raw_or_key is None:
        return float('nan')
    s = str(stage_raw_or_key)
    m = _RPE_RE.search(s)
    if not m:
        return float('nan')
    try:
        return float(m.group(1))
    except Exception:
        return float('nan')

def _time_overlap_seconds(start_a: pd.Timestamp, end_a: pd.Timestamp, start_b: pd.Timestamp, end_b: pd.Timestamp) -> float:
    left = max(start_a, start_b)
    right = min(end_a, end_b)
    return max(0.0, float((right - left).total_seconds()))

def align_sheet_dates_to_reference(sensor_dfs: Dict[str, pd.DataFrame], reference_sheet: str, min_overlap_ratio: float=0.8, max_start_clock_diff_sec: float=3600.0, max_abs_shift_days: int=366) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, object]]]:
    if reference_sheet not in sensor_dfs:
        return (sensor_dfs, [])
    out = {k: v.copy() for k, v in sensor_dfs.items()}
    ref = out[reference_sheet]
    ref_start = pd.to_datetime(ref['time'], errors='coerce').min()
    ref_end = pd.to_datetime(ref['time'], errors='coerce').max()
    if pd.isna(ref_start) or pd.isna(ref_end):
        return (out, [])
    report: List[Dict[str, object]] = []
    for sheet, df in out.items():
        if sheet == reference_sheet or len(df) == 0:
            continue
        sheet_start = pd.to_datetime(df['time'], errors='coerce').min()
        sheet_end = pd.to_datetime(df['time'], errors='coerce').max()
        if pd.isna(sheet_start) or pd.isna(sheet_end):
            continue
        duration_sec = max(1.0, float((sheet_end - sheet_start).total_seconds()))
        original_overlap = _time_overlap_seconds(sheet_start, sheet_end, ref_start, ref_end)
        day_shift = int(round((ref_start - sheet_start).total_seconds() / 86400.0))
        if day_shift == 0 or abs(day_shift) > int(max_abs_shift_days):
            continue
        delta = pd.Timedelta(days=day_shift)
        candidate_start = sheet_start + delta
        candidate_end = sheet_end + delta
        candidate_overlap = _time_overlap_seconds(candidate_start, candidate_end, ref_start, ref_end)
        overlap_ratio = candidate_overlap / duration_sec
        start_clock_diff = abs(float((candidate_start - ref_start).total_seconds()))
        should_shift = overlap_ratio >= float(min_overlap_ratio) and start_clock_diff <= float(max_start_clock_diff_sec) and (candidate_overlap > original_overlap + 0.5 * duration_sec)
        if should_shift:
            out[sheet]['time'] = pd.to_datetime(out[sheet]['time'], errors='coerce') + delta
            report.append({'sheet': sheet, 'day_shift': day_shift, 'old_start': sheet_start, 'new_start': candidate_start, 'reference_start': ref_start, 'overlap_ratio': overlap_ratio})
    return (out, report)

def print_sheet_time_ranges(sensor_dfs: Dict[str, pd.DataFrame], reference_sheet: Optional[str]=None) -> None:
    for sheet, df in sensor_dfs.items():
        if len(df) == 0:
            continue
        t = pd.to_datetime(df['time'], errors='coerce')
        ref_flag = ' [reference]' if sheet == reference_sheet else ''

def build_merged_1hz(excel_path: str | Path, sweat_cols: List[str], sweat_delay_sec: int=300, auto_align_sheet_dates: bool=True, date_align_min_overlap_ratio: float=0.8, date_align_max_clock_diff_sec: float=3600.0, print_time_ranges: bool=False) -> pd.DataFrame:
    xls = pd.ExcelFile(excel_path)
    sheets = xls.sheet_names
    sensor_dfs: Dict[str, pd.DataFrame] = {}
    for sh in sheets:
        sensor_dfs[sh] = load_and_clean_sheet(excel_path, sh)
    stage_source = None
    if 'HR' in sensor_dfs and 'stage_raw' in sensor_dfs['HR'].columns:
        stage_source = 'HR'
    else:
        for sh, df in sensor_dfs.items():
            if 'stage_raw' in df.columns:
                stage_source = sh
                break
    if print_time_ranges:
        print_sheet_time_ranges(sensor_dfs, reference_sheet=stage_source)
    if auto_align_sheet_dates and stage_source is not None:
        sensor_dfs, _time_alignment_report = align_sheet_dates_to_reference(sensor_dfs=sensor_dfs, reference_sheet=stage_source, min_overlap_ratio=date_align_min_overlap_ratio, max_start_clock_diff_sec=date_align_max_clock_diff_sec)
    tmin = min((df['time'].min() for df in sensor_dfs.values()))
    tmax = max((df['time'].max() for df in sensor_dfs.values()))
    timeline = pd.date_range(start=tmin.floor('s'), end=tmax.ceil('s'), freq='1s')
    merged = pd.DataFrame({'time': timeline}).set_index('time')
    for sh, df in sensor_dfs.items():
        d = df.set_index('time').sort_index()
        s0 = d[[sh]].resample('1s').mean()
        if sh == 'SR':
            sr_raw = s0[sh].copy()
            sr_num = pd.to_numeric(sr_raw, errors='coerce')
            sr_valid = (sr_num.notna() & (sr_num > 0)).astype(float)
            merged = merged.join(sr_raw.rename('SR_raw'), how='left')
            merged = merged.join(sr_valid.rename('SR_valid'), how='left')
            s = s0.interpolate(method='time').ffill()
            s[sh] = s[sh].fillna(0.0)
        elif sh == 'SIC':
            sic_raw = s0[sh].copy()
            sic_num = pd.to_numeric(sic_raw, errors='coerce')
            sic_valid = sic_num.notna().astype(float)
            merged = merged.join(sic_valid.rename('SIC_valid'), how='left')
            s = s0.interpolate(method='time').ffill()
            s[sh] = s[sh].fillna(0.0)
        else:
            s = s0.interpolate(method='time').ffill().bfill()
        merged = merged.join(s, how='left')
    if 'SR_raw' in merged.columns:
        merged['SR_raw'] = pd.to_numeric(merged['SR_raw'], errors='coerce').fillna(0.0)
    if 'SR_valid' in merged.columns:
        merged['SR_valid'] = pd.to_numeric(merged['SR_valid'], errors='coerce').fillna(0.0)
    if 'SIC_valid' in merged.columns:
        merged['SIC_valid'] = pd.to_numeric(merged['SIC_valid'], errors='coerce').fillna(0.0)
    if stage_source is not None:
        st = sensor_dfs[stage_source].set_index('time').sort_index()[['stage_raw']]
        st_1hz = st.resample('1s').ffill()
        merged = merged.join(st_1hz, how='left')
        merged['stage_raw'] = merged['stage_raw'].ffill().bfill()
        merged['label'] = merged['stage_raw'].apply(extract_stage_4class)
        merged['stage_key'] = merged['stage_raw'].apply(parse_stage_key)
        merged['rpe'] = merged['stage_raw'].apply(parse_rpe)
    else:
        merged['stage_raw'] = np.nan
        merged['label'] = 'Unknown'
        merged['stage_key'] = 'Unknown'
        merged['rpe'] = np.nan
    merged = merged.reset_index()
    if sweat_delay_sec != 0:
        for c in sweat_cols:
            if c in merged.columns:
                merged[c] = merged[c].shift(-sweat_delay_sec)
        have_cols = [c for c in sweat_cols if c in merged.columns]
        if len(have_cols) > 0:
            merged = merged.dropna(subset=have_cols).reset_index(drop=True)
    merged['rpe'] = pd.to_numeric(merged['rpe'], errors='coerce')
    return merged

def baseline_normalize_by_initial_rest(df: pd.DataFrame, sensor_cols: List[str], baseline_sec: int=180, scale_floor_fraction: float=0.1, clip_z: float=20.0, sic_clip_floor: float=0.0) -> pd.DataFrame:
    out = df.copy()
    out['time'] = pd.to_datetime(out['time'])
    t_start = out['time'].min()
    baseline_df = out[out['time'] <= t_start + pd.Timedelta(seconds=baseline_sec)]
    for c in sensor_cols:
        x_all = pd.to_numeric(out[c], errors='coerce').to_numpy(dtype=float)
        x_all_finite = x_all[np.isfinite(x_all)]
        x_base = pd.to_numeric(baseline_df[c], errors='coerce').to_numpy(dtype=float)
        x_base = x_base[np.isfinite(x_base)]
        if len(x_base) < 5:
            x_base = x_all_finite
        if len(x_base) == 0 or len(x_all_finite) == 0:
            out[c] = 0.0
            continue
        base_median = float(np.nanmedian(x_base))
        q25, q75 = np.nanpercentile(x_base, [25, 75])
        base_scale = float((q75 - q25) / 1.349)
        if not np.isfinite(base_scale) or base_scale < 1e-09:
            base_scale = float(np.nanstd(x_base))
        s10, s90 = np.nanpercentile(x_all_finite, [10, 90])
        session_scale = float((s90 - s10) / 2.563)
        if not np.isfinite(session_scale) or session_scale < 1e-09:
            session_scale = float(np.nanstd(x_all_finite))
        if not np.isfinite(session_scale) or session_scale < 1e-09:
            session_scale = 1.0
        scale_floor = max(0.0, float(scale_floor_fraction)) * session_scale
        scale = max(base_scale if np.isfinite(base_scale) else 0.0, scale_floor, 1e-06)
        x = pd.to_numeric(out[c], errors='coerce')
        z = (x - base_median) / scale
        sensor_clip_z = float(clip_z) if clip_z is not None else 0.0
        if str(c) == 'SIC' and sensor_clip_z > 0 and (float(sic_clip_floor) > 0):
            sensor_clip_z = max(sensor_clip_z, float(sic_clip_floor))
        if sensor_clip_z > 0:
            z = z.clip(-sensor_clip_z, sensor_clip_z)
        out[c] = z
    return out

def block_stats(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'first': 0.0, 'last': 0.0, 'delta': 0.0, 'range': 0.0, 'slope': 0.0}
    t = np.arange(len(x), dtype=float)
    slope = 0.0
    if len(x) >= 2:
        slope = float(np.polyfit(t, x, 1)[0])
    return {'mean': float(np.mean(x)), 'std': float(np.std(x)), 'min': float(np.min(x)), 'max': float(np.max(x)), 'first': float(x[0]), 'last': float(x[-1]), 'delta': float(x[-1] - x[0]), 'range': float(np.max(x) - np.min(x)), 'slope': slope}

def split_segments(arr: np.ndarray, n_segments: int) -> List[np.ndarray]:
    if n_segments <= 1:
        return [arr]
    length = len(arr)
    segs = []
    for i in range(n_segments):
        a = int(i * length / n_segments)
        b = int((i + 1) * length / n_segments)
        segs.append(arr[a:b])
    return segs

def make_substage_feature_dataset(df_1hz: pd.DataFrame, subject_id: str, n_segments: int=3, baseline_sec: int=180, scale_floor_fraction: float=0.1, clip_z: float=20.0, sic_clip_floor: float=0.0) -> pd.DataFrame:
    df = df_1hz.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df[df['label'] != 'Unknown'].reset_index(drop=True)
    meta_cols = {'time', 'stage_raw', 'label', 'stage_key', 'rpe'}
    sensor_cols = [c for c in df.columns if c not in meta_cols]
    norm_cols = [c for c in sensor_cols if c not in {'SR_valid', 'SIC_valid'}]
    df = baseline_normalize_by_initial_rest(df, sensor_cols=norm_cols, baseline_sec=baseline_sec, scale_floor_fraction=scale_floor_fraction, clip_z=clip_z, sic_clip_floor=sic_clip_floor)
    rows = []
    for stage_key, g in df.groupby('stage_key', sort=False):
        g = g.sort_values('time').reset_index(drop=True)
        if len(g) < 5:
            continue
        label = str(g['label'].mode().iloc[0])
        if label == 'Unknown':
            continue
        feats = {'subject_id': subject_id, 'stage_key': stage_key, 'label': label, 't_start': g['time'].iloc[0], 't_end': g['time'].iloc[-1], 'duration_sec': float((g['time'].iloc[-1] - g['time'].iloc[0]).total_seconds()), 'rpe_substage': float(np.nanmedian(g['rpe'].to_numpy(dtype=float)))}
        for col in sensor_cols:
            x = pd.to_numeric(g[col], errors='coerce').to_numpy(dtype=float)
            st = block_stats(x)
            for k, v in st.items():
                feats[f'{col}__stage__{k}'] = v
            x_finite = x[np.isfinite(x)]
            if len(x_finite) > 0:
                feats[f'{col}__stage__p25'] = float(np.percentile(x_finite, 25))
                feats[f'{col}__stage__p50'] = float(np.percentile(x_finite, 50))
                feats[f'{col}__stage__p75'] = float(np.percentile(x_finite, 75))
            else:
                feats[f'{col}__stage__p25'] = 0.0
                feats[f'{col}__stage__p50'] = 0.0
                feats[f'{col}__stage__p75'] = 0.0
            for si, seg in enumerate(split_segments(x, n_segments), start=1):
                st_seg = block_stats(seg)
                for k in ['mean', 'std', 'last', 'delta', 'slope']:
                    feats[f'{col}__seg{si}_{n_segments}__{k}'] = st_seg[k]
        if 'SR_valid' in g.columns:
            valid = pd.to_numeric(g['SR_valid'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            valid = np.clip(valid, 0.0, 1.0)
            n = len(valid)
            feats['SR_presence_frac'] = float(np.mean(valid)) if n > 0 else 0.0
            feats['SR_has_any'] = float(np.max(valid)) if n > 0 else 0.0
            if np.any(valid > 0.5):
                first_idx = int(np.argmax(valid > 0.5))
                feats['SR_first_present_frac'] = float(first_idx / max(1, n - 1))
                feats['SR_first_present_sec'] = float(first_idx)
            else:
                feats['SR_first_present_frac'] = 1.0
                feats['SR_first_present_sec'] = float(n)
            half = max(1, n // 2)
            feats['SR_valid_first_half_frac'] = float(np.mean(valid[:half])) if n > 0 else 0.0
            feats['SR_valid_second_half_frac'] = float(np.mean(valid[half:])) if n > half else feats['SR_valid_first_half_frac']
            feats['SR_valid_delta_half'] = feats['SR_valid_second_half_frac'] - feats['SR_valid_first_half_frac']
        if 'SR' in g.columns:
            sr = pd.to_numeric(g['SR'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            sr_st = block_stats(sr)
            feats['SR_zero_stage_mean'] = sr_st['mean']
            feats['SR_zero_stage_last'] = sr_st['last']
            feats['SR_zero_stage_delta'] = sr_st['delta']
            feats['SR_zero_stage_slope'] = sr_st['slope']
            segs_sr = split_segments(sr, n_segments)
            if len(segs_sr) >= 2:
                feats['SR_late_minus_early_mean'] = float(np.mean(segs_sr[-1]) - np.mean(segs_sr[0]))
            else:
                feats['SR_late_minus_early_mean'] = 0.0
        if 'SIC_valid' in g.columns:
            sic_valid = pd.to_numeric(g['SIC_valid'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            sic_valid = np.clip(sic_valid, 0.0, 1.0)
            n_sic = len(sic_valid)
            feats['SIC_presence_frac'] = float(np.mean(sic_valid)) if n_sic > 0 else 0.0
            feats['SIC_has_any'] = float(np.max(sic_valid)) if n_sic > 0 else 0.0
            if np.any(sic_valid > 0.5):
                sic_first_idx = int(np.argmax(sic_valid > 0.5))
                feats['SIC_first_present_frac'] = float(sic_first_idx / max(1, n_sic - 1))
            else:
                feats['SIC_first_present_frac'] = 1.0
        rows.append(feats)
    return pd.DataFrame(rows)

def make_window_feature_dataset(df_1hz: pd.DataFrame, subject_id: str, window_sec: int=60, stride_sec: int=30, n_segments: int=3, baseline_sec: int=180, scale_floor_fraction: float=0.1, clip_z: float=20.0, sic_clip_floor: float=0.0, min_window_coverage: float=0.8, min_label_fraction: float=0.8, transition_margin_sec: int=0) -> pd.DataFrame:
    if window_sec <= 0 or stride_sec <= 0:
        raise ValueError('window_sec and stride_sec must be positive integers.')
    df = df_1hz.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df[df['label'] != 'Unknown'].reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame()
    meta_cols = {'time', 'stage_raw', 'label', 'stage_key', 'rpe'}
    sensor_cols = [c for c in df.columns if c not in meta_cols]
    norm_cols = [c for c in sensor_cols if c not in {'SR_valid', 'SIC_valid'}]
    df = baseline_normalize_by_initial_rest(df, sensor_cols=norm_cols, baseline_sec=baseline_sec, scale_floor_fraction=scale_floor_fraction, clip_z=clip_z, sic_clip_floor=sic_clip_floor)
    start_time = df['time'].min().floor('s')
    end_time = df['time'].max().ceil('s')
    win = pd.Timedelta(seconds=int(window_sec))
    stride = pd.Timedelta(seconds=int(stride_sec))
    transition_times = []
    if int(transition_margin_sec) > 0 and 'label' in df.columns:
        label_change = df['label'].astype(str).ne(df['label'].astype(str).shift())
        transition_times = pd.to_datetime(df.loc[label_change, 'time']).iloc[1:].tolist()
    transition_margin = pd.Timedelta(seconds=max(0, int(transition_margin_sec)))
    rows = []
    window_index = 0
    current = start_time
    while current + win <= end_time + pd.Timedelta(seconds=1):
        w_end = current + win
        g = df[(df['time'] >= current) & (df['time'] < w_end)].copy()
        expected_n = max(1, int(window_sec))
        coverage = len(g) / expected_n
        if len(g) < 5 or coverage < float(min_window_coverage):
            current += stride
            window_index += 1
            continue
        label_counts = g['label'].astype(str).value_counts()
        label = str(label_counts.index[0])
        label_fraction = float(label_counts.iloc[0] / len(g))
        if label_fraction < float(min_label_fraction):
            current += stride
            window_index += 1
            continue
        if transition_times:
            center_time = current + win / 2
            too_close_to_transition = any((abs(center_time - tt) <= transition_margin for tt in transition_times))
            if too_close_to_transition:
                current += stride
                window_index += 1
                continue
        if 'stage_key' in g.columns and g['stage_key'].notna().any():
            stage_key = str(g['stage_key'].mode().iloc[0])
        else:
            stage_key = f'window_{window_index:04d}'
        feats = {'subject_id': subject_id, 'stage_key': stage_key, 'label': label, 't_start': current, 't_end': w_end - pd.Timedelta(seconds=1), 'duration_sec': float(window_sec), 'window_index': int(window_index), 'window_coverage': float(coverage), 'label_fraction': float(label_fraction), 'n_points': int(len(g)), 'rpe_substage': float(np.nanmedian(g['rpe'].to_numpy(dtype=float)))}
        for col in sensor_cols:
            x = pd.to_numeric(g[col], errors='coerce').to_numpy(dtype=float)
            st = block_stats(x)
            for k, v in st.items():
                feats[f'{col}__stage__{k}'] = v
            x_finite = x[np.isfinite(x)]
            if len(x_finite) > 0:
                feats[f'{col}__stage__p25'] = float(np.percentile(x_finite, 25))
                feats[f'{col}__stage__p50'] = float(np.percentile(x_finite, 50))
                feats[f'{col}__stage__p75'] = float(np.percentile(x_finite, 75))
            else:
                feats[f'{col}__stage__p25'] = 0.0
                feats[f'{col}__stage__p50'] = 0.0
                feats[f'{col}__stage__p75'] = 0.0
            for si, seg in enumerate(split_segments(x, n_segments), start=1):
                st_seg = block_stats(seg)
                for k in ['mean', 'std', 'last', 'delta', 'slope']:
                    feats[f'{col}__seg{si}_{n_segments}__{k}'] = st_seg[k]
        if 'Temp' in g.columns and 'HR' in g.columns:
            temp_x = pd.to_numeric(g['Temp'], errors='coerce').to_numpy(dtype=float)
            hr_x = pd.to_numeric(g['HR'], errors='coerce').to_numpy(dtype=float)
            temp_st = block_stats(temp_x)
            hr_st = block_stats(hr_x)
            feats['Temp_HR__stage__delta_sum'] = temp_st['delta'] + hr_st['delta']
            feats['Temp_HR__stage__slope_sum'] = temp_st['slope'] + hr_st['slope']
            feats['Temp_HR__stage__mean_sum'] = temp_st['mean'] + hr_st['mean']
            feats['Temp_HR__stage__delta_diff'] = temp_st['delta'] - hr_st['delta']
            feats['Temp_HR__stage__slope_diff'] = temp_st['slope'] - hr_st['slope']
            temp_segs = split_segments(temp_x, n_segments)
            hr_segs = split_segments(hr_x, n_segments)
            if len(temp_segs) >= 2 and len(hr_segs) >= 2:
                temp_late_early = float(np.mean(temp_segs[-1]) - np.mean(temp_segs[0]))
                hr_late_early = float(np.mean(hr_segs[-1]) - np.mean(hr_segs[0]))
                feats['Temp_HR__late_minus_early_sum'] = temp_late_early + hr_late_early
                feats['Temp_HR__late_minus_early_diff'] = temp_late_early - hr_late_early
            else:
                feats['Temp_HR__late_minus_early_sum'] = 0.0
                feats['Temp_HR__late_minus_early_diff'] = 0.0
        if 'SR_valid' in g.columns:
            valid = pd.to_numeric(g['SR_valid'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            valid = np.clip(valid, 0.0, 1.0)
            n = len(valid)
            feats['SR_presence_frac'] = float(np.mean(valid)) if n > 0 else 0.0
            feats['SR_has_any'] = float(np.max(valid)) if n > 0 else 0.0
            if np.any(valid > 0.5):
                first_idx = int(np.argmax(valid > 0.5))
                feats['SR_first_present_frac'] = float(first_idx / max(1, n - 1))
                feats['SR_first_present_sec'] = float(first_idx)
            else:
                feats['SR_first_present_frac'] = 1.0
                feats['SR_first_present_sec'] = float(n)
            half = max(1, n // 2)
            feats['SR_valid_first_half_frac'] = float(np.mean(valid[:half])) if n > 0 else 0.0
            feats['SR_valid_second_half_frac'] = float(np.mean(valid[half:])) if n > half else feats['SR_valid_first_half_frac']
            feats['SR_valid_delta_half'] = feats['SR_valid_second_half_frac'] - feats['SR_valid_first_half_frac']
        if 'SR' in g.columns:
            sr = pd.to_numeric(g['SR'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            sr_st = block_stats(sr)
            feats['SR_zero_stage_mean'] = sr_st['mean']
            feats['SR_zero_stage_last'] = sr_st['last']
            feats['SR_zero_stage_delta'] = sr_st['delta']
            feats['SR_zero_stage_slope'] = sr_st['slope']
            segs_sr = split_segments(sr, n_segments)
            if len(segs_sr) >= 2:
                feats['SR_late_minus_early_mean'] = float(np.mean(segs_sr[-1]) - np.mean(segs_sr[0]))
            else:
                feats['SR_late_minus_early_mean'] = 0.0
        if 'SIC_valid' in g.columns:
            sic_valid = pd.to_numeric(g['SIC_valid'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            sic_valid = np.clip(sic_valid, 0.0, 1.0)
            n_sic = len(sic_valid)
            feats['SIC_presence_frac'] = float(np.mean(sic_valid)) if n_sic > 0 else 0.0
            feats['SIC_has_any'] = float(np.max(sic_valid)) if n_sic > 0 else 0.0
            if np.any(sic_valid > 0.5):
                sic_first_idx = int(np.argmax(sic_valid > 0.5))
                feats['SIC_first_present_frac'] = float(sic_first_idx / max(1, n_sic - 1))
            else:
                feats['SIC_first_present_frac'] = 1.0
        rows.append(feats)
        current += stride
        window_index += 1
    return pd.DataFrame(rows)

def add_longitudinal_features(ds: pd.DataFrame, sensors: List[str], mode: str='offline') -> pd.DataFrame:
    if mode == 'none':
        return ds.copy()
    if mode not in {'causal', 'offline'}:
        raise ValueError(f'Unknown trajectory mode: {mode}')
    out = ds.copy()
    out['t_start'] = pd.to_datetime(out['t_start'])
    out = out.sort_values(['subject_id', 't_start']).reset_index(drop=True)
    source_cols: List[str] = []
    for sensor in sensors:
        for stat in ['mean', 'slope', 'delta']:
            c = f'{sensor}__stage__{stat}'
            if c in out.columns:
                source_cols.append(c)
    extra: Dict[str, pd.Series] = {}
    groups = out.groupby('subject_id', sort=False)
    for c in source_cols:
        x = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
        gx = x.groupby(out['subject_id'], sort=False)
        d1 = gx.diff().fillna(0.0)
        extra[f'{c}__traj_prev_diff'] = d1
        extra[f'{c}__traj_from_start'] = x - gx.transform('first')
        extra[f'{c}__traj_accel'] = d1.groupby(out['subject_id'], sort=False).diff().fillna(0.0)
        running_min = gx.cummin()
        running_max = gx.cummax()
        running_den = (running_max - running_min).replace(0.0, np.nan)
        extra[f'{c}__traj_running_pos'] = ((x - running_min) / running_den).fillna(0.5)
        if mode == 'offline':
            extra[f'{c}__traj_rank_pct'] = gx.rank(method='average', pct=True)
            med = gx.transform('median')
            q25 = groups[c].transform(lambda s: pd.to_numeric(s, errors='coerce').quantile(0.25))
            q75 = groups[c].transform(lambda s: pd.to_numeric(s, errors='coerce').quantile(0.75))
            scale = ((q75 - q25) / 1.349).clip(lower=1e-06)
            extra[f'{c}__traj_session_z'] = ((x - med) / scale).clip(-10.0, 10.0)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out

def add_subject_signal_quality_features(ds: pd.DataFrame, sensors: List[str]) -> pd.DataFrame:
    out = ds.copy()
    extra: Dict[str, pd.Series] = {}
    for sensor in sensors:
        c = f'{sensor}__stage__mean'
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
        gx = x.groupby(out['subject_id'], sort=False)
        x_min = gx.transform('min')
        x_max = gx.transform('max')
        q25 = gx.transform(lambda s: s.quantile(0.25))
        q75 = gx.transform(lambda s: s.quantile(0.75))
        extra[f'{sensor}__quality_session_range'] = (x_max - x_min).clip(lower=0.0)
        extra[f'{sensor}__quality_session_iqr'] = (q75 - q25).clip(lower=0.0)
        extra[f'{sensor}__quality_nonzero_frac'] = (x.abs() > 1e-12).astype(float).groupby(out['subject_id'], sort=False).transform('mean')
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out

def print_signal_quality_warnings(train_ds: pd.DataFrame, val_ds: pd.DataFrame, sensors: List[str], low_ratio: float=0.05, high_ratio: float=20.0) -> None:
    for sensor in sensors:
        qcol = f'{sensor}__quality_session_range'
        if qcol not in train_ds.columns or qcol not in val_ds.columns:
            continue
        train_by_subject = train_ds.groupby('subject_id')[qcol].first()
        train_median = float(np.nanmedian(train_by_subject.to_numpy(dtype=float)))
        val_range = float(pd.to_numeric(val_ds[qcol], errors='coerce').iloc[0])
        if not np.isfinite(train_median) or train_median <= 1e-12:
            continue
        ratio = val_range / train_median
        if ratio < low_ratio or ratio > high_ratio:
            pass
FORBIDDEN_FEATURE_COLUMNS = {'subject_id', 'stage_key', 'label', 't_start', 't_end', 'rpe_substage', 'duration_sec', 'window_index', 'window_coverage', 'label_fraction', 'n_points'}

def make_sensor_feature_matrix(ds: pd.DataFrame, feature_cols: Optional[List[str]]=None, feature_mode: str='compact') -> pd.DataFrame:
    if ds is None or len(ds) == 0:
        raise ValueError('Input dataset is empty.')
    if feature_cols is None:
        if feature_mode not in {'all', 'compact'}:
            raise ValueError(f'Unknown feature_mode={feature_mode}')
        cols = []
        for c in ds.columns:
            c_str = str(c)
            if c_str in FORBIDDEN_FEATURE_COLUMNS:
                continue
            if not pd.api.types.is_numeric_dtype(ds[c]):
                continue
            if feature_mode == 'all':
                cols.append(c_str)
                continue
            keep = False
            if '__traj_' in c_str or '__quality_' in c_str:
                keep = True
            if c_str.startswith('SR_') or c_str.startswith('SIC_') or c_str.startswith('Temp_HR__'):
                keep = True
            if '__stage__' in c_str and any((c_str.endswith(suffix) for suffix in ['__mean', '__std', '__delta', '__slope', '__p25', '__p50', '__p75'])):
                keep = True
            if any((tag in c_str for tag in ['__seg1_3__mean', '__seg3_3__mean', '__seg1_3__slope', '__seg3_3__slope'])):
                keep = True
            if keep:
                cols.append(c_str)
        if len(cols) == 0:
            raise ValueError(f'No numeric sensor feature columns found. Available columns: {list(ds.columns)}')
        X = ds.loc[:, cols].copy()
    else:
        X = ds.reindex(columns=feature_cols).copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    if feature_cols is None:
        nonconstant = X.nunique(dropna=False) > 1
        X = X.loc[:, nonconstant]
    X = X.astype(np.float32)
    return X

def train_xgb_4class(X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray]=None, seed: int=42, n_estimators: int=250, max_depth: int=2, min_child_weight: float=3.0, reg_alpha: float=0.3, reg_lambda: float=5.0, gamma: float=0.05, subsample: float=0.85, colsample_bytree: float=0.7, n_jobs: int=1) -> XGBClassifier:
    clf = XGBClassifier(objective='multi:softprob', num_class=4, n_estimators=n_estimators, learning_rate=0.05, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, min_child_weight=min_child_weight, gamma=gamma, tree_method='hist', eval_metric='mlogloss', random_state=seed, n_jobs=int(n_jobs))
    clf.fit(X, y, sample_weight=sample_weight)
    return clf

def train_reference_binary_active_model(X: pd.DataFrame, y4: np.ndarray, seed: int=42, n_estimators: int=250, n_jobs: int=1) -> XGBClassifier:
    y01 = (np.asarray(y4, dtype=int) != 0).astype(int)
    w = compute_class_weights(y01, n_class=2)
    clf = XGBClassifier(objective='binary:logistic', n_estimators=int(n_estimators), learning_rate=0.04, max_depth=3, subsample=0.9, colsample_bytree=0.85, reg_lambda=2.0, min_child_weight=2.0, gamma=0.0, tree_method='hist', eval_metric='logloss', random_state=int(seed), n_jobs=int(n_jobs))
    clf.fit(X, y01, sample_weight=w)
    return clf

def train_reference_active_3class_model(X: pd.DataFrame, y4: np.ndarray, seed: int=42, n_estimators: int=350, n_jobs: int=1) -> XGBClassifier:
    y4 = np.asarray(y4, dtype=int)
    active_mask = y4 != 0
    if int(np.sum(active_mask)) == 0:
        raise ValueError('No active rows are available for the reference active-3-class model.')
    y3 = y4[active_mask] - 1
    w = compute_class_weights(y3, n_class=3)
    clf = XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=int(n_estimators), learning_rate=0.03, max_depth=3, subsample=0.9, colsample_bytree=0.85, reg_lambda=2.0, min_child_weight=2.0, gamma=0.0, tree_method='hist', eval_metric='mlogloss', random_state=int(seed), n_jobs=int(n_jobs))
    clf.fit(X.loc[active_mask], y3, sample_weight=w)
    return clf

def predict_reference_hierarchical_probability(binary_model: XGBClassifier, active_model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    p_active = np.asarray(binary_model.predict_proba(X), dtype=float)[:, 1]
    p3_raw = np.asarray(active_model.predict_proba(X), dtype=float)
    p3 = np.zeros((len(X), 3), dtype=float)
    for col_i, cls in enumerate(np.asarray(active_model.classes_, dtype=int)):
        if 0 <= int(cls) < 3:
            p3[:, int(cls)] = p3_raw[:, col_i]
    p4 = np.zeros((len(X), 4), dtype=float)
    p4[:, 0] = 1.0 - p_active
    p4[:, 1:] = p_active[:, None] * p3
    row_sum = p4.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return p4 / row_sum

def fuse_reference_hierarchical_probability(direct_proba4: np.ndarray, hierarchical_proba4: np.ndarray, alpha: float=0.35, uncertainty_margin: float=0.25, allow_label_changes: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(direct_proba4, dtype=float)
    h = np.asarray(hierarchical_proba4, dtype=float)
    if p.shape != h.shape or p.ndim != 2 or p.shape[1] != 4:
        raise ValueError(f'Probability shape mismatch: direct={p.shape}, hierarchical={h.shape}')
    alpha = float(np.clip(alpha, 0.0, 1.0))
    margin_thr = max(float(uncertainty_margin), 1e-06)
    sorted_p = np.sort(p, axis=1)
    direct_margin = sorted_p[:, -1] - sorted_p[:, -2]
    gate = np.clip((margin_thr - direct_margin) / margin_thr, 0.0, 1.0)
    if not allow_label_changes:
        agree = np.argmax(p, axis=1) == np.argmax(h, axis=1)
        gate = gate * agree.astype(float)
    alpha_row = alpha * gate
    fused = (1.0 - alpha_row[:, None]) * p + alpha_row[:, None] * h
    row_sum = fused.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    fused = fused / row_sum
    return (fused, alpha_row)
TEMP_EARLY_SUFFIXES = ['__stage__mean', '__stage__std', '__stage__delta', '__stage__slope', '__stage__p25', '__stage__p50', '__stage__p75', '__stage__mean__traj_prev_diff', '__stage__mean__traj_from_start', '__stage__mean__traj_accel', '__stage__mean__traj_running_pos', '__stage__mean__traj_rank_pct', '__stage__mean__traj_session_z', '__stage__slope__traj_prev_diff', '__stage__slope__traj_rank_pct', '__stage__slope__traj_session_z']

def get_temp_early_feature_columns(ds: pd.DataFrame) -> List[str]:
    cols = []
    for suffix in TEMP_EARLY_SUFFIXES:
        c = f'Temp{suffix}'
        if c in ds.columns and pd.api.types.is_numeric_dtype(ds[c]):
            cols.append(c)
    return cols

def train_temp_warmup_auxiliary(train_ds: pd.DataFrame, seed: int=42, C: float=0.1):
    early = train_ds[train_ds['label'].isin(['Rest', 'Warmup'])].copy()
    cols = get_temp_early_feature_columns(early)
    if len(cols) == 0 or len(early) == 0:
        return (None, [])
    y = (early['label'].astype(str) == 'Warmup').astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return (None, cols)
    X = early.reindex(columns=cols).copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=float(C), class_weight='balanced', max_iter=5000, random_state=seed))
    model.fit(X, y)
    return (model, cols)

def predict_temp_warmup_probability(model, feature_cols: List[str], ds: pd.DataFrame) -> np.ndarray:
    if model is None or len(feature_cols) == 0:
        return np.zeros(len(ds), dtype=float)
    X = ds.reindex(columns=feature_cols).copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return model.predict_proba(X)[:, 1].astype(float)

def estimate_sensor_onset_index(ds: pd.DataFrame, feature_col: str, onset_fraction: float=0.15, baseline_segments: int=2) -> int:
    if feature_col not in ds.columns or len(ds) == 0:
        return int(len(ds))
    ordered = ds.sort_values('t_start').reset_index(drop=True)
    x = pd.to_numeric(ordered[feature_col], errors='coerce')
    x = x.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    values = x.to_numpy(dtype=float)
    n_base = max(1, min(int(baseline_segments), len(values)))
    baseline = float(np.median(values[:n_base]))
    peak = float(np.max(values))
    dynamic_range = peak - baseline
    if not np.isfinite(dynamic_range) or dynamic_range <= 1e-09:
        return int(len(values))
    fraction = float(np.clip(onset_fraction, 0.01, 0.95))
    threshold = baseline + fraction * dynamic_range
    hits = np.flatnonzero(values >= threshold)
    return int(hits[0]) if len(hits) > 0 else int(len(values))

def choose_temp_fusion_alpha(val_ds: pd.DataFrame, base_alpha: float=0.35, delayed_hr_alpha: float=0.7, hr_delay_segments: int=2, onset_fraction: float=0.15, baseline_segments: int=2) -> Tuple[float, int, int, int]:
    temp_onset = estimate_sensor_onset_index(val_ds, feature_col='Temp__stage__mean', onset_fraction=onset_fraction, baseline_segments=baseline_segments)
    hr_onset = estimate_sensor_onset_index(val_ds, feature_col='HR__stage__mean', onset_fraction=onset_fraction, baseline_segments=baseline_segments)
    detected_delay = int(hr_onset - temp_onset)
    use_delayed_weight = detected_delay >= int(hr_delay_segments)
    alpha = delayed_hr_alpha if use_delayed_weight else base_alpha
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (alpha, int(temp_onset), int(hr_onset), detected_delay)

def fuse_rest_warmup_probabilities(proba4: np.ndarray, temp_warmup_proba: np.ndarray, label_to_id: Dict[str, int], alpha: float=0.35) -> np.ndarray:
    p = np.asarray(proba4, dtype=float).copy()
    q = np.asarray(temp_warmup_proba, dtype=float)
    q = np.clip(q, 1e-06, 1.0 - 1e-06)
    a = float(np.clip(alpha, 0.0, 1.0))
    rest_id = label_to_id['Rest']
    warmup_id = label_to_id['Warmup']
    rw_mass = p[:, rest_id] + p[:, warmup_id]
    p[:, rest_id] = (1.0 - a) * p[:, rest_id] + a * rw_mass * (1.0 - q)
    p[:, warmup_id] = (1.0 - a) * p[:, warmup_id] + a * rw_mass * q
    row_sum = p.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return p / row_sum

def fuse_rest_warmup_probabilities_adaptive(proba4: np.ndarray, temp_warmup_proba: np.ndarray, label_to_id: Dict[str, int], alpha: float=0.35, min_temp_conf: float=0.3, min_rw_mass: float=0.05) -> np.ndarray:
    p = np.asarray(proba4, dtype=float).copy()
    q = np.asarray(temp_warmup_proba, dtype=float)
    q = np.clip(q, 1e-06, 1.0 - 1e-06)
    base_alpha = float(np.clip(alpha, 0.0, 1.0))
    rest_id = label_to_id['Rest']
    warmup_id = label_to_id['Warmup']
    rw_mass = p[:, rest_id] + p[:, warmup_id]
    safe_mass = np.maximum(rw_mass, 1e-12)
    main_warmup_frac = p[:, warmup_id] / safe_mass
    main_uncertainty = 1.0 - 2.0 * np.abs(main_warmup_frac - 0.5)
    main_uncertainty = np.clip(main_uncertainty, 0.0, 1.0)
    temp_conf = 2.0 * np.abs(q - 0.5)
    temp_conf = np.clip(temp_conf, 0.0, 1.0)
    row_alpha = base_alpha * temp_conf * (0.25 + 0.75 * main_uncertainty)
    row_alpha = np.where(temp_conf >= float(min_temp_conf), row_alpha, 0.0)
    row_alpha = np.where(rw_mass >= float(min_rw_mass), row_alpha, 0.0)
    row_alpha = np.clip(row_alpha, 0.0, 1.0)
    p[:, rest_id] = (1.0 - row_alpha) * p[:, rest_id] + row_alpha * rw_mass * (1.0 - q)
    p[:, warmup_id] = (1.0 - row_alpha) * p[:, warmup_id] + row_alpha * rw_mass * q
    row_sum = p.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return p / row_sum

def apply_early_temp_rescue(val_ds: pd.DataFrame, y_pred: np.ndarray, temp_warmup_proba: np.ndarray, label_to_id: Dict[str, int], threshold: float=0.5, min_baseline_segments: int=2) -> np.ndarray:
    y = np.asarray(y_pred, dtype=int).copy()
    p_temp = np.asarray(temp_warmup_proba, dtype=float)
    rest_id = label_to_id['Rest']
    warmup_id = label_to_id['Warmup']
    exercise_id = label_to_id['Exercise']
    recovery_id = label_to_id['Recovery']
    order = np.argsort(pd.to_datetime(val_ds['t_start']).to_numpy())
    seen_exercise = False
    warmup_started = False
    for position, idx in enumerate(order):
        current = int(y[idx])
        if current == exercise_id:
            seen_exercise = True
        if seen_exercise:
            continue
        if current == recovery_id:
            current = rest_id
        if position >= int(min_baseline_segments) and p_temp[idx] >= float(threshold) and (current in [rest_id, recovery_id]):
            current = warmup_id
        if current == warmup_id:
            warmup_started = True
        elif warmup_started and current == rest_id:
            current = warmup_id
        y[idx] = current
    return y
TEMP_HR_EARLY_SUFFIXES = ['__stage__mean', '__stage__std', '__stage__delta', '__stage__slope', '__stage__p25', '__stage__p50', '__stage__p75', '__seg1_3__mean', '__seg3_3__mean', '__seg1_3__slope', '__seg3_3__slope', '__stage__mean__traj_prev_diff', '__stage__mean__traj_from_start', '__stage__mean__traj_accel', '__stage__mean__traj_running_pos', '__stage__slope__traj_prev_diff']

def get_temp_hr_early_feature_columns(ds: pd.DataFrame) -> List[str]:
    cols = []
    for sensor in ['Temp', 'HR']:
        for suffix in TEMP_HR_EARLY_SUFFIXES:
            c = f'{sensor}{suffix}'
            if c in ds.columns and pd.api.types.is_numeric_dtype(ds[c]):
                cols.append(c)
    for c in ds.columns:
        c_str = str(c)
        if c_str.startswith('Temp_HR__') and pd.api.types.is_numeric_dtype(ds[c]):
            cols.append(c_str)
    return list(dict.fromkeys(cols))

def train_temp_hr_warmup_auxiliary(train_ds: pd.DataFrame, seed: int=42, C: float=0.15):
    early = train_ds[train_ds['label'].isin(['Rest', 'Warmup'])].copy()
    cols = get_temp_hr_early_feature_columns(early)
    if len(cols) == 0 or len(early) == 0:
        return (None, [])
    y = (early['label'].astype(str) == 'Warmup').astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return (None, cols)
    X = early.reindex(columns=cols).copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=float(C), class_weight='balanced', max_iter=5000, random_state=seed))
    model.fit(X, y)
    return (model, cols)

def predict_temp_hr_warmup_probability(model, feature_cols: List[str], ds: pd.DataFrame) -> np.ndarray:
    if model is None or len(feature_cols) == 0:
        return np.zeros(len(ds), dtype=float)
    X = ds.reindex(columns=feature_cols).copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return model.predict_proba(X)[:, 1].astype(float)

def fuse_rest_warmup_probabilities_with_expert(proba4: np.ndarray, expert_warmup_proba: np.ndarray, label_to_id: Dict[str, int], alpha: float=0.25, min_expert_conf: float=0.2, min_rw_mass: float=0.05) -> np.ndarray:
    p = np.asarray(proba4, dtype=float).copy()
    q = np.asarray(expert_warmup_proba, dtype=float)
    q = np.clip(q, 1e-06, 1.0 - 1e-06)
    rest_id = label_to_id['Rest']
    warmup_id = label_to_id['Warmup']
    rw_mass = p[:, rest_id] + p[:, warmup_id]
    safe_mass = np.maximum(rw_mass, 1e-12)
    main_warmup_frac = p[:, warmup_id] / safe_mass
    main_uncertainty = 1.0 - 2.0 * np.abs(main_warmup_frac - 0.5)
    main_uncertainty = np.clip(main_uncertainty, 0.0, 1.0)
    expert_conf = 2.0 * np.abs(q - 0.5)
    expert_conf = np.clip(expert_conf, 0.0, 1.0)
    row_alpha = float(np.clip(alpha, 0.0, 1.0)) * expert_conf * (0.3 + 0.7 * main_uncertainty)
    row_alpha = np.where(expert_conf >= float(min_expert_conf), row_alpha, 0.0)
    row_alpha = np.where(rw_mass >= float(min_rw_mass), row_alpha, 0.0)
    row_alpha = np.clip(row_alpha, 0.0, 1.0)
    p[:, rest_id] = (1.0 - row_alpha) * p[:, rest_id] + row_alpha * rw_mass * (1.0 - q)
    p[:, warmup_id] = (1.0 - row_alpha) * p[:, warmup_id] + row_alpha * rw_mass * q
    row_sum = p.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return p / row_sum

def apply_recovery_trend_boost(val_ds: pd.DataFrame, proba4: np.ndarray, label_to_id: Dict[str, int], start_frac: float=0.68, strength: float=0.28, min_fall_score: float=0.05) -> np.ndarray:
    p = np.asarray(proba4, dtype=float).copy()
    if len(p) == 0:
        return p
    rest_id = label_to_id['Rest']
    warmup_id = label_to_id['Warmup']
    exercise_id = label_to_id['Exercise']
    recovery_id = label_to_id['Recovery']
    ordered = val_ds.copy().reset_index(drop=True)
    order = np.argsort(pd.to_datetime(ordered['t_start']).to_numpy())
    n = len(order)
    start_frac = float(np.clip(start_frac, 0.0, 1.0))
    strength = float(np.clip(strength, 0.0, 1.0))
    for pos, idx in enumerate(order):
        frac = pos / max(1, n - 1)
        if frac < start_frac:
            continue
        row = ordered.iloc[idx]
        fall_components = []
        for c in ['HR__stage__delta', 'HR__stage__slope', 'HR__stage__mean__traj_prev_diff', 'Temp__stage__delta', 'Temp__stage__slope', 'Temp__stage__mean__traj_prev_diff', 'Temp_HR__stage__delta_sum', 'Temp_HR__stage__slope_sum', 'Temp_HR__late_minus_early_sum', 'SR_zero_stage_delta', 'SR_zero_stage_slope', 'SR_late_minus_early_mean']:
            if c in row.index:
                v = float(pd.to_numeric(row[c], errors='coerce'))
                if np.isfinite(v):
                    fall_components.append(max(0.0, -v))
        if len(fall_components) == 0:
            continue
        fall_score = float(np.mean(fall_components))
        if fall_score < float(min_fall_score):
            continue
        fall_conf = float(1.0 - np.exp(-fall_score))
        boost = strength * fall_conf
        movable_mass = p[idx, exercise_id] + 0.5 * p[idx, warmup_id] + 0.2 * p[idx, rest_id]
        p[idx, recovery_id] += boost * movable_mass
        p[idx, exercise_id] *= 1.0 - 0.7 * boost
        p[idx, warmup_id] *= 1.0 - 0.5 * boost
        p[idx, rest_id] *= 1.0 - 0.2 * boost
    row_sum = p.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return p / row_sum

def estimate_protocol_transition_fractions(train_ds: pd.DataFrame, label_to_id: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, int]:
    fallback = np.array([0.14, 0.36, 0.86], dtype=float)
    if train_ds is None or len(train_ds) == 0:
        return (fallback, np.full(3, np.nan, dtype=float), 0)
    time_col = 't_start' if 't_start' in train_ds.columns else 't_center' if 't_center' in train_ds.columns else None
    rows: List[List[float]] = []
    for _, g in train_ds.groupby('subject_id', sort=False):
        if time_col is not None:
            g = g.sort_values(time_col).reset_index(drop=True)
        else:
            g = g.reset_index(drop=True)
        n = len(g)
        if n < 4:
            continue
        y = np.array([label_to_id.get(str(x), -1) for x in g['label'].astype(str)], dtype=int)
        starts: List[float] = []
        complete = True
        for class_id in [1, 2, 3]:
            idx = np.flatnonzero(y == class_id)
            if len(idx) == 0:
                complete = False
                break
            starts.append(float((idx[0] + 0.5) / n))
        if complete:
            rows.append(starts)
    if len(rows) == 0:
        return (fallback, np.full(3, np.nan, dtype=float), 0)
    arr = np.asarray(rows, dtype=float)
    median = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - median[None, :]), axis=0)
    median = np.where(np.isfinite(median), median, fallback)
    b1 = float(np.clip(median[0], 0.03, 0.45))
    b2 = float(np.clip(median[1], b1 + 0.03, 0.75))
    b3 = float(np.clip(median[2], b2 + 0.05, 0.97))
    median = np.array([b1, b2, b3], dtype=float)
    return (median, mad.astype(float), int(len(rows)))

def build_protocol_position_prior(n_rows: int, transition_fractions: np.ndarray, width: float=0.01, probability_floor: float=0.0001) -> np.ndarray:
    n_rows = int(n_rows)
    if n_rows <= 0:
        return np.zeros((0, 4), dtype=float)
    bounds = np.asarray(transition_fractions, dtype=float).reshape(-1)
    if len(bounds) != 3 or not np.all(np.isfinite(bounds)):
        bounds = np.array([0.14, 0.36, 0.86], dtype=float)
    edges = np.array([0.0, bounds[0], bounds[1], bounds[2], 1.0], dtype=float)
    frac = (np.arange(n_rows, dtype=float) + 0.5) / float(n_rows)
    width = max(float(width), 1e-06)
    floor = max(float(probability_floor), 1e-12)
    prior = np.zeros((n_rows, 4), dtype=float)
    for class_id in range(4):
        lo = float(edges[class_id])
        hi = float(edges[class_id + 1])
        distance = np.where(frac < lo, lo - frac, np.where(frac > hi, frac - hi, 0.0))
        prior[:, class_id] = np.exp(-0.5 * (distance / width) ** 2)
    prior = np.maximum(prior, floor)
    row_sum = prior.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return prior / row_sum

def fuse_training_protocol_prior(proba4_sorted: np.ndarray, train_ds: pd.DataFrame, label_to_id: Dict[str, int], strength: float=2.0, width: float=0.01, probability_floor: float=0.0001, max_transition_mad: float=0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    p = np.asarray(proba4_sorted, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError(f'Expected proba4_sorted with shape (N,4), got {p.shape}')
    bounds, mad, n_complete = estimate_protocol_transition_fractions(train_ds=train_ds, label_to_id=label_to_id)
    prior = build_protocol_position_prior(n_rows=len(p), transition_fractions=bounds, width=width, probability_floor=probability_floor)
    finite_mad = mad[np.isfinite(mad)]
    if len(finite_mad) == 0 or float(max_transition_mad) <= 0.0:
        confidence = 1.0
    else:
        confidence = float(np.clip(1.0 - float(np.max(finite_mad)) / float(max_transition_mad), 0.0, 1.0))
    effective_strength = max(0.0, float(strength)) * confidence
    log_fused = np.log(np.clip(p, 1e-12, 1.0)) + effective_strength * np.log(np.clip(prior, 1e-12, 1.0))
    log_fused -= np.max(log_fused, axis=1, keepdims=True)
    fused = np.exp(log_fused)
    row_sum = fused.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    fused = fused / row_sum
    return (fused, prior, bounds, mad, float(effective_strength), int(n_complete))

def ordered_decode_stage_level(proba4: np.ndarray, stay_bonus: float=0.25, forward_penalty: float=0.1, skip_penalty: float=0.35, forbidden_penalty: float=1000000.0, force_end_recovery: bool=True) -> np.ndarray:
    eps = 1e-12
    logp = np.log(np.clip(proba4, eps, 1.0))
    n, k = logp.shape
    if k != 4:
        raise ValueError(f'Expected proba4 shape (N,4), got {proba4.shape}')
    trans = np.full((4, 4), -forbidden_penalty, dtype=float)
    for i in range(4):
        trans[i, i] = stay_bonus
    trans[0, 1] = -forward_penalty
    trans[1, 2] = -forward_penalty
    trans[2, 3] = -forward_penalty
    trans[0, 2] = -skip_penalty
    trans[1, 3] = -skip_penalty
    dp = np.full((n, 4), -1e+18, dtype=float)
    ptr = np.zeros((n, 4), dtype=int)
    dp[0, :] = logp[0, :]
    dp[0, 0] += 0.8
    for t in range(1, n):
        for j in range(4):
            scores = dp[t - 1, :] + trans[:, j]
            ptr[t, j] = int(np.argmax(scores))
            dp[t, j] = scores[ptr[t, j]] + logp[t, j]
    y = np.zeros(n, dtype=int)
    y[-1] = 3 if force_end_recovery else int(np.argmax(dp[-1, :]))
    for t in range(n - 2, -1, -1):
        y[t] = ptr[t + 1, y[t + 1]]
    return y

def apply_sr_rules(val_ds: pd.DataFrame, proba4: np.ndarray, y_pred: np.ndarray, label_to_id: Dict[str, int], sr_presence_thr: float=0.05, sr_falling_slope_thr: float=-0.02, sr_rising_slope_thr: float=0.02) -> np.ndarray:
    y = np.asarray(y_pred, dtype=int).copy()
    rest_id = label_to_id['Rest']
    warmup_id = label_to_id['Warmup']
    exercise_id = label_to_id['Exercise']
    recovery_id = label_to_id['Recovery']
    active_ids = [warmup_id, exercise_id, recovery_id]
    for i, row in val_ds.reset_index(drop=True).iterrows():
        sr_presence = float(row.get('SR_presence_frac', row.get('SR_valid__stage__mean', 0.0)))
        sr_slope = float(row.get('SR_zero_stage_slope', row.get('SR__stage__slope', 0.0)))
        sr_delta = float(row.get('SR_zero_stage_delta', row.get('SR__stage__delta', 0.0)))
        sr_late_early = float(row.get('SR_late_minus_early_mean', 0.0))
        if y[i] == rest_id and sr_presence >= sr_presence_thr:
            if sr_slope <= sr_falling_slope_thr or sr_delta < 0 or sr_late_early < 0:
                y[i] = recovery_id
            elif sr_slope >= sr_rising_slope_thr or sr_delta > 0 or sr_late_early > 0:
                y[i] = warmup_id if proba4[i, warmup_id] >= proba4[i, exercise_id] else exercise_id
            else:
                best_active_local = int(np.argmax(proba4[i, active_ids]))
                y[i] = active_ids[best_active_local]
        if sr_presence >= 0.5 and (sr_slope <= sr_falling_slope_thr or sr_late_early < -1e-06):
            if y[i] in [rest_id, warmup_id] or proba4[i, recovery_id] >= 0.2:
                y[i] = recovery_id
    return y

def apply_delayed_hr_temp_rescue_v5(val_ds: pd.DataFrame, y_pred: np.ndarray, temp_warmup_proba: np.ndarray, label_to_id: Dict[str, int], detected_hr_delay: int, hr_delay_trigger: int=12, temp_threshold: float=0.55, min_temp_delta: float=0.05, min_temp_slope: float=0.0005, hold_monotonic: bool=True) -> np.ndarray:
    y = np.asarray(y_pred, dtype=int).copy()
    if len(y) == 0:
        return y
    if int(detected_hr_delay) < int(hr_delay_trigger):
        return y
    rest_id = label_to_id['Rest']
    warmup_id = label_to_id['Warmup']
    exercise_id = label_to_id['Exercise']
    recovery_id = label_to_id['Recovery']
    ordered = val_ds.reset_index(drop=True).copy()
    order = np.argsort(pd.to_datetime(ordered['t_start']).to_numpy())
    y_sorted = y[order].copy()
    p_temp = np.asarray(temp_warmup_proba, dtype=float)[order]
    first_exercise_pos = len(y_sorted)
    hit = np.flatnonzero(y_sorted == exercise_id)
    if len(hit) > 0:
        first_exercise_pos = int(hit[0])
    warmup_started = False
    for pos in range(first_exercise_pos):
        idx = int(order[pos])
        current = int(y_sorted[pos])
        row = ordered.iloc[idx]
        temp_delta = float(row.get('Temp__stage__delta', 0.0))
        temp_slope = float(row.get('Temp__stage__slope', 0.0))
        temp_rising = np.isfinite(temp_delta) and temp_delta >= float(min_temp_delta) or (np.isfinite(temp_slope) and temp_slope >= float(min_temp_slope))
        if current in [rest_id, recovery_id] and p_temp[pos] >= float(temp_threshold) and temp_rising:
            current = warmup_id
        if hold_monotonic:
            if current == warmup_id:
                warmup_started = True
            elif warmup_started and current == rest_id and (p_temp[pos] >= float(temp_threshold) - 0.1):
                current = warmup_id
        y_sorted[pos] = current
    y_out = y.copy()
    y_out[order] = y_sorted
    return y_out

def force_tail_recovery_windows_v5(val_ds: pd.DataFrame, y_pred: np.ndarray, label_to_id: Dict[str, int], n_tail_windows: int=0) -> np.ndarray:
    y = np.asarray(y_pred, dtype=int).copy()
    n_tail = int(n_tail_windows)
    if len(y) == 0 or n_tail <= 0:
        return y
    recovery_id = label_to_id['Recovery']
    order = np.argsort(pd.to_datetime(val_ds['t_start']).to_numpy())
    tail_order = order[max(0, len(order) - n_tail):]
    y[tail_order] = recovery_id
    return y

def plot_fig4e_like(summary_df: pd.DataFrame, out_png: str | Path, out_pdf: str | Path, reference_y: float=90.0, title: str=''):
    df = summary_df.copy().sort_values('subject_index').reset_index(drop=True)
    x = df['subject_index'].to_numpy(dtype=int)
    y = df['macro_f1_percent'].to_numpy(dtype=float)
    fig_width = max(6.2, 0.26 * len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))
    ax.bar(x, y, width=0.62, facecolor='#eaf2f8', edgecolor='black', linewidth=1.0)
    ax.axhline(reference_y, color='red', linestyle=(0, (5, 5)), linewidth=1.2)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Macro $F_1$-score (%)', fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=7)
    ax.tick_params(axis='y', labelsize=10)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)

def make_subject_metric_boxplot_data(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = summary_df.copy().sort_values('subject_index').reset_index(drop=True)
    if 'accuracy_percent' not in df.columns:
        if 'micro_f1_percent' not in df.columns:
            raise ValueError('summary_df must contain accuracy_percent or micro_f1_percent.')
        df['accuracy_percent'] = df['micro_f1_percent']
    metric_specs = [('Accuracy', 'accuracy_percent'), ('Macro-F1', 'macro_f1_percent'), ('Precision', 'macro_precision_percent'), ('Recall', 'macro_recall_percent')]
    wide_df = df.loc[:, ['subject_index', 'val_subject'] + [col for _, col in metric_specs]].copy()
    long_df = wide_df.rename(columns={col: label for label, col in metric_specs}).melt(id_vars=['subject_index', 'val_subject'], value_vars=[label for label, _ in metric_specs], var_name='metric', value_name='value_percent')
    long_df['metric'] = pd.Categorical(long_df['metric'], categories=[label for label, _ in metric_specs], ordered=True)
    long_df = long_df.sort_values(['metric', 'subject_index']).reset_index(drop=True)
    return (wide_df, long_df)

def plot_subject_metric_boxplot(summary_df: pd.DataFrame, out_png: str | Path, out_pdf: str | Path, out_wide_csv: Optional[str | Path]=None, out_long_csv: Optional[str | Path]=None, panel_label: str='k', annotation: str='Subject-independent'):
    wide_df, long_df = make_subject_metric_boxplot_data(summary_df)
    if out_wide_csv is not None:
        wide_df.to_csv(out_wide_csv, index=False, encoding='utf-8-sig')
    if out_long_csv is not None:
        long_df.to_csv(out_long_csv, index=False, encoding='utf-8-sig')
    metric_specs = [('Accuracy', 'accuracy_percent'), ('Macro-F1', 'macro_f1_percent'), ('Precision', 'macro_precision_percent'), ('Recall', 'macro_recall_percent')]
    labels = [label for label, _ in metric_specs]
    values = [pd.to_numeric(wide_df[col], errors='coerce').dropna().to_numpy(dtype=float) for _, col in metric_specs]
    finite_values = np.concatenate([v for v in values if len(v) > 0])
    if len(finite_values) == 0:
        raise ValueError('No finite metric values are available for the boxplot.')
    y_min_data = float(np.nanmin(finite_values))
    y_max_data = float(np.nanmax(finite_values))
    y_min = 50.0 if y_min_data >= 50.0 else max(0.0, np.floor((y_min_data - 5.0) / 10.0) * 10.0)
    y_max = 110.0 if y_max_data <= 110.0 else np.ceil((y_max_data + 5.0) / 10.0) * 10.0
    fig, ax = plt.subplots(figsize=(4.7, 4.0))
    box = ax.boxplot(values, labels=labels, widths=0.58, patch_artist=True, showfliers=False, medianprops={'color': '#333333', 'linewidth': 1.3}, boxprops={'edgecolor': '#555555', 'linewidth': 1.1}, whiskerprops={'color': '#555555', 'linewidth': 1.1}, capprops={'color': '#555555', 'linewidth': 1.1})
    colors = ['#c98f8f', '#78add2', '#f0b88c', '#bdbdbd']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    rng = np.random.default_rng(2024)
    for i, v in enumerate(values, start=1):
        if len(v) == 0:
            continue
        x = rng.normal(loc=i, scale=0.045, size=len(v))
        ax.scatter(x, v, s=13, marker='o', color='black', alpha=0.85, linewidths=0.0, zorder=3)
    ax.set_ylabel('Value (%)', fontsize=12)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0.45, len(labels) + 0.55)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ticks = [t for t in [60, 80, 100] if y_min <= t <= y_max]
    if len(ticks) >= 2:
        ax.set_yticks(ticks)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    if panel_label:
        ax.text(-0.07, 1.02, panel_label, transform=ax.transAxes, ha='left', va='bottom', fontsize=14, fontweight='bold')
    if annotation:
        ax.text(0.98, 0.18, annotation, transform=ax.transAxes, ha='right', va='center', fontsize=12)
    fig.tight_layout(pad=1.0)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)

def smooth_probabilities_in_time(proba: np.ndarray, kernel: List[float]) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or len(p) == 0:
        return p.copy()
    k = np.asarray(kernel, dtype=float)
    if len(k) == 0 or not np.all(np.isfinite(k)) or np.sum(np.abs(k)) <= 0:
        return p.copy()
    k = k / np.sum(k)
    radius = len(k) // 2
    out = np.zeros_like(p, dtype=float)
    for i in range(len(p)):
        acc = np.zeros(p.shape[1], dtype=float)
        weight_sum = 0.0
        for j, w in enumerate(k):
            idx = i + j - radius
            if 0 <= idx < len(p):
                acc += float(w) * p[idx]
                weight_sum += float(w)
        if weight_sum <= 0:
            out[i] = p[i]
        else:
            out[i] = acc / weight_sum
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return out / row_sum

def fix_isolated_label_flips(y_pred: np.ndarray, proba: np.ndarray, confidence_threshold: float=0.65) -> np.ndarray:
    y = np.asarray(y_pred, dtype=int).copy()
    p = np.asarray(proba, dtype=float)
    if len(y) < 3 or p.ndim != 2:
        return y
    top_conf = np.max(p, axis=1)
    thr = float(confidence_threshold)
    for i in range(1, len(y) - 1):
        if y[i - 1] == y[i + 1] and y[i] != y[i - 1] and (top_conf[i] < thr):
            y[i] = y[i - 1]
    return y

def format_percent_text(x: float) -> str:
    x = float(x)
    if abs(x - round(x)) < 1e-06:
        return f'{int(round(x))}%'
    return f'{x:.1f}%'

def build_confusion_matrix_tables(y_true: np.ndarray, y_pred: np.ndarray, label_order: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_ids = list(range(len(label_order)))
    cm_counts = confusion_matrix(y_true, y_pred, labels=label_ids)
    row_sums = cm_counts.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums <= 0.0] = 1.0
    cm_percent = cm_counts / row_sums * 100.0
    counts_df = pd.DataFrame(cm_counts, index=label_order, columns=label_order)
    counts_df.index.name = 'true_label'
    counts_df.columns.name = 'pred_label'
    percent_df = pd.DataFrame(cm_percent, index=label_order, columns=label_order)
    percent_df.index.name = 'true_label'
    percent_df.columns.name = 'pred_label'
    long_rows = []
    for true_lab in label_order:
        for pred_lab in label_order:
            long_rows.append({'true_label': true_lab, 'pred_label': pred_lab, 'count': int(counts_df.loc[true_lab, pred_lab]), 'percent': float(percent_df.loc[true_lab, pred_lab])})
    long_df = pd.DataFrame(long_rows)
    return (counts_df, percent_df, long_df)

def plot_confusion_matrix_4class(y_true: np.ndarray, y_pred: np.ndarray, label_order: List[str], out_png: str | Path, out_pdf: str | Path, out_counts_csv: Optional[str | Path]=None, out_percent_csv: Optional[str | Path]=None, out_long_csv: Optional[str | Path]=None):
    counts_df, percent_df, long_df = build_confusion_matrix_tables(y_true=y_true, y_pred=y_pred, label_order=label_order)
    if out_counts_csv is not None:
        counts_df.to_csv(out_counts_csv, encoding='utf-8-sig')
    if out_percent_csv is not None:
        percent_df.to_csv(out_percent_csv, encoding='utf-8-sig')
    if out_long_csv is not None:
        long_df.to_csv(out_long_csv, index=False, encoding='utf-8-sig')
    display_order = list(label_order)
    display_percent = percent_df.reindex(index=display_order, columns=display_order)
    short_name = {'Rest': 'Rest', 'Warmup': 'WU', 'Exercise': 'EX', 'Recovery': 'Rec'}
    fig, ax = plt.subplots(figsize=(4.65, 4.25))
    im = ax.imshow(display_percent.to_numpy(dtype=float), cmap='Reds', vmin=0.0, vmax=100.0)
    ax.set_xticks(np.arange(len(display_order)))
    ax.set_yticks(np.arange(len(display_order)))
    ax.set_xticklabels([short_name.get(x, str(x)) for x in display_order], fontsize=12)
    ax.set_yticklabels([short_name.get(y, str(y)) for y in display_order], fontsize=12)
    for i in range(display_percent.shape[0]):
        for j in range(display_percent.shape[1]):
            pct = float(display_percent.iloc[i, j])
            is_correct = i == j
            ax.text(j, i, format_percent_text(pct), ha='center', va='center', color='white' if is_correct else '#6f1d1b', fontsize=12, fontweight='bold' if is_correct else 'normal')
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    overall_accuracy = float(np.mean(np.asarray(y_true, dtype=int) == np.asarray(y_pred, dtype=int)) * 100.0)
    ax.set_title(f'Accuracy = {overall_accuracy:.1f}%', fontsize=15, loc='right', pad=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_visible(True)
    cbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.04)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)
    return (counts_df, percent_df, long_df)

def compute_multiclass_shap_values_for_model(clf: XGBClassifier, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    booster = clf.get_booster()
    dmat = xgb.DMatrix(X, feature_names=[str(c) for c in X.columns])
    contrib = booster.predict(dmat, pred_contribs=True)
    contrib = np.asarray(contrib)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = int(getattr(clf, 'n_classes_', 4))
    if contrib.ndim == 2:
        expected = n_classes * (n_features + 1)
        if contrib.shape[1] != expected:
            raise ValueError(f'Unexpected pred_contribs shape {contrib.shape}, expected second dim {expected}.')
        contrib = contrib.reshape(n_samples, n_classes, n_features + 1)
    if contrib.ndim != 3 or contrib.shape[2] != n_features + 1:
        raise ValueError(f'Unexpected multiclass SHAP tensor shape: {contrib.shape}, expected (n_samples, n_classes, n_features + 1).')
    shap_values = contrib[:, :, :-1]
    base_values = contrib[:, :, -1]
    return (shap_values.astype(np.float32), base_values.astype(np.float32))

def compute_multiclass_shap_values_for_ensemble(clfs: List[XGBClassifier], X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if len(clfs) == 0:
        raise ValueError('No models were provided for SHAP computation.')
    shap_list = []
    base_list = []
    for clf in clfs:
        shap_values, base_values = compute_multiclass_shap_values_for_model(clf, X)
        shap_list.append(shap_values)
        base_list.append(base_values)
    shap_mean = np.mean(np.stack(shap_list, axis=0), axis=0)
    base_mean = np.mean(np.stack(base_list, axis=0), axis=0)
    return (shap_mean.astype(np.float32), base_mean.astype(np.float32))

def save_multiclass_shap_payload_npz(out_npz: str | Path, shap_values: np.ndarray, base_values: np.ndarray, feature_values: np.ndarray, feature_names: List[str], class_names: List[str], sample_uid: List[str], subject_id: List[str], true_label: List[str], pred_label: List[str]):
    np.savez_compressed(out_npz, shap_values=np.asarray(shap_values, dtype=np.float32), base_values=np.asarray(base_values, dtype=np.float32), feature_values=np.asarray(feature_values, dtype=np.float32), feature_names=np.asarray(feature_names, dtype=object), class_names=np.asarray(class_names, dtype=object), sample_uid=np.asarray(sample_uid, dtype=object), subject_id=np.asarray(subject_id, dtype=object), true_label=np.asarray(true_label, dtype=object), pred_label=np.asarray(pred_label, dtype=object))

def class_name_to_slug(class_name: str) -> str:
    mapping = {'Rest': 'rest', 'Warmup': 'warmup', 'Exercise': 'exercise', 'Recovery': 'recovery'}
    return mapping.get(str(class_name), str(class_name).strip().lower().replace(' ', '_'))

def remove_legacy_class_shap_dirs(out_shap: str | Path, class_names: List[str]) -> None:
    root = Path(out_shap)
    for class_name in class_names:
        class_dir = root / class_name_to_slug(class_name)
        if class_dir.is_dir():
            shutil.rmtree(class_dir)

def class_name_to_plot_title(class_name: str) -> str:
    if str(class_name) == 'Recovery':
        return 'Recover'
    return str(class_name)
SENSOR_DISPLAY_ORDER = ['HR', 'GLU', 'LA', 'AA', 'UA', 'PH', 'ST', 'GSR', 'SR', 'SIC']
CLASS_COLOR_MAP = {'Rest': '#e67c6c', 'Warmup': '#e2b06d', 'Exercise': '#73a7d1', 'Recovery': '#8bb7ae'}

def get_sensor_prefix_map() -> Dict[str, List[str]]:
    return {'HR': ['HR'], 'GLU': ['GLU', 'Glu', 'Glucose'], 'LA': ['LA', 'Lac'], 'AA': ['AA'], 'UA': ['UA'], 'PH': ['PH', 'pH'], 'ST': ['Temp', 'ST'], 'GSR': ['GSR', 'GRS'], 'SR': ['SR'], 'SIC': ['SIC']}

def _feature_matches_alias(feature_name: str, alias: str) -> bool:
    f = str(feature_name)
    a = str(alias)
    return f == a or f.startswith(a + '__') or f.startswith(a + '_')

def get_sensor_feature_index_map(feature_names: List[str]) -> Dict[str, List[int]]:
    prefix_map = get_sensor_prefix_map()
    sensor_to_indices = {sensor: [] for sensor in SENSOR_DISPLAY_ORDER}
    matching_order = ['HR', 'GLU', 'LA', 'AA', 'UA', 'PH', 'SIC', 'ST', 'GSR', 'SR']
    for feat_idx, feat_name in enumerate(feature_names):
        feat_name = str(feat_name)
        if feat_name.startswith('SR_valid') or feat_name.startswith('SIC_valid'):
            continue
        for sensor in matching_order:
            aliases = prefix_map.get(sensor, [])
            if any((_feature_matches_alias(feat_name, alias) for alias in aliases)):
                sensor_to_indices[sensor].append(feat_idx)
                break
    return sensor_to_indices

def build_shap_sensor_feature_mapping_table(shap_payloads: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for payload_i, payload in enumerate(shap_payloads, start=1):
        feature_names = [str(x) for x in payload.get('feature_names', [])]
        sensor_to_indices = get_sensor_feature_index_map(feature_names)
        subject_values = payload.get('subject_id', [])
        subject_id = str(subject_values[0]) if len(subject_values) > 0 else f'payload_{payload_i}'
        assigned = set()
        for sensor in SENSOR_DISPLAY_ORDER:
            idxs = sensor_to_indices.get(sensor, [])
            for idx in idxs:
                assigned.add(idx)
                rows.append({'payload_index': int(payload_i), 'subject_id': subject_id, 'sensor': sensor, 'feature_index': int(idx), 'feature_name': feature_names[idx]})
        for idx, feature_name in enumerate(feature_names):
            if idx not in assigned:
                rows.append({'payload_index': int(payload_i), 'subject_id': subject_id, 'sensor': 'UNASSIGNED', 'feature_index': int(idx), 'feature_name': feature_name})
    return pd.DataFrame(rows)

def build_sensor_level_shap_tables(shap_payloads: List[Dict[str, object]], class_names: List[str], sensor_order: List[str] | None=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if sensor_order is None:
        sensor_order = list(SENSOR_DISPLAY_ORDER)
    plot_rows = []
    for payload in shap_payloads:
        shap_values = np.asarray(payload['shap_values'], dtype=float)
        feature_values = np.asarray(payload['feature_values'], dtype=float)
        feature_names = [str(x) for x in payload['feature_names']]
        sensor_to_indices = get_sensor_feature_index_map(feature_names)
        sample_uid = [str(x) for x in payload['sample_uid']]
        subject_id = [str(x) for x in payload['subject_id']]
        true_label = [str(x) for x in payload['true_label']]
        pred_label = [str(x) for x in payload['pred_label']]
        for class_idx, class_name in enumerate(class_names):
            for sensor in sensor_order:
                idxs = sensor_to_indices.get(sensor, [])
                if len(idxs) > 0:
                    shap_vec = np.sum(shap_values[:, class_idx, idxs], axis=1)
                    feat_mat = feature_values[:, idxs]
                    feat_vec = np.nanmean(feat_mat, axis=1)
                else:
                    shap_vec = np.zeros(shap_values.shape[0], dtype=float)
                    feat_vec = np.full(shap_values.shape[0], np.nan, dtype=float)
                for i in range(shap_values.shape[0]):
                    plot_rows.append({'class_name': str(class_name), 'sensor': str(sensor), 'summary_rank': int(sensor_order.index(sensor) + 1), 'sample_uid': sample_uid[i], 'subject_id': subject_id[i], 'true_label': true_label[i], 'pred_label': pred_label[i], 'shap_value': float(shap_vec[i]), 'feature_value': float(feat_vec[i]) if np.isfinite(feat_vec[i]) else np.nan})
    plot_df = pd.DataFrame(plot_rows)
    if len(plot_df) == 0:
        return (pd.DataFrame(), plot_df)
    feature_count_map = {}
    for payload in shap_payloads:
        feature_names = [str(x) for x in payload['feature_names']]
        sensor_to_indices = get_sensor_feature_index_map(feature_names)
        for sensor in sensor_order:
            feature_count_map[sensor] = max(int(feature_count_map.get(sensor, 0)), int(len(sensor_to_indices.get(sensor, []))))
    importance_df = plot_df.groupby(['class_name', 'sensor', 'summary_rank'], as_index=False).agg(mean_abs_shap=('shap_value', lambda s: float(np.mean(np.abs(pd.to_numeric(s, errors='coerce'))))), sum_abs_shap=('shap_value', lambda s: float(np.sum(np.abs(pd.to_numeric(s, errors='coerce'))))), n_points=('shap_value', 'size')).sort_values(['class_name', 'mean_abs_shap', 'summary_rank'], ascending=[True, False, True]).reset_index(drop=True)
    importance_df['mapped_engineered_feature_count'] = importance_df['sensor'].map(feature_count_map).fillna(0).astype(int)
    importance_df['decision_rank'] = importance_df.groupby('class_name')['mean_abs_shap'].rank(method='first', ascending=False).astype(int)
    return (importance_df, plot_df)

def build_sensor_stacked_importance_table(sensor_importance_df: pd.DataFrame, class_names: List[str]) -> pd.DataFrame:
    if sensor_importance_df is None or len(sensor_importance_df) == 0:
        return pd.DataFrame()
    pivot_df = sensor_importance_df.pivot_table(index='sensor', columns='class_name', values='mean_abs_shap', aggfunc='first').fillna(0.0)
    for class_name in class_names:
        if class_name not in pivot_df.columns:
            pivot_df[class_name] = 0.0
    pivot_df = pivot_df.reindex(columns=class_names).fillna(0.0)
    pivot_df['total_mean_abs_shap'] = pivot_df[class_names].sum(axis=1)
    pivot_df = pivot_df.sort_values('total_mean_abs_shap', ascending=False).reset_index()
    return pivot_df

def plot_sensor_stacked_importance(stacked_df: pd.DataFrame, class_names: List[str], out_png: str | Path, out_pdf: str | Path, out_svg: str | Path):
    if stacked_df is None or len(stacked_df) == 0:
        raise ValueError('stacked_df is empty; cannot draw stacked importance chart.')
    sensors = stacked_df['sensor'].astype(str).tolist()
    y = np.arange(len(sensors), dtype=float)
    fig_height = max(4.8, 0.48 * len(sensors) + 0.8)
    fig, ax = plt.subplots(figsize=(5.2, fig_height))
    left = np.zeros(len(sensors), dtype=float)
    for class_name in class_names:
        vals = pd.to_numeric(stacked_df[class_name], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.78, color=CLASS_COLOR_MAP.get(class_name, None), edgecolor='black', linewidth=0.9, label=class_name)
        left += vals
    for yi in y:
        ax.axhline(yi + 0.5, color='#b7b7b7', linestyle=(0, (4, 4)), linewidth=1.0, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(sensors, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel('mean(|SHAP value|)', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    leg = ax.legend(loc='lower right', frameon=True, fontsize=10, fancybox=False, framealpha=1.0, borderpad=0.4, handlelength=1.8, handleheight=0.8, handletextpad=0.5)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close(fig)

def plot_single_class_shap_summary(class_name: str, plot_df: pd.DataFrame, importance_df: pd.DataFrame, out_png: str | Path, out_pdf: str | Path, out_svg: str | Path, sensor_order: List[str] | None=None):
    if sensor_order is None:
        sensor_order = list(SENSOR_DISPLAY_ORDER)
    cls_plot = plot_df[plot_df['class_name'] == class_name].copy()
    if len(cls_plot) == 0:
        raise ValueError(f'No SHAP summary data available for class={class_name}.')
    shap_abs = np.abs(pd.to_numeric(cls_plot['shap_value'], errors='coerce').to_numpy(dtype=float))
    shap_abs = shap_abs[np.isfinite(shap_abs)]
    x_lim = float(np.nanpercentile(shap_abs, 99)) if len(shap_abs) > 0 else 1.0
    x_lim = max(1.0, 1.08 * x_lim)
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    cmap = plt.cm.cool
    scatter_for_cbar = None
    rng = np.random.default_rng(2024)
    for y_base in range(len(sensor_order)):
        ax.axhline(y=y_base, color='#9e9e9e', linestyle=(0, (3, 4)), linewidth=1.0, zorder=0)
    for y_base, sensor in enumerate(sensor_order):
        sub = cls_plot[cls_plot['sensor'] == sensor].copy()
        if len(sub) == 0:
            continue
        feat_vals = pd.to_numeric(sub['feature_value'], errors='coerce').to_numpy(dtype=float)
        finite_mask = np.isfinite(feat_vals)
        if np.any(finite_mask):
            lo, hi = np.nanpercentile(feat_vals[finite_mask], [5, 95])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = (float(np.nanmin(feat_vals[finite_mask])), float(np.nanmax(feat_vals[finite_mask])))
            denom = hi - lo if hi > lo else 1.0
            c_vals = np.clip((feat_vals - lo) / denom, 0.0, 1.0)
        else:
            c_vals = np.full(len(sub), 0.5, dtype=float)
        y = y_base + rng.normal(loc=0.0, scale=0.08, size=len(sub))
        scatter_for_cbar = ax.scatter(pd.to_numeric(sub['shap_value'], errors='coerce').to_numpy(dtype=float), y, c=c_vals, cmap=cmap, vmin=0.0, vmax=1.0, s=11, alpha=0.88, linewidths=0.0, zorder=2)
    ax.axvline(0.0, color='#c4c4c4', linewidth=1.0, zorder=1)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_yticks(np.arange(len(sensor_order)))
    ax.set_yticklabels(sensor_order, fontsize=11)
    ax.invert_yaxis()
    ax.set_title(class_name_to_plot_title(class_name), fontsize=14)
    ax.set_xlabel('SHAP value', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if scatter_for_cbar is not None:
        cbar = fig.colorbar(scatter_for_cbar, ax=ax, fraction=0.045, pad=0.05)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(['Low', 'High'])
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Feature value', rotation=270, labelpad=14, fontsize=11)
    fig.tight_layout(pad=0.7)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close(fig)

def build_single_class_decision_data(shap_payloads: List[Dict[str, object]], class_name: str, importance_df: pd.DataFrame, class_names: List[str], sensor_order: List[str] | None=None) -> pd.DataFrame:
    if sensor_order is None:
        sensor_order = list(SENSOR_DISPLAY_ORDER)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    if class_name not in class_to_idx:
        raise ValueError(f'Unknown class_name={class_name}')
    class_idx = class_to_idx[class_name]
    cls_imp = importance_df[importance_df['class_name'] == class_name].sort_values(['mean_abs_shap', 'summary_rank'], ascending=[False, True]).reset_index(drop=True)
    ranked = [str(s) for s in cls_imp['sensor'].astype(str).tolist() if str(s) in sensor_order]
    ordered_sensors_top = ranked + [str(s) for s in sensor_order if str(s) not in ranked]
    decision_rows = []
    for payload in shap_payloads:
        feature_names = [str(x) for x in payload['feature_names']]
        sensor_to_indices = get_sensor_feature_index_map(feature_names)
        shap_values = np.asarray(payload['shap_values'], dtype=float)[:, class_idx, :]
        base_values = np.asarray(payload['base_values'], dtype=float)[:, class_idx]
        sample_uid = [str(x) for x in payload['sample_uid']]
        subject_id = [str(x) for x in payload['subject_id']]
        true_label = [str(x) for x in payload['true_label']]
        pred_label = [str(x) for x in payload['pred_label']]
        sensor_contrib = {}
        for sensor in ordered_sensors_top:
            idxs = sensor_to_indices.get(sensor, [])
            if len(idxs) > 0:
                sensor_contrib[sensor] = np.sum(shap_values[:, idxs], axis=1)
            else:
                sensor_contrib[sensor] = np.zeros(shap_values.shape[0], dtype=float)
        selected_sum = np.zeros(shap_values.shape[0], dtype=float)
        for sensor in ordered_sensors_top:
            selected_sum += sensor_contrib[sensor]
        total_output = base_values + np.sum(shap_values, axis=1)
        other_contrib = total_output - (base_values + selected_sum)
        start_value = base_values + other_contrib
        reversed_sensors = list(reversed(ordered_sensors_top))
        for sample_i in range(shap_values.shape[0]):
            current = float(start_value[sample_i])
            cumulative_map = {}
            for sensor in reversed_sensors:
                current += float(sensor_contrib[sensor][sample_i])
                cumulative_map[sensor] = current
            for display_rank, sensor in enumerate(ordered_sensors_top, start=1):
                decision_rows.append({'class_name': class_name, 'sensor': sensor, 'decision_rank': int(display_rank), 'sample_uid': sample_uid[sample_i], 'subject_id': subject_id[sample_i], 'true_label': true_label[sample_i], 'pred_label': pred_label[sample_i], 'start_value': float(start_value[sample_i]), 'model_output': float(total_output[sample_i]), 'sensor_contribution': float(sensor_contrib[sensor][sample_i]), 'cumulative_value': float(cumulative_map[sensor])})
    return pd.DataFrame(decision_rows)

def plot_single_class_decision(class_name: str, decision_df: pd.DataFrame, out_png: str | Path, out_pdf: str | Path, out_svg: str | Path):
    if decision_df is None or len(decision_df) == 0:
        raise ValueError(f'No SHAP decision data available for class={class_name}.')
    ordered_sensors = decision_df[['sensor', 'decision_rank']].drop_duplicates().sort_values('decision_rank')['sensor'].astype(str).tolist()
    y_positions = np.arange(len(ordered_sensors), dtype=float)
    sensor_to_y = {sensor: y for y, sensor in enumerate(ordered_sensors)}
    finite_outputs = pd.to_numeric(decision_df['model_output'], errors='coerce').to_numpy(dtype=float)
    finite_outputs = finite_outputs[np.isfinite(finite_outputs)]
    if len(finite_outputs) > 0:
        out_lo, out_hi = np.nanpercentile(finite_outputs, [5, 95])
        if not np.isfinite(out_lo) or not np.isfinite(out_hi) or out_hi <= out_lo:
            out_lo, out_hi = (float(np.nanmin(finite_outputs)), float(np.nanmax(finite_outputs)))
    else:
        out_lo, out_hi = (-1.0, 1.0)
    denom = out_hi - out_lo if out_hi > out_lo else 1.0
    all_x = []
    for sample_uid, sub in decision_df.groupby('sample_uid', sort=False):
        sub = sub.sort_values('decision_rank')
        all_x.extend(pd.to_numeric(sub['cumulative_value'], errors='coerce').tolist())
        all_x.append(float(pd.to_numeric(sub['start_value'], errors='coerce').iloc[0]))
    all_x = np.asarray(all_x, dtype=float)
    all_x = all_x[np.isfinite(all_x)]
    if len(all_x) > 0:
        x_min = float(np.nanpercentile(all_x, 1))
        x_max = float(np.nanpercentile(all_x, 99))
        pad = 0.08 * max(1e-06, x_max - x_min)
        x_min -= pad
        x_max += pad
    else:
        x_min, x_max = (-1.0, 1.0)
    fig_height = max(5.4, 0.42 * len(ordered_sensors) + 1.0)
    fig, ax = plt.subplots(figsize=(5.4, fig_height))
    cmap = plt.cm.cool
    for y in y_positions:
        ax.axhline(y=y, color='#9e9e9e', linestyle=(0, (3, 4)), linewidth=1.0, zorder=0)
    for sample_uid, sub in decision_df.groupby('sample_uid', sort=False):
        sub = sub.sort_values('decision_rank')
        model_output = float(pd.to_numeric(sub['model_output'], errors='coerce').iloc[0])
        start_value = float(pd.to_numeric(sub['start_value'], errors='coerce').iloc[0])
        color_val = float(np.clip((model_output - out_lo) / denom, 0.0, 1.0))
        x = [start_value] + pd.to_numeric(sub['cumulative_value'], errors='coerce').to_numpy(dtype=float).tolist()
        y = [len(ordered_sensors) - 0.05] + [sensor_to_y[s] for s in sub['sensor'].astype(str).tolist()]
        ax.plot(x, y, color=cmap(color_val), alpha=0.16, linewidth=1.0, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_sensors, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Model output', fontsize=12)
    ax.set_title(class_name_to_plot_title(class_name), fontsize=14)
    ax.tick_params(axis='x', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.05)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Model output', rotation=270, labelpad=14, fontsize=11)
    fig.tight_layout(pad=0.7)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close(fig)

def _stratified_sample_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if n <= 0 or len(y) <= n:
        return np.arange(len(y), dtype=int)
    rng = np.random.RandomState(int(seed))
    classes = np.unique(y)
    per_class = max(1, int(n // max(1, len(classes))))
    chosen: List[int] = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        chosen.extend(idx[:min(per_class, len(idx))].tolist())
    chosen_set = set(chosen)
    remaining = np.array([i for i in range(len(y)) if i not in chosen_set], dtype=int)
    rng.shuffle(remaining)
    chosen.extend(remaining[:max(0, n - len(chosen))].tolist())
    out = np.asarray(chosen[:n], dtype=int)
    rng.shuffle(out)
    return out

def _normalize_multiclass_shap_array(shap_values: object, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    if isinstance(shap_values, list):
        arrs = [np.asarray(v, dtype=float) for v in shap_values]
        if len(arrs) != n_classes:
            raise ValueError(f'Expected {n_classes} SHAP class arrays, got {len(arrs)}')
        return np.stack(arrs, axis=1)
    arr = np.asarray(shap_values, dtype=float)
    if arr.shape == (n_samples, n_features, n_classes):
        return np.transpose(arr, (0, 2, 1))
    if arr.shape == (n_classes, n_samples, n_features):
        return np.transpose(arr, (1, 0, 2))
    if arr.shape == (n_samples, n_classes, n_features):
        return arr
    raise ValueError(f'Cannot normalize multiclass SHAP shape={arr.shape}; expected N={n_samples}, K={n_classes}, F={n_features}.')

def compute_reference_global_shap_values(clf: XGBClassifier, X_background: pd.DataFrame, X_explain: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
    n_samples, n_features = X_explain.shape
    n_classes = int(getattr(clf, 'n_classes_', 4))
    try:
        import shap
        try:
            explainer = shap.TreeExplainer(clf, data=X_background, feature_perturbation='interventional')
        except (TypeError, ValueError):
            explainer = shap.TreeExplainer(clf, data=X_background)
        raw = explainer.shap_values(X_explain)
        shap_values = _normalize_multiclass_shap_array(raw, n_samples=n_samples, n_features=n_features, n_classes=n_classes)
        ev = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
        if ev.size == 1:
            ev = np.repeat(ev, n_classes)
        if ev.size < n_classes:
            ev = np.pad(ev, (0, n_classes - ev.size), mode='edge')
        base_values = np.tile(ev[:n_classes][None, :], (n_samples, 1))
        return (shap_values.astype(np.float32), base_values.astype(np.float32), 'TreeExplainer-interventional')
    except Exception as e:
        shap_values, base_values = compute_multiclass_shap_values_for_model(clf, X_explain)
        return (shap_values, base_values, 'xgboost-pred_contribs')

def build_reference_decision_data(shap_payload: Dict[str, object], class_name: str, importance_df: pd.DataFrame, class_names: List[str], max_display: int=10, max_samples: int=30, seed: int=42, sensor_order: List[str] | None=None) -> pd.DataFrame:
    class_idx = {name: i for i, name in enumerate(class_names)}[class_name]
    if sensor_order is None:
        sensor_order = list(SENSOR_DISPLAY_ORDER)
    cls_imp = importance_df[importance_df['class_name'] == class_name].sort_values(['mean_abs_shap', 'summary_rank'], ascending=[False, True]).reset_index(drop=True)
    ranked_top = [str(s) for s in cls_imp['sensor'].astype(str).tolist() if str(s) in sensor_order]
    ranked_top = ranked_top + [str(s) for s in sensor_order if str(s) not in ranked_top]
    path_sensors = list(reversed(ranked_top))
    feature_names = [str(x) for x in shap_payload['feature_names']]
    sensor_map = get_sensor_feature_index_map(feature_names)
    sv = np.asarray(shap_payload['shap_values'], dtype=float)[:, class_idx, :]
    base = np.asarray(shap_payload['base_values'], dtype=float)[:, class_idx]
    true_label = np.asarray(shap_payload['true_label'], dtype=str)
    sample_uid = np.asarray(shap_payload['sample_uid'], dtype=str)
    subject_id = np.asarray(shap_payload['subject_id'], dtype=str)
    pred_label = np.asarray(shap_payload['pred_label'], dtype=str)
    y_for_sample = np.array([class_names.index(x) if x in class_names else -1 for x in true_label], dtype=int)
    selected = _stratified_sample_indices(y_for_sample, min(int(max_samples), len(sv)), int(seed) + class_idx * 101)
    contrib = {}
    for sensor in path_sensors:
        idxs = sensor_map.get(sensor, [])
        contrib[sensor] = np.sum(sv[:, idxs], axis=1) if len(idxs) > 0 else np.zeros(len(sv), dtype=float)
    rows = []
    for sample_i in selected:
        current = float(base[sample_i])
        exact_output = float(base[sample_i] + np.sum(sv[sample_i, :]))
        displayed_total = float(base[sample_i] + sum((float(contrib[s][sample_i]) for s in path_sensors)))
        for step, sensor in enumerate(path_sensors, start=1):
            before = current
            current += float(contrib[sensor][sample_i])
            rows.append({'class_name': class_name, 'sample_uid': str(sample_uid[sample_i]), 'subject_id': str(subject_id[sample_i]), 'true_label': str(true_label[sample_i]), 'pred_label': str(pred_label[sample_i]), 'path_step': int(step), 'sensor': sensor, 'sensor_contribution': float(contrib[sensor][sample_i]), 'cumulative_before': float(before), 'cumulative_value': float(current), 'start_value': float(base[sample_i]), 'displayed_output': displayed_total, 'exact_model_output': exact_output})
    return pd.DataFrame(rows)

def plot_reference_decision(class_name: str, decision_df: pd.DataFrame, out_png: str | Path, out_pdf: str | Path, out_svg: str | Path):
    if decision_df is None or len(decision_df) == 0:
        raise ValueError(f'No decision data for class={class_name}')
    sensors = decision_df[['sensor', 'path_step']].drop_duplicates().sort_values('path_step')['sensor'].astype(str).tolist()
    y_map = {s: i + 1 for i, s in enumerate(sensors)}
    outputs = pd.to_numeric(decision_df['exact_model_output'], errors='coerce').to_numpy(float)
    lo, hi = np.nanpercentile(outputs[np.isfinite(outputs)], [5, 95]) if np.any(np.isfinite(outputs)) else (-1.0, 1.0)
    denom = max(float(hi - lo), 1e-09)
    all_x = np.r_[pd.to_numeric(decision_df['start_value'], errors='coerce').to_numpy(float), pd.to_numeric(decision_df['cumulative_value'], errors='coerce').to_numpy(float)]
    all_x = all_x[np.isfinite(all_x)]
    x0, x1 = np.nanpercentile(all_x, [1, 99]) if len(all_x) else (-1.0, 1.0)
    pad = 0.08 * max(float(x1 - x0), 1e-06)
    fig, ax = plt.subplots(figsize=(5.6, max(5.2, 0.42 * len(sensors) + 1.4)))
    cmap = plt.cm.cool
    for y in range(1, len(sensors) + 1):
        ax.axhline(y, color='#9e9e9e', linestyle=(0, (3, 4)), linewidth=0.9, zorder=0)
    for _, sub in decision_df.groupby('sample_uid', sort=False):
        sub = sub.sort_values('path_step')
        exact = float(pd.to_numeric(sub['exact_model_output'], errors='coerce').iloc[0])
        color = cmap(float(np.clip((exact - lo) / denom, 0.0, 1.0)))
        x = [float(sub['start_value'].iloc[0])] + sub['cumulative_value'].astype(float).tolist()
        y = [0.0] + [y_map[s] for s in sub['sensor'].astype(str)]
        ax.plot(x, y, color=color, alpha=0.35, linewidth=0.9, marker='o', markersize=1.6, markeredgewidth=0.0, zorder=1)
    ax.set_yticks(range(1, len(sensors) + 1))
    ax.set_yticklabels(sensors, fontsize=11)
    ax.set_ylim(-0.25, len(sensors) + 0.45)
    ax.set_xlim(float(x0 - pad), float(x1 + pad))
    ax.set_xlabel('Model output', fontsize=12)
    ax.set_title(class_name_to_plot_title(class_name), fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9], pad=0.7)
    ax_pos = ax.get_position()
    cax = fig.add_axes([ax_pos.x0, min(0.94, ax_pos.y1 + 0.025), ax_pos.width, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=10, pad=1)
    cbar.set_label('Model output', fontsize=11, labelpad=2)
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close(fig)

def make_reference_global_shap_outputs(all_ds: pd.DataFrame, label_order: List[str], label_to_id: Dict[str, int], out_shap: Path, args: argparse.Namespace) -> None:
    X_all = make_sensor_feature_matrix(all_ds, feature_cols=None, feature_mode=str(args.shap_model_feature_mode))
    y_all = np.array([label_to_id[x] for x in all_ds['label'].astype(str)], dtype=int)
    w_all = compute_class_weights(y_all, n_class=4)
    clf = train_xgb_4class(X_all, y_all, sample_weight=w_all, seed=int(args.seed), n_estimators=int(args.shap_model_estimators), max_depth=int(args.shap_model_max_depth), min_child_weight=2.0, reg_alpha=0.0, reg_lambda=2.0, gamma=0.0, subsample=0.9, colsample_bytree=0.9, n_jobs=int(args.xgb_n_jobs))
    clf.save_model(str(out_shap / 'global_4class_model_for_shap.json'))
    bg_idx = _stratified_sample_indices(y_all, min(int(args.shap_bg_n), len(X_all)), int(args.seed))
    ex_idx = _stratified_sample_indices(y_all, min(int(args.shap_explain_n), len(X_all)), int(args.seed) + 123)
    X_bg = X_all.iloc[bg_idx].copy()
    X_explain = X_all.iloc[ex_idx].copy()
    sv, bv, method = compute_reference_global_shap_values(clf, X_bg, X_explain)
    pred = np.argmax(clf.predict_proba(X_explain), axis=1)
    meta = all_ds.iloc[ex_idx].reset_index(drop=True)
    payload = {'subject_id': meta['subject_id'].astype(str).tolist(), 'sample_uid': [f'global__{int(i):06d}' for i in ex_idx], 'true_label': meta['label'].astype(str).tolist(), 'pred_label': [label_order[int(i)] for i in pred], 'feature_names': list(X_explain.columns), 'feature_values': X_explain.to_numpy(dtype=np.float32, copy=True), 'shap_values': sv.astype(np.float32), 'base_values': bv.astype(np.float32)}
    save_multiclass_shap_payload_npz(out_npz=out_shap / 'global_4class_shap_raw.npz', shap_values=payload['shap_values'], base_values=payload['base_values'], feature_values=payload['feature_values'], feature_names=payload['feature_names'], class_names=label_order, sample_uid=payload['sample_uid'], subject_id=payload['subject_id'], true_label=payload['true_label'], pred_label=payload['pred_label'])
    importance_df, plot_df = build_sensor_level_shap_tables([payload], label_order, SENSOR_DISPLAY_ORDER)
    importance_df.to_csv(out_shap / 'sensor_level_shap_feature_importance.csv', index=False, encoding='utf-8-sig')
    plot_df.to_csv(out_shap / 'sensor_level_shap_plot_data.csv', index=False, encoding='utf-8-sig')
    build_shap_sensor_feature_mapping_table([payload]).to_csv(out_shap / 'sensor_level_feature_mapping.csv', index=False, encoding='utf-8-sig')
    stacked_df = build_sensor_stacked_importance_table(importance_df, label_order)
    stacked_df.to_csv(out_shap / 'sensor_level_stacked_importance_data.csv', index=False, encoding='utf-8-sig')
    plot_sensor_stacked_importance(stacked_df, label_order, out_shap / 'sensor_level_stacked_importance.png', out_shap / 'sensor_level_stacked_importance.pdf', out_shap / 'sensor_level_stacked_importance.svg')
    pd.DataFrame([{'shap_method': method, 'n_background': int(len(bg_idx)), 'n_explained': int(len(ex_idx)), 'n_features': int(X_all.shape[1]), 'feature_mode': str(args.shap_model_feature_mode), 'model_estimators': int(args.shap_model_estimators), 'model_max_depth': int(args.shap_model_max_depth), 'prediction_clip_z': float(args.clip_z), 'shap_sic_clip_z': float(args.shap_sic_clip_z), 'classification_metrics_affected_by_shap_dataset': False}]).to_csv(out_shap / 'global_shap_run_info.csv', index=False, encoding='utf-8-sig')

def classification_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--excels', nargs='+', default=None)
    ap.add_argument('--excel_dir', type=str, default=None, help='folder containing excel files')
    ap.add_argument('--excel_pattern', type=str, default='*.xlsx', help='excel filename pattern')
    ap.add_argument('--out', default='outputs_fig4e_loso_only')
    ap.add_argument('--sweat_delay', type=int, default=300, help='seconds, shift(-delay) for sweat cols')
    ap.add_argument('--sweat_cols', type=str, default='LA,PH,AA,UA,GLU')
    ap.add_argument('--n_segments', type=int, default=3)
    ap.add_argument('--sample_level', choices=['window', 'substage'], default='window', help='window is recommended for Fig. 4e-like subject-wise macro F1; substage reproduces the older whole-stage-row mode')
    ap.add_argument('--window_sec', type=int, default=60, help='fixed evaluation window length in seconds')
    ap.add_argument('--stride_sec', type=int, default=30, help='stride between evaluation windows in seconds')
    ap.add_argument('--min_window_coverage', type=float, default=0.8, help='minimum fraction of expected 1Hz samples in a window')
    ap.add_argument('--min_label_fraction', type=float, default=0.8, help='minimum majority-label fraction; transition windows below this are skipped')
    ap.add_argument('--transition_margin_sec', type=int, default=0, help='skip windows whose center is within this many seconds of a true stage transition; useful for steady-state Fig. 4e-style evaluation')
    ap.add_argument('--baseline_sec', type=int, default=180)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_model_seeds', type=int, default=1, help='train several XGBoost models with different random seeds and average probabilities; 3-7 is usually more stable')
    ap.add_argument('--seed_step', type=int, default=17, help='seed increment used when --n_model_seeds > 1')
    ap.add_argument('--n_estimators', type=int, default=250, help='XGBoost trees. Increase for final run if needed.')
    ap.add_argument('--xgb_n_jobs', type=int, default=1, help='XGBoost worker count; 1 avoids severe oversubscription on many-core systems')
    ap.add_argument('--reference_y', type=float, default=90.0, help='red dashed reference line in percent')
    ap.add_argument('--no_auto_align_sheet_dates', action='store_true', help='disable automatic correction of whole-day timestamp offsets')
    ap.add_argument('--date_align_min_overlap_ratio', type=float, default=0.8, help='minimum shifted recording overlap for automatic date correction')
    ap.add_argument('--date_align_max_clock_diff_sec', type=float, default=3600.0, help='maximum start-time difference after whole-day correction')
    ap.add_argument('--print_time_ranges', action='store_true', help='print timestamp ranges for all sheets before alignment')
    ap.add_argument('--scale_floor_fraction', type=float, default=0.1, help='minimum normalization scale as a fraction of whole-session robust scale')
    ap.add_argument('--clip_z', type=float, default=20.0, help='clip normalized sensor values to +/- this value; <=0 disables clipping')
    ap.add_argument('--shap_sic_clip_z', type=float, default=30.0, help='SIC clip used only when rebuilding the global SHAP feature dataset; does not affect LOSO predictions or boxplot/confusion-matrix metrics')
    ap.add_argument('--trajectory_mode', choices=['none', 'causal', 'offline'], default='none', help='between-window/substage trend features; default none avoids whole-session/order information')
    ap.add_argument('--trajectory_sensors', type=str, default='GRS,Temp,HR,SR,SIC', help='sensors used for cross-substage trajectory and quality features')
    ap.add_argument('--feature_mode', choices=['all', 'compact'], default='compact', help='compact is recommended for the small sample size')
    ap.add_argument('--prediction_profile', choices=['paper_faithful', 'stable', 'high_accuracy', 'protocol_calibrated'], default='protocol_calibrated', help='paper_faithful: independent-window argmax only; stable: add light smoothing and isolated-flip correction; high_accuracy: stable plus the flexible protocol decoder; protocol_calibrated: high_accuracy plus a training-only phase-duration prior and 120-second windows unless --window_sec is explicitly supplied')
    ap.add_argument('--use_protocol_duration_prior', action='store_true', help='fuse a soft stage-position prior estimated only from LOSO training subjects')
    ap.add_argument('--disable_protocol_duration_prior', action='store_true', help='disable the duration prior even when prediction_profile=protocol_calibrated')
    ap.add_argument('--protocol_prior_strength', type=float, default=2.0, help='log-probability fusion strength for the training-only protocol prior')
    ap.add_argument('--protocol_prior_width', type=float, default=0.01, help='soft transition width as a fraction of the held-out sequence')
    ap.add_argument('--protocol_prior_floor', type=float, default=0.0001, help='minimum class probability in the protocol-position prior')
    ap.add_argument('--protocol_prior_max_mad', type=float, default=0.05, help='transition-fraction MAD that reduces the prior confidence to zero')
    ap.add_argument('--max_depth', type=int, default=2)
    ap.add_argument('--min_child_weight', type=float, default=3.0)
    ap.add_argument('--reg_alpha', type=float, default=0.3)
    ap.add_argument('--reg_lambda', type=float, default=5.0)
    ap.add_argument('--gamma', type=float, default=0.05)
    ap.add_argument('--subsample', type=float, default=0.85)
    ap.add_argument('--colsample_bytree', type=float, default=0.7)
    ap.add_argument('--disable_reference_hierarchical_fusion', action='store_true', help='disable Rest-vs-Active + Active-3-class auxiliary probability fusion')
    ap.add_argument('--enable_reference_hierarchical_fusion', action='store_true', help='enable the additional hierarchical probability model. It is off by default in v12 because temporal decoding can otherwise turn small probability changes into different hard labels')
    ap.add_argument('--reference_hierarchical_alpha', type=float, default=0.35, help='maximum blend weight for reference hierarchical probabilities')
    ap.add_argument('--reference_hierarchical_margin', type=float, default=0.25, help='only uncertain direct predictions receive substantial hierarchical weight')
    ap.add_argument('--allow_reference_hierarchical_label_changes', action='store_true', help='allow hierarchical fusion to change the direct model top class; off by default for metric stability')
    ap.add_argument('--reference_binary_estimators', type=int, default=250)
    ap.add_argument('--reference_active_estimators', type=int, default=350)
    ap.add_argument('--use_order_decode', action='store_true', help='legacy alias; flexible sequence decoding is enabled by default in v4')
    ap.add_argument('--disable_sequence_decode', action='store_true', help='legacy compatibility flag; sequence decoding is disabled by default in the paper-faithful mode')
    ap.add_argument('--enable_sequence_decode', action='store_true', help='enable protocol-order sequence decoding; disabled by default to avoid inflating Fig. 4e-like scores')
    ap.add_argument('--force_end_recovery', action='store_true', help='offline protocol prior: force the final substage/window to Recovery')
    ap.add_argument('--use_sr_rules', action='store_true', help='apply SR presence/trend post-processing rules')
    ap.add_argument('--use_early_temp_rescue', action='store_true', help='use a LOSO-trained Temp-only Rest/Warmup auxiliary model')
    ap.add_argument('--early_temp_threshold', type=float, default=0.5, help='Temp auxiliary Warmup probability threshold')
    ap.add_argument('--min_baseline_segments', type=int, default=2, help='minimum initial baseline segments before Temp can trigger Warmup')
    ap.add_argument('--temp_aux_C', type=float, default=0.1, help='regularization strength C for the Temp auxiliary logistic model')
    ap.add_argument('--enable_temp_hr_trend_fusion', action='store_true', help='enable an additional Temp+HR Rest/Warmup auxiliary model and fuse its probability')
    ap.add_argument('--temp_hr_aux_C', type=float, default=0.15, help='regularization strength C for the Temp+HR auxiliary logistic model')
    ap.add_argument('--temp_hr_fusion_alpha', type=float, default=0.25, help='fusion weight for the Temp+HR Rest/Warmup auxiliary model')
    ap.add_argument('--temp_hr_min_conf', type=float, default=0.2, help='minimum confidence for Temp+HR auxiliary fusion')
    ap.add_argument('--enable_recovery_trend_boost', action='store_true', help='boost late Recovery probability when HR/Temp/SR show falling trends')
    ap.add_argument('--recovery_trend_start_frac', type=float, default=0.68, help='start applying recovery trend boost after this fraction of the held-out sequence')
    ap.add_argument('--recovery_trend_strength', type=float, default=0.28, help='strength of late Recovery trend boosting')
    ap.add_argument('--recovery_trend_min_fall', type=float, default=0.05, help='minimum falling-trend score required for Recovery trend boost')
    ap.add_argument('--enable_delayed_hr_temp_rescue', action='store_true', help='use Temp-only Warmup rescue only when HR onset is clearly delayed relative to Temp')
    ap.add_argument('--delayed_hr_trigger', type=int, default=12, help='minimum detected HR-vs-Temp onset delay, in windows, required for delayed-HR Temp rescue')
    ap.add_argument('--delayed_hr_temp_threshold', type=float, default=0.55, help='Temp auxiliary Warmup probability threshold for delayed-HR rescue')
    ap.add_argument('--delayed_hr_min_temp_delta', type=float, default=0.05, help='minimum Temp window delta for delayed-HR rescue')
    ap.add_argument('--delayed_hr_min_temp_slope', type=float, default=0.0005, help='minimum Temp window slope for delayed-HR rescue')
    ap.add_argument('--force_tail_recovery_windows', type=int, default=0, help='offline protocol prior: force the final N windows to Recovery; use 16 for window_sec=120,stride_sec=30 in this protocol')
    ap.add_argument('--disable_temp_fusion', action='store_true', help='legacy compatibility flag; Temp fusion is disabled by default unless --enable_temp_fusion is set')
    ap.add_argument('--enable_temp_fusion', action='store_true', help='enable Temp Rest/Warmup probability fusion; disabled by default for paper-faithful evaluation')
    ap.add_argument('--add_subject_quality_features', action='store_true', help='add subject-level dynamic-range features; disabled by default because they use the whole held-out session')
    ap.add_argument('--legacy_v4_defaults', action='store_true', help='reproduce the older v4 behaviour: substage rows, offline trajectory, sequence decode, Temp fusion, and quality features')
    ap.add_argument('--adaptive_temp_fusion', action='store_true', help='use confidence-gated Temp fusion instead of fixed-alpha fusion; usually safer for Rest/Warmup boundary cases')
    ap.add_argument('--adaptive_temp_min_conf', type=float, default=0.3, help='minimum Temp auxiliary confidence for adaptive fusion')
    ap.add_argument('--temp_fusion_alpha', type=float, default=0.35, help='normal Temp fusion weight applied to every LOSO subject')
    ap.add_argument('--temp_fusion_alpha_delayed_hr', type=float, default=0.7, help='Temp fusion weight when HR onset is delayed relative to Temp')
    ap.add_argument('--hr_delay_segments', type=int, default=2, help='minimum HR-vs-Temp onset delay that activates the stronger weight')
    ap.add_argument('--sensor_onset_fraction', type=float, default=0.15, help='fraction of session dynamic range used to estimate sensor onset')
    ap.add_argument('--sr_presence_thr', type=float, default=0.05, help='SR present fraction threshold for not-Rest rule')
    ap.add_argument('--sr_falling_slope_thr', type=float, default=-0.02, help='SR slope threshold for Recovery rule')
    ap.add_argument('--sr_rising_slope_thr', type=float, default=0.02, help='SR slope threshold for Warmup/Exercise rule')
    ap.add_argument('--smooth_proba', action='store_true', help='apply a light temporal probability smoothing before argmax; this uses neighbouring windows and is an offline evaluation option')
    ap.add_argument('--smooth_kernel', type=str, default='0.20,0.60,0.20', help='comma-separated smoothing kernel, e.g. 0.2,0.6,0.2')
    ap.add_argument('--isolated_fix', action='store_true', help='replace low-confidence isolated one-window label flips between identical neighbouring predictions')
    ap.add_argument('--isolated_fix_conf', type=float, default=0.65, help='only fix isolated flips when the original top probability is below this threshold')
    ap.add_argument('--disable_confusion_matrix_plot', action='store_true', help='skip the overall 4-class confusion-matrix figure and CSV exports')
    ap.add_argument('--disable_shap_plots', action='store_true', help='skip the multiclass SHAP figure and raw SHAP exports')
    ap.add_argument('--shap_max_display', type=int, default=10, help='top sensor count shown in each class-wise SHAP/decision panel')
    ap.add_argument('--shap_bg_n', type=int, default=500, help='stratified background sample count for TreeExplainer')
    ap.add_argument('--shap_explain_n', type=int, default=2000, help='stratified sample count explained by the final global SHAP model')
    ap.add_argument('--shap_decision_n', type=int, default=1600, help='number of deterministic stratified lines in each decision plot')
    ap.add_argument('--shap_model_feature_mode', choices=['all', 'compact'], default='all', help='program-2-style global interpretation model uses all sensor features by default')
    ap.add_argument('--shap_model_estimators', type=int, default=400)
    ap.add_argument('--shap_model_max_depth', type=int, default=4)
    ap.add_argument('--save_loso_shap_raw', action='store_true', help='also export old per-fold SHAP tensors; not needed for the new global figures')
    args = ap.parse_args()
    if args.legacy_v4_defaults:
        args.sample_level = 'substage'
        args.trajectory_mode = 'offline'
        args.enable_sequence_decode = True
        args.enable_temp_fusion = True
        args.add_subject_quality_features = True
    if args.prediction_profile in {'stable', 'high_accuracy', 'protocol_calibrated'}:
        args.smooth_proba = True
        args.isolated_fix = True
    if args.prediction_profile in {'high_accuracy', 'protocol_calibrated'}:
        args.enable_sequence_decode = True
    if args.prediction_profile == 'protocol_calibrated':
        args.use_protocol_duration_prior = True
        if '--window_sec' not in sys.argv:
            args.window_sec = 120
    if args.disable_protocol_duration_prior:
        args.use_protocol_duration_prior = False
    if args.enable_reference_hierarchical_fusion:
        args.disable_reference_hierarchical_fusion = False
    else:
        args.disable_reference_hierarchical_fusion = True
    if not args.enable_temp_fusion:
        args.disable_temp_fusion = True
    if args.use_early_temp_rescue:
        pass
    out_root = ensure_dir(args.out)
    out_intermediate = ensure_dir(out_root / 'intermediate')
    out_confusion = ensure_dir(out_root / 'confusion_matrix_outputs')
    out_shap = ensure_dir(out_root / 'shap_outputs')
    remove_legacy_class_shap_dirs(out_shap, ['Rest', 'Warmup', 'Exercise', 'Recovery'])
    out_shap_raw = ensure_dir(out_shap / 'raw_per_subject')
    sweat_cols = [x.strip() for x in args.sweat_cols.split(',') if x.strip()]
    excel_paths: List[Path] = []
    if args.excel_dir is not None:
        excel_dir = Path(args.excel_dir)
        discovered = sorted(excel_dir.glob(args.excel_pattern))
        for p in discovered:
            if p.name.startswith('~$'):
                continue
            excel_paths.append(p)
    if args.excels is not None:
        for raw_path in args.excels:
            p = Path(raw_path)
            if p.name.startswith('~$'):
                continue
            excel_paths.append(p)
    seen = set()
    excel_paths_unique = []
    for p in excel_paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            excel_paths_unique.append(p)
    excel_paths = excel_paths_unique
    if len(excel_paths) == 0:
        raise ValueError('No Excel files found. Use --excel_dir or --excels.')
    all_subject_rows = []
    all_subject_rows_shap = []
    need_separate_shap_features = not args.disable_shap_plots and float(args.shap_sic_clip_z) > max(0.0, float(args.clip_z))
    for i, excel_path in enumerate(excel_paths, start=1):
        sid = subject_name_from_path(excel_path)
        merged = build_merged_1hz(excel_path=excel_path, sweat_cols=sweat_cols, sweat_delay_sec=args.sweat_delay, auto_align_sheet_dates=not args.no_auto_align_sheet_dates, date_align_min_overlap_ratio=args.date_align_min_overlap_ratio, date_align_max_clock_diff_sec=args.date_align_max_clock_diff_sec, print_time_ranges=args.print_time_ranges)
        merged.to_csv(out_intermediate / f'{sid}__merged_1hz.csv', index=False, encoding='utf-8-sig')
        if args.sample_level == 'window':
            sub_ds = make_window_feature_dataset(df_1hz=merged, subject_id=sid, window_sec=args.window_sec, stride_sec=args.stride_sec, n_segments=args.n_segments, baseline_sec=args.baseline_sec, scale_floor_fraction=args.scale_floor_fraction, clip_z=args.clip_z, sic_clip_floor=0.0, min_window_coverage=args.min_window_coverage, min_label_fraction=args.min_label_fraction, transition_margin_sec=args.transition_margin_sec)
        else:
            sub_ds = make_substage_feature_dataset(df_1hz=merged, subject_id=sid, n_segments=args.n_segments, baseline_sec=args.baseline_sec, scale_floor_fraction=args.scale_floor_fraction, clip_z=args.clip_z, sic_clip_floor=0.0)
        if len(sub_ds) == 0:
            continue
        all_subject_rows.append(sub_ds)
        if need_separate_shap_features:
            if args.sample_level == 'window':
                shap_sub_ds = make_window_feature_dataset(df_1hz=merged, subject_id=sid, window_sec=args.window_sec, stride_sec=args.stride_sec, n_segments=args.n_segments, baseline_sec=args.baseline_sec, scale_floor_fraction=args.scale_floor_fraction, clip_z=args.clip_z, sic_clip_floor=args.shap_sic_clip_z, min_window_coverage=args.min_window_coverage, min_label_fraction=args.min_label_fraction, transition_margin_sec=args.transition_margin_sec)
            else:
                shap_sub_ds = make_substage_feature_dataset(df_1hz=merged, subject_id=sid, n_segments=args.n_segments, baseline_sec=args.baseline_sec, scale_floor_fraction=args.scale_floor_fraction, clip_z=args.clip_z, sic_clip_floor=args.shap_sic_clip_z)
            if len(shap_sub_ds) > 0:
                all_subject_rows_shap.append(shap_sub_ds)
    if len(all_subject_rows) == 0:
        raise ValueError('No valid substage feature rows were built.')
    all_ds = pd.concat(all_subject_rows, ignore_index=True)
    trajectory_sensors = [x.strip() for x in args.trajectory_sensors.split(',') if x.strip()]
    all_ds = add_longitudinal_features(all_ds, sensors=trajectory_sensors, mode=args.trajectory_mode)
    if args.add_subject_quality_features:
        all_ds = add_subject_signal_quality_features(all_ds, sensors=trajectory_sensors)
    if need_separate_shap_features and len(all_subject_rows_shap) > 0:
        all_ds_shap = pd.concat(all_subject_rows_shap, ignore_index=True)
        all_ds_shap = add_longitudinal_features(all_ds_shap, sensors=trajectory_sensors, mode=args.trajectory_mode)
        if args.add_subject_quality_features:
            all_ds_shap = add_subject_signal_quality_features(all_ds_shap, sensors=trajectory_sensors)
        all_ds_shap.to_csv(out_root / 'shap_only_feature_dataset.csv', index=False, encoding='utf-8-sig')
    else:
        all_ds_shap = all_ds.copy()
    feature_dataset_name = 'window_feature_dataset.csv' if args.sample_level == 'window' else 'substage_feature_dataset.csv'
    all_ds.to_csv(out_root / feature_dataset_name, index=False, encoding='utf-8-sig')
    label_order = ['Rest', 'Warmup', 'Exercise', 'Recovery']
    label_to_id = {lab: i for i, lab in enumerate(label_order)}
    all_ds = all_ds[all_ds['label'].isin(label_order)].reset_index(drop=True)
    all_ds_shap = all_ds_shap[all_ds_shap['label'].isin(label_order)].reset_index(drop=True)
    subjects = sorted(all_ds['subject_id'].unique().tolist())
    for i, sid in enumerate(subjects, start=1):
        pass
    summary_rows = []
    pred_rows = []
    shap_payloads = []
    shap_raw_manifest_rows = []
    for subject_index, val_sid in enumerate(subjects, start=1):
        train_ds = all_ds[all_ds['subject_id'] != val_sid].reset_index(drop=True)
        val_ds = all_ds[all_ds['subject_id'] == val_sid].reset_index(drop=True)
        print_signal_quality_warnings(train_ds, val_ds, trajectory_sensors)
        X_train = make_sensor_feature_matrix(train_ds, feature_cols=None, feature_mode=args.feature_mode)
        feature_cols = list(X_train.columns)
        X_val = make_sensor_feature_matrix(val_ds, feature_cols=feature_cols, feature_mode=args.feature_mode)
        y_train = np.array([label_to_id[x] for x in train_ds['label'].astype(str)], dtype=int)
        y_val = np.array([label_to_id[x] for x in val_ds['label'].astype(str)], dtype=int)
        w_train = compute_class_weights(y_train, n_class=4)
        n_model_seeds = max(1, int(args.n_model_seeds))
        proba_list = []
        clfs = []
        for model_i in range(n_model_seeds):
            model_seed = int(args.seed + model_i * args.seed_step)
            clf = train_xgb_4class(X_train, y_train, sample_weight=w_train, seed=model_seed, n_estimators=args.n_estimators, max_depth=args.max_depth, min_child_weight=args.min_child_weight, reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda, gamma=args.gamma, subsample=args.subsample, colsample_bytree=args.colsample_bytree, n_jobs=args.xgb_n_jobs)
            clfs.append(clf)
            proba_list.append(clf.predict_proba(X_val))
        proba4_model = np.mean(proba_list, axis=0)
        row_sum_model = proba4_model.sum(axis=1, keepdims=True)
        row_sum_model[row_sum_model <= 0.0] = 1.0
        proba4_model = proba4_model / row_sum_model
        y_pred_model_raw = np.argmax(proba4_model, axis=1)
        reference_hierarchical_proba = proba4_model.copy()
        reference_alpha_row = np.zeros(len(X_val), dtype=float)
        if not args.disable_reference_hierarchical_fusion:
            ref_binary = train_reference_binary_active_model(X_train, y_train, seed=args.seed, n_estimators=args.reference_binary_estimators, n_jobs=args.xgb_n_jobs)
            ref_active = train_reference_active_3class_model(X_train, y_train, seed=args.seed, n_estimators=args.reference_active_estimators, n_jobs=args.xgb_n_jobs)
            reference_hierarchical_proba = predict_reference_hierarchical_probability(ref_binary, ref_active, X_val)
            proba4_prediction_base, reference_alpha_row = fuse_reference_hierarchical_probability(direct_proba4=proba4_model, hierarchical_proba4=reference_hierarchical_proba, alpha=args.reference_hierarchical_alpha, uncertainty_margin=args.reference_hierarchical_margin, allow_label_changes=bool(args.allow_reference_hierarchical_label_changes))
        else:
            proba4_prediction_base = proba4_model.copy()
        temp_aux_model, temp_aux_cols = train_temp_warmup_auxiliary(train_ds=train_ds, seed=args.seed, C=args.temp_aux_C)
        temp_warmup_proba = predict_temp_warmup_probability(model=temp_aux_model, feature_cols=temp_aux_cols, ds=val_ds)
        temp_hr_aux_model, temp_hr_aux_cols = train_temp_hr_warmup_auxiliary(train_ds=train_ds, seed=args.seed, C=args.temp_hr_aux_C)
        temp_hr_warmup_proba = predict_temp_hr_warmup_probability(model=temp_hr_aux_model, feature_cols=temp_hr_aux_cols, ds=val_ds)
        temp_fusion_alpha, temp_onset_idx, hr_onset_idx, detected_hr_delay = choose_temp_fusion_alpha(val_ds=val_ds, base_alpha=args.temp_fusion_alpha, delayed_hr_alpha=args.temp_fusion_alpha_delayed_hr, hr_delay_segments=args.hr_delay_segments, onset_fraction=args.sensor_onset_fraction, baseline_segments=2)
        if args.disable_temp_fusion:
            temp_fusion_alpha = 0.0
            proba4 = proba4_prediction_base.copy()
        elif args.adaptive_temp_fusion:
            proba4 = fuse_rest_warmup_probabilities_adaptive(proba4=proba4_prediction_base, temp_warmup_proba=temp_warmup_proba, label_to_id=label_to_id, alpha=temp_fusion_alpha, min_temp_conf=args.adaptive_temp_min_conf)
        else:
            proba4 = fuse_rest_warmup_probabilities(proba4=proba4_prediction_base, temp_warmup_proba=temp_warmup_proba, label_to_id=label_to_id, alpha=temp_fusion_alpha)
        if args.enable_temp_hr_trend_fusion:
            proba4 = fuse_rest_warmup_probabilities_with_expert(proba4=proba4, expert_warmup_proba=temp_hr_warmup_proba, label_to_id=label_to_id, alpha=args.temp_hr_fusion_alpha, min_expert_conf=args.temp_hr_min_conf)
        if args.enable_recovery_trend_boost:
            proba4 = apply_recovery_trend_boost(val_ds=val_ds, proba4=proba4, label_to_id=label_to_id, start_frac=args.recovery_trend_start_frac, strength=args.recovery_trend_strength, min_fall_score=args.recovery_trend_min_fall)
        val_order_for_smooth = np.argsort(pd.to_datetime(val_ds['t_start']).to_numpy())
        inv_order_for_smooth = np.empty_like(val_order_for_smooth)
        inv_order_for_smooth[val_order_for_smooth] = np.arange(len(val_order_for_smooth))
        if args.smooth_proba:
            try:
                smooth_kernel = [float(x.strip()) for x in args.smooth_kernel.split(',') if x.strip()]
            except Exception:
                smooth_kernel = [0.2, 0.6, 0.2]
            proba_sorted = smooth_probabilities_in_time(proba4[val_order_for_smooth, :], smooth_kernel)
            proba4 = proba_sorted[inv_order_for_smooth, :]
        protocol_prior = np.full_like(proba4, 0.25, dtype=float)
        protocol_transition_fractions = np.full(3, np.nan, dtype=float)
        protocol_transition_mad = np.full(3, np.nan, dtype=float)
        protocol_prior_strength_used = 0.0
        protocol_prior_training_subjects = 0
        if args.use_protocol_duration_prior:
            proba_sorted_for_prior = proba4[val_order_for_smooth, :]
            proba_sorted_for_prior, protocol_prior_sorted, protocol_transition_fractions, protocol_transition_mad, protocol_prior_strength_used, protocol_prior_training_subjects = fuse_training_protocol_prior(proba4_sorted=proba_sorted_for_prior, train_ds=train_ds, label_to_id=label_to_id, strength=args.protocol_prior_strength, width=args.protocol_prior_width, probability_floor=args.protocol_prior_floor, max_transition_mad=args.protocol_prior_max_mad)
            proba4 = proba_sorted_for_prior[inv_order_for_smooth, :]
            protocol_prior = protocol_prior_sorted[inv_order_for_smooth, :]
        y_pred_raw = np.argmax(proba4, axis=1)
        if args.isolated_fix:
            y_sorted_fix = fix_isolated_label_flips(y_pred_raw[val_order_for_smooth], proba4[val_order_for_smooth, :], confidence_threshold=args.isolated_fix_conf)
            y_pred_raw = y_sorted_fix[inv_order_for_smooth]
        val_order = np.argsort(pd.to_datetime(val_ds['t_start']).to_numpy())
        inv_order = np.empty_like(val_order)
        inv_order[val_order] = np.arange(len(val_order))
        use_sequence_decode = bool(args.enable_sequence_decode or args.use_order_decode) and (not args.disable_sequence_decode)
        if use_sequence_decode:
            y_sorted = ordered_decode_stage_level(proba4[val_order, :], force_end_recovery=bool(args.force_end_recovery))
            y_pred = y_sorted[inv_order]
        else:
            y_pred = y_pred_raw.copy()
        if args.use_sr_rules:
            y_pred = apply_sr_rules(val_ds=val_ds, proba4=proba4, y_pred=y_pred, label_to_id=label_to_id, sr_presence_thr=args.sr_presence_thr, sr_falling_slope_thr=args.sr_falling_slope_thr, sr_rising_slope_thr=args.sr_rising_slope_thr)
        if args.enable_delayed_hr_temp_rescue:
            y_pred = apply_delayed_hr_temp_rescue_v5(val_ds=val_ds, y_pred=y_pred, temp_warmup_proba=temp_warmup_proba, label_to_id=label_to_id, detected_hr_delay=detected_hr_delay, hr_delay_trigger=args.delayed_hr_trigger, temp_threshold=args.delayed_hr_temp_threshold, min_temp_delta=args.delayed_hr_min_temp_delta, min_temp_slope=args.delayed_hr_min_temp_slope)
        if args.force_tail_recovery_windows > 0:
            y_pred = force_tail_recovery_windows_v5(val_ds=val_ds, y_pred=y_pred, label_to_id=label_to_id, n_tail_windows=args.force_tail_recovery_windows)
        accuracy = float(np.mean(y_val == y_pred))
        macro_f1 = f1_score(y_val, y_pred, labels=list(range(4)), average='macro', zero_division=0)
        micro_f1 = f1_score(y_val, y_pred, labels=list(range(4)), average='micro', zero_division=0)
        macro_precision, macro_recall, _, _ = precision_recall_fscore_support(y_val, y_pred, labels=list(range(4)), average='macro', zero_division=0)
        per_p, per_r, per_f1, per_support = precision_recall_fscore_support(y_val, y_pred, labels=list(range(4)), average=None, zero_division=0)
        summary_rows.append({'subject_index': subject_index, 'val_subject': val_sid, 'accuracy': float(accuracy), 'accuracy_percent': float(accuracy * 100.0), 'macro_f1': float(macro_f1), 'macro_f1_percent': float(macro_f1 * 100.0), 'micro_f1': float(micro_f1), 'micro_f1_percent': float(micro_f1 * 100.0), 'macro_precision': float(macro_precision), 'macro_recall': float(macro_recall), 'macro_precision_percent': float(macro_precision * 100.0), 'macro_recall_percent': float(macro_recall * 100.0), 'f1_Rest': float(per_f1[0]), 'f1_Warmup': float(per_f1[1]), 'f1_Exercise': float(per_f1[2]), 'f1_Recovery': float(per_f1[3]), 'support_Rest': int(per_support[0]), 'support_Warmup': int(per_support[1]), 'support_Exercise': int(per_support[2]), 'support_Recovery': int(per_support[3]), 'n_val_rows': int(len(val_ds)), 'use_order_decode': bool(use_sequence_decode), 'force_end_recovery': bool(args.force_end_recovery), 'use_sr_rules': bool(args.use_sr_rules), 'use_early_temp_rescue': bool(args.use_early_temp_rescue), 'early_temp_threshold': float(args.early_temp_threshold), 'min_baseline_segments': int(args.min_baseline_segments), 'temp_fusion_enabled': bool(not args.disable_temp_fusion), 'adaptive_temp_fusion': bool(args.adaptive_temp_fusion), 'adaptive_temp_min_conf': float(args.adaptive_temp_min_conf), 'temp_fusion_alpha_used': float(temp_fusion_alpha), 'temp_hr_trend_fusion_enabled': bool(args.enable_temp_hr_trend_fusion), 'temp_hr_fusion_alpha': float(args.temp_hr_fusion_alpha), 'temp_hr_min_conf': float(args.temp_hr_min_conf), 'n_temp_hr_features': int(len(temp_hr_aux_cols)), 'recovery_trend_boost_enabled': bool(args.enable_recovery_trend_boost), 'recovery_trend_start_frac': float(args.recovery_trend_start_frac), 'recovery_trend_strength': float(args.recovery_trend_strength), 'recovery_trend_min_fall': float(args.recovery_trend_min_fall), 'delayed_hr_temp_rescue_enabled': bool(args.enable_delayed_hr_temp_rescue), 'delayed_hr_trigger': int(args.delayed_hr_trigger), 'delayed_hr_temp_threshold': float(args.delayed_hr_temp_threshold), 'force_tail_recovery_windows': int(args.force_tail_recovery_windows), 'temp_onset_index': int(temp_onset_idx), 'hr_onset_index': int(hr_onset_idx), 'detected_hr_delay': int(detected_hr_delay), 'sample_level': str(args.sample_level), 'window_sec': int(args.window_sec) if args.sample_level == 'window' else 0, 'stride_sec': int(args.stride_sec) if args.sample_level == 'window' else 0, 'min_label_fraction': float(args.min_label_fraction) if args.sample_level == 'window' else 0.0, 'transition_margin_sec': int(args.transition_margin_sec) if args.sample_level == 'window' else 0, 'trajectory_mode': str(args.trajectory_mode), 'feature_mode': str(args.feature_mode), 'prediction_profile': str(args.prediction_profile), 'protocol_duration_prior_enabled': bool(args.use_protocol_duration_prior), 'protocol_prior_strength_requested': float(args.protocol_prior_strength), 'protocol_prior_strength_used': float(protocol_prior_strength_used), 'protocol_prior_width': float(args.protocol_prior_width), 'protocol_transition_rest_to_warmup': float(protocol_transition_fractions[0]), 'protocol_transition_warmup_to_exercise': float(protocol_transition_fractions[1]), 'protocol_transition_exercise_to_recovery': float(protocol_transition_fractions[2]), 'protocol_transition_mad_max': float(np.nanmax(protocol_transition_mad)) if np.any(np.isfinite(protocol_transition_mad)) else float('nan'), 'protocol_prior_training_subjects': int(protocol_prior_training_subjects), 'prediction_clip_z': float(args.clip_z), 'shap_sic_clip_z': float(args.shap_sic_clip_z), 'add_subject_quality_features': bool(args.add_subject_quality_features), 'n_model_seeds': int(n_model_seeds), 'reference_hierarchical_enabled': bool(not args.disable_reference_hierarchical_fusion), 'reference_hierarchical_alpha': float(args.reference_hierarchical_alpha), 'reference_hierarchical_margin': float(args.reference_hierarchical_margin), 'reference_hierarchical_allow_label_changes': bool(args.allow_reference_hierarchical_label_changes), 'reference_hierarchical_mean_alpha_used': float(np.mean(reference_alpha_row)), 'smooth_proba': bool(args.smooth_proba), 'smooth_kernel': str(args.smooth_kernel), 'isolated_fix': bool(args.isolated_fix), 'isolated_fix_conf': float(args.isolated_fix_conf), 'n_features': int(len(feature_cols)), 'auto_align_sheet_dates': bool(not args.no_auto_align_sheet_dates)})
        tmp_base_cols = [c for c in ['subject_id', 'stage_key', 'label', 't_start', 't_end', 'window_index', 'window_coverage', 'label_fraction', 'n_points'] if c in val_ds.columns]
        tmp = val_ds[tmp_base_cols].copy()
        tmp['sample_uid'] = [f'{val_sid}__{i:05d}' for i in range(len(tmp))]
        tmp['subject_index'] = subject_index
        tmp['y_true'] = y_val
        tmp['y_pred_model_raw'] = y_pred_model_raw
        tmp['y_pred_raw'] = y_pred_raw
        tmp['y_pred'] = y_pred
        tmp['true_label'] = [label_order[i] for i in y_val]
        tmp['pred_label_model_raw'] = [label_order[i] for i in y_pred_model_raw]
        tmp['pred_label_raw'] = [label_order[i] for i in y_pred_raw]
        tmp['pred_label'] = [label_order[i] for i in y_pred]
        tmp['temp_aux_p_warmup'] = temp_warmup_proba
        tmp['temp_hr_aux_p_warmup'] = temp_hr_warmup_proba
        tmp['temp_aux_n_features'] = int(len(temp_aux_cols))
        tmp['temp_hr_aux_n_features'] = int(len(temp_hr_aux_cols))
        tmp['temp_fusion_alpha'] = float(temp_fusion_alpha)
        tmp['temp_onset_index'] = int(temp_onset_idx)
        tmp['hr_onset_index'] = int(hr_onset_idx)
        tmp['detected_hr_delay'] = int(detected_hr_delay)
        tmp['reference_hierarchical_alpha_used'] = reference_alpha_row
        tmp['protocol_prior_strength_used'] = float(protocol_prior_strength_used)
        tmp['protocol_transition_rest_to_warmup'] = float(protocol_transition_fractions[0])
        tmp['protocol_transition_warmup_to_exercise'] = float(protocol_transition_fractions[1])
        tmp['protocol_transition_exercise_to_recovery'] = float(protocol_transition_fractions[2])
        for class_id, class_name in enumerate(label_order):
            tmp[f'model_p_{class_name}'] = proba4_model[:, class_id]
            tmp[f'hierarchical_p_{class_name}'] = reference_hierarchical_proba[:, class_id]
            tmp[f'reference_fused_p_{class_name}'] = proba4_prediction_base[:, class_id]
            tmp[f'protocol_prior_p_{class_name}'] = protocol_prior[:, class_id]
            tmp[f'fused_p_{class_name}'] = proba4[:, class_id]
        for c in ['SR_presence_frac', 'SR_has_any', 'SR_first_present_frac', 'SR_zero_stage_mean', 'SR_zero_stage_delta', 'SR_zero_stage_slope', 'SR_late_minus_early_mean', 'SIC_presence_frac', 'SIC_has_any', 'SIC_first_present_frac', 'SIC_valid__stage__mean', 'SIC__stage__mean', 'SIC__stage__slope', 'HR__stage__mean', 'HR__stage__slope', 'Temp__stage__mean', 'Temp__stage__slope', 'HR__stage__delta', 'HR__stage__mean__traj_prev_diff', 'Temp__stage__delta', 'Temp__stage__mean__traj_prev_diff', 'Temp_HR__stage__delta_sum', 'Temp_HR__stage__slope_sum', 'Temp_HR__late_minus_early_sum', 'GRS__stage__mean', 'GRS__stage__slope', 'GRS__stage__mean__traj_prev_diff', 'GRS__stage__mean__traj_from_start', 'GRS__stage__mean__traj_rank_pct', 'Temp__stage__mean__traj_prev_diff', 'Temp__stage__mean__traj_from_start', 'Temp__stage__mean__traj_rank_pct', 'HR__stage__mean__traj_prev_diff', 'HR__stage__mean__traj_from_start', 'HR__stage__mean__traj_rank_pct', 'GRS__quality_session_range', 'Temp__quality_session_range', 'HR__quality_session_range']:
            if c in val_ds.columns:
                tmp[c] = val_ds[c].to_numpy()
        if not args.disable_shap_plots and args.save_loso_shap_raw:
            try:
                shap_values_avg, base_values_avg = compute_multiclass_shap_values_for_ensemble(clfs=clfs, X=X_val)
                shap_npz_path = out_shap_raw / f'{subject_index:02d}__{val_sid}__multiclass_shap_raw.npz'
                save_multiclass_shap_payload_npz(out_npz=shap_npz_path, shap_values=shap_values_avg, base_values=base_values_avg, feature_values=X_val.to_numpy(dtype=np.float32, copy=True), feature_names=feature_cols, class_names=label_order, sample_uid=tmp['sample_uid'].astype(str).tolist(), subject_id=[str(val_sid)] * len(tmp), true_label=tmp['true_label'].astype(str).tolist(), pred_label=tmp['pred_label'].astype(str).tolist())
                shap_payloads.append({'subject_id': [str(val_sid)] * len(tmp), 'sample_uid': tmp['sample_uid'].astype(str).tolist(), 'true_label': tmp['true_label'].astype(str).tolist(), 'pred_label': tmp['pred_label'].astype(str).tolist(), 'feature_names': list(feature_cols), 'feature_values': X_val.to_numpy(dtype=np.float32, copy=True), 'shap_values': shap_values_avg.astype(np.float32), 'base_values': base_values_avg.astype(np.float32)})
                shap_raw_manifest_rows.append({'subject_index': int(subject_index), 'val_subject': str(val_sid), 'n_samples': int(len(tmp)), 'n_features': int(len(feature_cols)), 'shap_npz_path': str(shap_npz_path)})
            except Exception as e:
                pass
        pred_rows.append(tmp)
    summary_df = pd.DataFrame(summary_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    summary_df.to_csv(out_root / 'subject_macro_f1_summary.csv', index=False, encoding='utf-8-sig')
    pred_df.to_csv(out_root / 'all_loso_predictions.csv', index=False, encoding='utf-8-sig')
    plot_fig4e_like(summary_df=summary_df, out_png=out_root / 'fig4e_like_subject_macro_f1.png', out_pdf=out_root / 'fig4e_like_subject_macro_f1.pdf', reference_y=args.reference_y, title='')
    plot_subject_metric_boxplot(summary_df=summary_df, out_png=out_root / 'fig4e_subject_independent_metrics_boxplot.png', out_pdf=out_root / 'fig4e_subject_independent_metrics_boxplot.pdf', out_wide_csv=out_root / 'subject_independent_metrics_boxplot_data.csv', out_long_csv=out_root / 'subject_independent_metrics_boxplot_long.csv', panel_label='k', annotation='Subject-independent')
    if not args.disable_confusion_matrix_plot:
        plot_confusion_matrix_4class(y_true=pred_df['y_true'].to_numpy(dtype=int), y_pred=pred_df['y_pred'].to_numpy(dtype=int), label_order=label_order, out_png=out_confusion / 'four_class_confusion_matrix.png', out_pdf=out_confusion / 'four_class_confusion_matrix.pdf', out_counts_csv=out_confusion / 'four_class_confusion_matrix_counts.csv', out_percent_csv=out_confusion / 'four_class_confusion_matrix_percent.csv', out_long_csv=out_confusion / 'four_class_confusion_matrix_long.csv')
    if not args.disable_shap_plots:
        try:
            make_reference_global_shap_outputs(all_ds=all_ds_shap, label_order=label_order, label_to_id=label_to_id, out_shap=out_shap, args=args)
        except Exception as e:
            pass
    if len(shap_raw_manifest_rows) > 0:
        pd.DataFrame(shap_raw_manifest_rows).to_csv(out_shap / 'raw_shap_manifest.csv', index=False, encoding='utf-8-sig')
    if not args.disable_shap_plots and len(shap_payloads) > 0:
        shap_importance_df, shap_plot_df = build_sensor_level_shap_tables(shap_payloads=shap_payloads, class_names=label_order, sensor_order=SENSOR_DISPLAY_ORDER)
        shap_importance_df.to_csv(out_shap / 'sensor_level_shap_feature_importance.csv', index=False, encoding='utf-8-sig')
        shap_plot_df.to_csv(out_shap / 'sensor_level_shap_plot_data.csv', index=False, encoding='utf-8-sig')
        sensor_mapping_df = build_shap_sensor_feature_mapping_table(shap_payloads)
        sensor_mapping_df.to_csv(out_shap / 'sensor_level_feature_mapping.csv', index=False, encoding='utf-8-sig')
        stacked_df = build_sensor_stacked_importance_table(sensor_importance_df=shap_importance_df, class_names=label_order)
        stacked_df.to_csv(out_shap / 'sensor_level_stacked_importance_data.csv', index=False, encoding='utf-8-sig')
        plot_sensor_stacked_importance(stacked_df=stacked_df, class_names=label_order, out_png=out_shap / 'sensor_level_stacked_importance.png', out_pdf=out_shap / 'sensor_level_stacked_importance.pdf', out_svg=out_shap / 'sensor_level_stacked_importance.svg')
    mean_accuracy = summary_df['accuracy_percent'].mean()
    std_accuracy = summary_df['accuracy_percent'].std()
    mean_macro = summary_df['macro_f1_percent'].mean()
    std_macro = summary_df['macro_f1_percent'].std()
    mean_precision = summary_df['macro_precision_percent'].mean()
    std_precision = summary_df['macro_precision_percent'].std()
    mean_recall = summary_df['macro_recall_percent'].mean()
    std_recall = summary_df['macro_recall_percent'].std()
    if not args.disable_confusion_matrix_plot:
        pass
    if not args.disable_shap_plots:
        if args.save_loso_shap_raw:
            pass
        for class_name in label_order:
            class_slug = class_name_to_slug(class_name)
            class_dir = out_shap / class_slug

def run_embedded_shap_redraw():
    import argparse
    import re
    from pathlib import Path
    from typing import Dict, List, Optional, Sequence, Tuple
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import shap
    META_COLS = {'explain_sample_id', 'source_row_index'}
    FEATURE_NAME_MAP = {'LA': 'LA', 'SR': 'SR', 'SIC': 'SIC', 'GRS': 'GSR', 'HR': 'HR', 'Temp': 'T', 'PH': 'pH', 'AA': 'AA', 'UA': 'UA', 'GLU': 'Glu', 'NH4': 'NH4+', 'GSR': 'GSR', 'ST': 'T'}
    SUMMARY_ORDER_RAW = ['HR', 'GLU', 'LA', 'AA', 'UA', 'PH', 'ST', 'GSR', 'SR', 'SIC']
    CLASS_ORDER_DEFAULT = ['Rest', 'Warmup', 'Exercise', 'Recovery']

    def parse_csv_list(text: str, cast=str) -> List:
        return [cast(x.strip()) for x in str(text).split(',') if x.strip()]

    def parse_pair(text: Optional[str], cast=float) -> Optional[Tuple[float, float]]:
        if text is None or str(text).strip() == '':
            return None
        parts = parse_csv_list(text, cast=cast)
        if len(parts) != 2:
            raise ValueError(f'Expected two comma-separated values, got: {text}')
        return (cast(parts[0]), cast(parts[1]))

    def ensure_dir(p: str | Path) -> Path:
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def safe_name(text: str) -> str:
        text = str(text).strip()
        return re.sub('[^A-Za-z0-9_\\-]+', '_', text) or 'class'

    def apply_style(font='Arial', dpi=600, font_size=7):
        plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False, 'figure.dpi': 120, 'savefig.dpi': dpi, 'axes.linewidth': 0.5, 'axes.titlesize': font_size, 'axes.labelsize': font_size, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size, 'legend.fontsize': font_size, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

    def rename_features(X: pd.DataFrame):
        new_cols = [FEATURE_NAME_MAP.get(c, c) for c in X.columns]
        X2 = X.copy()
        X2.columns = new_cols
        return (X2, new_cols)

    def set_all_fontsizes(fig, size=7, font='Arial'):
        for text in fig.texts:
            text.set_fontsize(size)
            text.set_fontname(font)
        for ax in fig.axes:
            ax.tick_params(axis='both', labelsize=size)
            ax.xaxis.label.set_fontsize(size)
            ax.yaxis.label.set_fontsize(size)
            ax.xaxis.label.set_fontname(font)
            ax.yaxis.label.set_fontname(font)
            ax.title.set_fontsize(size)
            ax.title.set_fontname(font)
            for text in ax.texts:
                text.set_fontsize(size)
                text.set_fontname(font)
            for label in ax.get_xticklabels():
                label.set_fontsize(size)
                label.set_fontname(font)
                label.set_color('black')
            for label in ax.get_yticklabels():
                label.set_fontsize(size)
                label.set_fontname(font)
                label.set_color('black')
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(size)
                    text.set_fontname(font)

    def save_fig(fig, out_base: Path, formats: Sequence[str], dpi=600):
        out_base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            fmt = fmt.lower().strip().lstrip('.')
            if fmt:
                fig.savefig(out_base.with_suffix(f'.{fmt}'), bbox_inches='tight', dpi=dpi)
        plt.close(fig)

    def _looks_like_v15_shap_root(p: Path) -> bool:
        return (p / 'global_4class_shap_raw.npz').exists() or (p / 'sensor_level_shap_plot_data.csv').exists() or any(((p / slug / f'{slug}_shap_decision_plot_data.csv').exists() for slug in ['rest', 'warmup', 'exercise', 'recovery']))

    def find_v15_shap_root(shap_dir: str | Path) -> Path:
        p = Path(shap_dir).expanduser().resolve()
        if p.is_file():
            if p.name == 'global_4class_shap_raw.npz':
                return p.parent
            if p.name in {'sensor_level_shap_plot_data.csv', 'sensor_level_shap_feature_importance.csv'}:
                return p.parent
            if p.name.endswith('_shap_decision_plot_data.csv'):
                return p.parent.parent
        candidates: List[Path] = []
        for c in [p, p / 'shap_outputs', p / 'shap_global_4class', p.parent]:
            if c not in candidates:
                candidates.append(c)
        if p.name.lower() in {'rest', 'warmup', 'exercise', 'recovery'}:
            candidates.insert(0, p.parent)
        for c in candidates:
            if c.exists() and _looks_like_v15_shap_root(c):
                return c
        if p.exists() and p.is_dir():
            hits = list(p.rglob('global_4class_shap_raw.npz'))
            if hits:
                hits = sorted(hits, key=lambda x: (0 if x.parent.name == 'shap_outputs' else 1, len(str(x))))
                return hits[0].parent
        raise FileNotFoundError('Cannot find v15 SHAP output files. Please pass the folder that contains global_4class_shap_raw.npz, usually <v15_out>/shap_outputs. Current input: ' + str(shap_dir))

    def get_sensor_prefix_map() -> Dict[str, List[str]]:
        return {'HR': ['HR'], 'GLU': ['GLU', 'Glu', 'Glucose'], 'LA': ['LA', 'Lac'], 'AA': ['AA'], 'UA': ['UA'], 'PH': ['PH', 'pH'], 'ST': ['Temp', 'ST'], 'GSR': ['GSR', 'GRS'], 'SR': ['SR'], 'SIC': ['SIC']}

    def _feature_matches_alias(feature_name: str, alias: str) -> bool:
        f = str(feature_name)
        a = str(alias)
        return f == a or f.startswith(a + '__') or f.startswith(a + '_')

    def get_sensor_feature_index_map(feature_names: Sequence[str]) -> Dict[str, List[int]]:
        prefix_map = get_sensor_prefix_map()
        sensor_to_indices = {sensor: [] for sensor in SUMMARY_ORDER_RAW}
        matching_order = ['HR', 'GLU', 'LA', 'AA', 'UA', 'PH', 'SIC', 'ST', 'GSR', 'SR']
        for feat_idx, feat_name in enumerate(feature_names):
            feat_name = str(feat_name)
            if feat_name.startswith('SR_valid') or feat_name.startswith('SIC_valid'):
                continue
            for sensor in matching_order:
                aliases = prefix_map.get(sensor, [])
                if any((_feature_matches_alias(feat_name, alias) for alias in aliases)):
                    sensor_to_indices[sensor].append(feat_idx)
                    break
        return sensor_to_indices

    def _as_str_list(arr: object) -> List[str]:
        a = np.asarray(arr, dtype=object).reshape(-1)
        return [str(x) for x in a.tolist()]

    def _normalize_saved_shap_values(shap_values: np.ndarray, n_samples: int, n_classes: int, n_features: int) -> np.ndarray:
        arr = np.asarray(shap_values, dtype=float)
        if arr.shape == (n_samples, n_classes, n_features):
            return arr
        if arr.shape == (n_samples, n_features, n_classes):
            return np.transpose(arr, (0, 2, 1))
        if arr.shape == (n_classes, n_samples, n_features):
            return np.transpose(arr, (1, 0, 2))
        raise ValueError(f'Unsupported shap_values shape={arr.shape}; expected N,K,F ordering from v15.')

    def load_npz_payload(npz_path: Path) -> Dict[str, object]:
        z = np.load(npz_path, allow_pickle=True)
        required = ['shap_values', 'base_values', 'feature_values', 'feature_names', 'class_names']
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f'Missing arrays in {npz_path}: {missing}')
        feature_values = np.asarray(z['feature_values'], dtype=float)
        n_samples, n_features = feature_values.shape
        class_names = _as_str_list(z['class_names'])
        n_classes = len(class_names)
        shap_values = _normalize_saved_shap_values(z['shap_values'], n_samples, n_classes, n_features)
        base_values = np.asarray(z['base_values'], dtype=float)
        if base_values.ndim == 1:
            if base_values.size == n_classes:
                base_values = np.tile(base_values.reshape(1, -1), (n_samples, 1))
            elif base_values.size == n_samples:
                base_values = np.tile(base_values.reshape(-1, 1), (1, n_classes))
            else:
                base_values = np.tile(base_values.reshape(1, -1), (n_samples, 1))
        if base_values.shape[0] != n_samples:
            base_values = np.tile(base_values.reshape(1, -1), (n_samples, 1))
        if base_values.shape[1] < n_classes:
            base_values = np.pad(base_values, ((0, 0), (0, n_classes - base_values.shape[1])), mode='edge')
        base_values = base_values[:, :n_classes]
        return {'shap_values': shap_values, 'base_values': base_values, 'feature_values': feature_values, 'feature_names': _as_str_list(z['feature_names']), 'class_names': class_names, 'sample_uid': _as_str_list(z['sample_uid']) if 'sample_uid' in z.files else [f'sample_{i:06d}' for i in range(n_samples)], 'subject_id': _as_str_list(z['subject_id']) if 'subject_id' in z.files else ['' for _ in range(n_samples)], 'true_label': _as_str_list(z['true_label']) if 'true_label' in z.files else ['' for _ in range(n_samples)], 'pred_label': _as_str_list(z['pred_label']) if 'pred_label' in z.files else ['' for _ in range(n_samples)]}

    def build_native_data_from_v15(root: Path, sensor_order: Sequence[str]) -> Dict[str, object]:
        npz_path = root / 'global_4class_shap_raw.npz'
        if not npz_path.exists():
            raise FileNotFoundError(f'Native-style redraw requires saved raw SHAP tensor: {npz_path}\nPlease run v15 with SHAP output enabled, or pass the correct <v15_out>/shap_outputs folder.')
        payload = load_npz_payload(npz_path)
        shap_values = np.asarray(payload['shap_values'], dtype=float)
        feature_values = np.asarray(payload['feature_values'], dtype=float)
        base_values = np.asarray(payload['base_values'], dtype=float)
        feature_names = [str(x) for x in payload['feature_names']]
        class_names = [str(x) for x in payload['class_names']]
        sensor_to_indices = get_sensor_feature_index_map(feature_names)
        n_samples = shap_values.shape[0]
        X_cols: Dict[str, np.ndarray] = {}
        sv_by_class: Dict[str, np.ndarray] = {c: np.zeros((n_samples, len(sensor_order)), dtype=float) for c in class_names}
        for j, sensor in enumerate(sensor_order):
            idxs = sensor_to_indices.get(sensor, [])
            if idxs:
                with np.errstate(invalid='ignore'):
                    feat_vec = np.nanmean(feature_values[:, idxs], axis=1)
                if np.all(~np.isfinite(feat_vec)):
                    feat_vec = np.zeros(n_samples, dtype=float)
                else:
                    fill_value = float(np.nanmedian(feat_vec[np.isfinite(feat_vec)]))
                    feat_vec = np.where(np.isfinite(feat_vec), feat_vec, fill_value)
                for k, cname in enumerate(class_names):
                    sv_by_class[cname][:, j] = np.sum(shap_values[:, k, idxs], axis=1)
            else:
                feat_vec = np.zeros(n_samples, dtype=float)
                for cname in class_names:
                    sv_by_class[cname][:, j] = 0.0
            X_cols[str(sensor)] = feat_vec.astype(float)
        X_agg = pd.DataFrame(X_cols)
        expected_values = {cname: float(np.nanmean(base_values[:, k])) for k, cname in enumerate(class_names)}
        importance_df = pd.DataFrame(index=list(sensor_order))
        for cname in class_names:
            importance_df[cname] = np.mean(np.abs(sv_by_class[cname]), axis=0)
        importance_df = importance_df.fillna(0.0)
        return {'paths': {'root_dir': root, 'npz': npz_path}, 'X_agg': X_agg, 'class_names': class_names, 'expected_values': expected_values, 'sv_by_class': sv_by_class, 'importance_df': importance_df, 'decision_selected_df': None, 'sample_uid': payload.get('sample_uid', [f'sample_{i:06d}' for i in range(n_samples)]), 'subject_id': payload.get('subject_id', ['' for _ in range(n_samples)]), 'true_label': payload.get('true_label', ['' for _ in range(n_samples)]), 'pred_label': payload.get('pred_label', ['' for _ in range(n_samples)])}
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MaxNLocator
    PAPER_LOW = '#2E73B8'
    PAPER_MID = '#7867B2'
    PAPER_HIGH = '#ED174C'
    PAPER_CMAP = LinearSegmentedColormap.from_list('figure3_blue_to_red', [PAPER_LOW, PAPER_MID, PAPER_HIGH], N=256)
    CLASS_DISPLAY_MAP = {'Rest': 'Rest', 'Warmup': 'Warmup', 'Exercise': 'Exercise', 'Recover': 'Recovery', 'Recovery': 'Recovery'}

    def display_class_name(name: str) -> str:
        return CLASS_DISPLAY_MAP.get(str(name), str(name))

    def _finite_1d(values) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]

    def _complete_xlim(values, requested_xlim: Optional[Tuple[float, float]]=None, pad_frac: float=0.06, min_pad: float=0.12) -> Tuple[float, float]:
        vals = _finite_1d(values)
        if vals.size:
            data_left = float(np.min(vals))
            data_right = float(np.max(vals))
            span = max(data_right - data_left, 1e-09)
            pad = max(float(min_pad), float(pad_frac) * span)
            left = data_left - pad
            right = data_right + pad
        elif requested_xlim is not None:
            left, right = map(float, requested_xlim)
        else:
            left, right = (-1.0, 1.0)
        if requested_xlim is not None:
            left = min(left, float(requested_xlim[0]))
            right = max(right, float(requested_xlim[1]))
        if not np.isfinite(left) or not np.isfinite(right) or right <= left:
            return (-1.0, 1.0)
        return (float(left), float(right))

    def _set_paper_xticks(ax, visible_xlim: Tuple[float, float], xtick_step: Optional[float], requested_xlim: Optional[Tuple[float, float]]=None) -> None:
        left, right = map(float, visible_xlim)
        ax.set_xlim(left, right)
        if xtick_step is not None and float(xtick_step) > 0:
            step = float(xtick_step)
            if requested_xlim is not None:
                anchor = float(requested_xlim[0])
                while anchor - step >= left - 1e-12:
                    anchor -= step
            else:
                anchor = np.floor(left / step) * step
            ticks = np.arange(anchor, right + 0.5 * step, step)
            ticks = ticks[(ticks >= left - 1e-09) & (ticks <= right + 1e-09)]
            if ticks.size >= 2:
                ax.set_xticks(ticks)
                labels = []
                for value in ticks:
                    if abs(value - round(value)) < 1e-09:
                        labels.append(f'{int(round(value))}')
                    else:
                        labels.append(f'{value:g}')
                ax.set_xticklabels(labels)
                return
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 3, 4, 5, 10]))

    def _beeswarm_offsets(x: np.ndarray, feature_index: int, max_spread: float=0.26) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        offsets = np.zeros(len(x), dtype=float)
        finite = np.isfinite(x)
        if finite.sum() <= 1:
            return offsets
        xf = x[finite]
        xmin = float(np.min(xf))
        xmax = float(np.max(xf))
        span = max(xmax - xmin, 1e-12)
        n_bins = 64
        bins = np.floor((xf - xmin) / span * (n_bins - 1)).astype(int)
        finite_ids = np.where(finite)[0]
        rng = np.random.default_rng(1200 + int(feature_index))
        for b in np.unique(bins):
            local = np.where(bins == b)[0]
            ids = finite_ids[local]
            n = len(ids)
            if n == 1:
                offsets[ids] = 0.0
                continue
            spread = min(float(max_spread), 0.025 * np.sqrt(float(n)) + 0.018)
            vals = np.linspace(-spread, spread, n)
            rng.shuffle(vals)
            offsets[ids] = vals
        return offsets

    def _feature_color_values(values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        finite = np.isfinite(vals)
        out = np.full(len(vals), 0.5, dtype=float)
        if not np.any(finite):
            return out
        vf = vals[finite]
        lo, hi = np.nanpercentile(vf, [5.0, 95.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(vf))
            hi = float(np.nanmax(vf))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[finite] = 0.5
            return out
        out[finite] = np.clip((vals[finite] - lo) / (hi - lo), 0.0, 1.0)
        return out

    def _style_common_axis(ax, font: str, font_size: float) -> None:
        ax.tick_params(axis='both', which='major', direction='out', length=2.5, width=0.55, labelsize=font_size, color='black', labelcolor='black')
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            label.set_fontname(font)
            label.set_fontsize(font_size)
            label.set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.spines['bottom'].set_color('black')

    def _draw_summary_on_ax(ax, shap_values: np.ndarray, X: pd.DataFrame, feature_names: Sequence[str], *, title: str, xlim: Optional[Tuple[float, float]], xtick_step: Optional[float], font: str, font_size: float, point_size: float, xlabel: str, show_xlabel: bool=True) -> ScalarMappable:
        sv = np.asarray(shap_values, dtype=float)
        if sv.ndim != 2:
            raise ValueError(f'Summary SHAP array must be 2-D, got shape={sv.shape}')
        if sv.shape[1] != len(feature_names):
            raise ValueError('Summary SHAP columns and feature_names length do not match.')
        n_feat = len(feature_names)
        for y in range(n_feat):
            ax.axhline(y, color='#A7A7A7', linestyle=(0, (3.0, 3.0)), linewidth=0.6, alpha=0.95, zorder=0)
        ax.axvline(0.0, color='#B7B7B7', linewidth=0.7, alpha=0.95, zorder=0)
        for j, feat in enumerate(feature_names):
            x = sv[:, j]
            feat_vals = pd.to_numeric(X.iloc[:, j], errors='coerce').to_numpy(dtype=float)
            c = _feature_color_values(feat_vals)
            y = float(j) + _beeswarm_offsets(x, j, max_spread=0.26)
            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                ax.scatter(x[finite], y[finite], c=c[finite], cmap=PAPER_CMAP, vmin=0.0, vmax=1.0, s=max(float(point_size), 1.0), alpha=0.92, linewidths=0.0, rasterized=False, zorder=2)
        all_vals = sv[np.isfinite(sv)]
        visible_xlim = _complete_xlim(all_vals, requested_xlim=xlim, pad_frac=0.055, min_pad=0.1)
        _set_paper_xticks(ax, visible_xlim, xtick_step, requested_xlim=xlim)
        ax.set_yticks(np.arange(n_feat, dtype=float))
        ax.set_yticklabels(list(feature_names))
        ax.set_ylim(n_feat - 0.45, -0.55)
        ax.set_title(display_class_name(title), fontsize=font_size + 0.4, fontname=font, pad=5)
        ax.set_xlabel(xlabel if show_xlabel else '', fontsize=font_size, fontname=font, labelpad=4)
        ax.tick_params(axis='y', length=0, pad=2)
        _style_common_axis(ax, font, font_size)
        mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=PAPER_CMAP)
        mappable.set_array([])
        return mappable

    def plot_single_class_summary(shap_values: np.ndarray, X: pd.DataFrame, feature_names: List[str], out_dir: Path, out_name: str, *, class_name: str='', max_display: int=10, width_mm: float=50, height_mm: float=72, formats=('png', 'svg'), dpi=600, font='Arial', font_size=7, point_size=4, xlim=None, xtick_step=None, xlabel='SHAP value'):
        n = min(int(max_display), len(feature_names), X.shape[1], np.asarray(shap_values).shape[1])
        feature_names = list(feature_names[:n])
        X_plot = X.iloc[:, :n].copy()
        sv_plot = np.asarray(shap_values, dtype=float)[:, :n]
        fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
        mappable = _draw_summary_on_ax(ax, sv_plot, X_plot, feature_names, title=class_name or out_name, xlim=xlim, xtick_step=xtick_step, font=font, font_size=font_size, point_size=point_size, xlabel=xlabel, show_xlabel=True)
        cbar = fig.colorbar(mappable, ax=ax, pad=0.035, fraction=0.055, aspect=22)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(['Low', 'High'])
        cbar.ax.tick_params(labelsize=font_size, length=0, pad=2)
        cbar.set_label('Feature value', rotation=270, labelpad=9, fontsize=font_size, fontname=font)
        cbar.outline.set_linewidth(0.55)
        fig.subplots_adjust(left=0.24, right=0.84, bottom=0.16, top=0.93)
        save_fig(fig, out_dir / out_name, formats, dpi=dpi)

    def choose_decision_indices(class_name: str, n_total: int, n_take: int, seed: int, decision_selected_df: Optional[pd.DataFrame], use_default_selected: bool=True) -> np.ndarray:
        if use_default_selected and decision_selected_df is not None:
            required = {'class', 'explain_sample_id', 'selected_in_default_decision_plot'}
            if required.issubset(set(decision_selected_df.columns)):
                sub = decision_selected_df[(decision_selected_df['class'].astype(str) == class_name) & decision_selected_df['selected_in_default_decision_plot'].astype(bool)]
                idx = sub['explain_sample_id'].astype(int).to_numpy()
                idx = idx[(idx >= 0) & (idx < n_total)]
                if len(idx) > 0:
                    if n_take is None or int(n_take) <= 0:
                        return idx
                    return idx[:min(len(idx), int(n_take))]
        if n_take is None or int(n_take) <= 0 or int(n_take) >= n_total:
            return np.arange(n_total, dtype=int)
        stable = sum(((i + 1) * ord(ch) for i, ch in enumerate(str(class_name)))) % 100000
        rng = np.random.RandomState(int(seed) + stable)
        idx = np.arange(n_total, dtype=int)
        rng.shuffle(idx)
        return idx[:min(int(n_take), n_total)]

    def _decision_path_arrays(class_name: str, shap_values: np.ndarray, X: pd.DataFrame, expected_value: float, feature_order: Sequence[str], *, decision_n: int, seed: int, decision_selected_df: Optional[pd.DataFrame], use_default_selected: bool) -> Dict[str, object]:
        feature_order = [c for c in feature_order if c in X.columns]
        if not feature_order:
            raise ValueError(f'No valid decision features for class={class_name}')
        path_order = list(reversed(feature_order))
        cols = [X.columns.get_loc(c) for c in path_order]
        sv = np.asarray(shap_values, dtype=float)[:, cols]
        take = choose_decision_indices(class_name=class_name, n_total=len(X), n_take=decision_n, seed=seed, decision_selected_df=decision_selected_df, use_default_selected=use_default_selected)
        sv_take = sv[take, :]
        base = np.full((len(take), 1), float(expected_value), dtype=float)
        cumulative = base + np.cumsum(sv_take, axis=1)
        x_paths = np.concatenate([base, cumulative], axis=1)
        y_path = np.concatenate([np.array([-0.55]), np.arange(len(path_order), dtype=float)])
        outputs = x_paths[:, -1]
        return {'path_order': path_order, 'take': take, 'x_paths': x_paths, 'y_path': y_path, 'outputs': outputs}

    def _draw_decision_on_ax(ax, class_name: str, shap_values: np.ndarray, X: pd.DataFrame, expected_value: float, feature_order: Sequence[str], *, decision_n: int, seed: int, decision_selected_df: Optional[pd.DataFrame], use_default_selected: bool, line_width: float, xlim: Optional[Tuple[float, float]], xtick_step: Optional[float], font: str, font_size: float, xlabel: str, show_xlabel: bool=True) -> Dict[str, object]:
        paths = _decision_path_arrays(class_name, shap_values, X, expected_value, feature_order, decision_n=decision_n, seed=seed, decision_selected_df=decision_selected_df, use_default_selected=use_default_selected)
        x_paths = np.asarray(paths['x_paths'], dtype=float)
        y_path = np.asarray(paths['y_path'], dtype=float)
        outputs = np.asarray(paths['outputs'], dtype=float)
        path_order = list(paths['path_order'])
        visible_xlim = _complete_xlim(x_paths, requested_xlim=xlim, pad_frac=0.05, min_pad=0.18)
        out_finite = outputs[np.isfinite(outputs)]
        if out_finite.size and np.max(out_finite) > np.min(out_finite):
            norm = Normalize(vmin=float(np.min(out_finite)), vmax=float(np.max(out_finite)))
        else:
            norm = Normalize(vmin=0.0, vmax=1.0)
        if len(x_paths) <= 50:
            alpha = 0.62
        elif len(x_paths) <= 150:
            alpha = 0.38
        elif len(x_paths) <= 300:
            alpha = 0.24
        else:
            alpha = 0.13
        segments = [np.column_stack([row, y_path]) for row in x_paths]
        rgba = PAPER_CMAP(norm(outputs))
        rgba[:, 3] = alpha
        lc = LineCollection(segments, colors=rgba, linewidths=max(float(line_width), 0.25), capstyle='round', joinstyle='round', zorder=2)
        ax.add_collection(lc)
        n_feat = len(path_order)
        for y in range(n_feat):
            ax.axhline(y, color='#A7A7A7', linestyle=(0, (3.0, 3.0)), linewidth=0.6, alpha=0.95, zorder=0)
        gradient = np.linspace(0.0, 1.0, 512).reshape(1, -1)
        bar_bottom = n_feat - 0.28
        bar_top = n_feat - 0.08
        ax.imshow(gradient, extent=[visible_xlim[0], visible_xlim[1], bar_bottom, bar_top], aspect='auto', origin='lower', cmap=PAPER_CMAP, vmin=0.0, vmax=1.0, interpolation='bicubic', zorder=4)
        _set_paper_xticks(ax, visible_xlim, xtick_step, requested_xlim=xlim)
        ax.set_yticks(np.arange(n_feat, dtype=float))
        ax.set_yticklabels(path_order)
        ax.set_ylim(-0.78, n_feat + 0.02)
        ax.set_title(display_class_name(class_name), fontsize=font_size + 0.4, fontname=font, pad=5)
        ax.set_xlabel(xlabel if show_xlabel else '', fontsize=font_size, fontname=font, labelpad=4)
        ax.tick_params(axis='y', length=0, pad=2)
        _style_common_axis(ax, font, font_size)
        return paths

    def plot_single_class_decision(class_name: str, shap_values: np.ndarray, X: pd.DataFrame, feature_names: List[str], expected_value: float, feature_order: List[str], out_dir: Path, out_name: str, *, decision_n: int=1600, seed: int=42, decision_selected_df: Optional[pd.DataFrame]=None, use_default_selected: bool=True, width_mm: float=50, height_mm: float=72, formats=('png', 'svg'), dpi=600, font='Arial', font_size=7, line_width=0.45, xlim=None, xtick_step=None, xlabel='Model output'):
        fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
        _draw_decision_on_ax(ax, class_name, shap_values, X, expected_value, feature_order, decision_n=decision_n, seed=seed, decision_selected_df=decision_selected_df, use_default_selected=use_default_selected, line_width=line_width, xlim=xlim, xtick_step=xtick_step, font=font, font_size=font_size, xlabel=xlabel, show_xlabel=True)
        fig.subplots_adjust(left=0.25, right=0.98, bottom=0.14, top=0.93)
        save_fig(fig, out_dir / out_name, formats, dpi=dpi)

    def _labels_to_indices(labels: Sequence[str], class_name: str) -> np.ndarray:
        target = str(class_name).strip().lower()
        labels_arr = np.asarray([str(x).strip() for x in labels], dtype=object)
        idx = [i for i, lab in enumerate(labels_arr) if str(lab).strip().lower() == target]
        return np.asarray(idx, dtype=int)

    def get_state_specific_indices(data: Dict[str, object], class_name: str, label_source: str='true_label') -> np.ndarray:
        if label_source not in {'true_label', 'pred_label'}:
            raise ValueError("label_source must be 'true_label' or 'pred_label'")
        labels = data.get(label_source, [])
        idx = _labels_to_indices(labels, class_name)
        if len(idx) == 0 and label_source == 'true_label':
            idx = _labels_to_indices(data.get('pred_label', []), class_name)
        return idx

    def get_class_specific_summary_axis(args, class_name: str, fallback_xlim: Optional[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        slug = safe_name(class_name).lower()
        xlim_arg_name = f'{slug}_summary_xlim'
        tick_arg_name = f'{slug}_summary_xtick_step'
        xlim_text = getattr(args, xlim_arg_name, None)
        if xlim_text is None:
            xlim = fallback_xlim
        elif str(xlim_text).strip() == '':
            xlim = None
        else:
            xlim = parse_pair(xlim_text, float)
        xtick_step = getattr(args, tick_arg_name, None)
        if xtick_step is None:
            xtick_step = args.summary_xtick_step
        return (xlim, xtick_step)

    def plot_combined_figure3(data: Dict[str, object], class_names: Sequence[str], X_disp: pd.DataFrame, summary_order_disp: Sequence[str], summary_col_idx: Sequence[int], args, out_dir: Path, formats: Sequence[str], summary_xlim_fallback: Optional[Tuple[float, float]]) -> None:
        n_cols = len(class_names)
        if n_cols == 0:
            return
        fig = plt.figure(figsize=(12.0, 7.2))
        gs = fig.add_gridspec(2, n_cols + 1, width_ratios=[1.0] * n_cols + [0.075], height_ratios=[1.0, 1.0], left=0.055, right=0.965, bottom=0.075, top=0.965, wspace=0.3, hspace=0.38)
        top_axes = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
        bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]
        cax = fig.add_subplot(gs[0, n_cols])
        ax_blank = fig.add_subplot(gs[1, n_cols])
        ax_blank.axis('off')
        mappable = None
        for i, cname in enumerate(class_names):
            sv = np.asarray(data['sv_by_class'][cname], dtype=float)
            sv_summary = sv[:, list(summary_col_idx)]
            X_summary = X_disp.loc[:, list(summary_order_disp)].copy()
            class_summary_xlim, class_summary_step = get_class_specific_summary_axis(args, cname, summary_xlim_fallback)
            mappable = _draw_summary_on_ax(top_axes[i], sv_summary, X_summary, summary_order_disp, title=display_class_name(cname), xlim=class_summary_xlim, xtick_step=class_summary_step, font=args.font, font_size=args.font_size, point_size=max(args.point_size, 3.0), xlabel='SHAP value', show_xlabel=True)
            top_feats_raw = data['importance_df'][cname].sort_values(ascending=False).index.astype(str).tolist()[:min(args.max_display, len(data['importance_df']))]
            top_feats_disp = [FEATURE_NAME_MAP.get(f, f) for f in top_feats_raw]
            _draw_decision_on_ax(bottom_axes[i], cname, sv, X_disp, float(data['expected_values'][cname]), top_feats_disp, decision_n=args.decision_n, seed=args.seed, decision_selected_df=data['decision_selected_df'], use_default_selected=not args.no_default_selected, line_width=args.decision_line_width, xlim=parse_pair(args.decision_xlim, float) if str(args.decision_xlim).strip() else None, xtick_step=args.decision_xtick_step, font=args.font, font_size=args.font_size, xlabel='Model output', show_xlabel=True)
        if mappable is None:
            mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=PAPER_CMAP)
            mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=cax, orientation='vertical')
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(['Low', 'High'])
        cbar.ax.tick_params(labelsize=args.font_size, length=0, pad=2)
        cbar.set_label('Feature value', rotation=270, labelpad=11, fontsize=args.font_size, fontname=args.font)
        cbar.outline.set_linewidth(0.6)
        fig.text(0.012, 0.975, 'A', fontsize=args.font_size + 5, fontweight='bold', va='top', ha='left', fontname=args.font)
        fig.text(0.012, 0.495, 'B', fontsize=args.font_size + 5, fontweight='bold', va='top', ha='left', fontname=args.font)
        save_fig(fig, out_dir / 'Figure3_SHAP_4state', formats, dpi=args.dpi)

    def main():
        parser = argparse.ArgumentParser(description='Redraw v15 saved 4-class SHAP with original style, plus 4 state-only decision plots.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--shap_dir', required=True, help='Path to v15 output root, v15 shap_outputs, or global_4class_shap_raw.npz.')
        parser.add_argument('--out_dir', default=None, help='Default: <shap_outputs>/restyled_summary_decision_native_from_v15')
        parser.add_argument('--class_name', default=None, help='Only draw one class if specified, e.g. Rest')
        parser.add_argument('--max_display', type=int, default=10)
        parser.add_argument('--decision_n', type=int, default=1600, help='0 means draw all saved samples; 80 matches the old drawing script default.')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--formats', default='png,svg,pdf')
        parser.add_argument('--dpi', type=int, default=600)
        parser.add_argument('--font', default='Arial')
        parser.add_argument('--font_size', type=float, default=7)
        parser.add_argument('--point_size', type=float, default=4)
        parser.add_argument('--decision_line_width', type=float, default=0.45)
        parser.add_argument('--summary_xlim', default='-3,6', help='Default all-sample summary x-axis, e.g. "-3,6"; set empty to auto')
        parser.add_argument('--summary_xtick_step', type=float, default=4, help='Default tick step for all-sample summary plots.')
        parser.add_argument('--rest_summary_xlim', default=None, help='All-sample Rest summary x-axis, e.g. "-3,5"; omit to use --summary_xlim')
        parser.add_argument('--warmup_summary_xlim', default=None, help='All-sample Warmup summary x-axis, e.g. "-3,5"; omit to use --summary_xlim')
        parser.add_argument('--exercise_summary_xlim', default=None, help='All-sample Exercise summary x-axis, e.g. "-3,6"; omit to use --summary_xlim')
        parser.add_argument('--recovery_summary_xlim', default=None, help='All-sample Recovery summary x-axis, e.g. "-3,6"; omit to use --summary_xlim')
        parser.add_argument('--rest_summary_xtick_step', type=float, default=None, help='Tick step for all-sample Rest summary; omit to use --summary_xtick_step.')
        parser.add_argument('--warmup_summary_xtick_step', type=float, default=None, help='Tick step for all-sample Warmup summary; omit to use --summary_xtick_step.')
        parser.add_argument('--exercise_summary_xtick_step', type=float, default=None, help='Tick step for all-sample Exercise summary; omit to use --summary_xtick_step.')
        parser.add_argument('--recovery_summary_xtick_step', type=float, default=None, help='Tick step for all-sample Recovery summary; omit to use --summary_xtick_step.')
        parser.add_argument('--decision_xlim', default='-6,7.5', help='e.g. "-6,7.5"; set empty to auto; used by the original all-sample decision plots unless overridden in code')
        parser.add_argument('--decision_xtick_step', type=float, default=4)
        parser.add_argument('--state_decision_xlim', default='-6,6', help='Only for the 4 state-only decision plots, e.g. "-6,6"; set empty to auto')
        parser.add_argument('--state_decision_xtick_step', type=float, default=3, help='Only for the 4 state-only decision plots.')
        parser.add_argument('--summary_xlabel', default='SHAP value')
        parser.add_argument('--decision_xlabel', default='Model output')
        parser.add_argument('--summary_width_mm', type=float, default=50)
        parser.add_argument('--summary_height_mm', type=float, default=72)
        parser.add_argument('--decision_width_mm', type=float, default=50)
        parser.add_argument('--decision_height_mm', type=float, default=72)
        parser.add_argument('--skip_summary', action='store_true')
        parser.add_argument('--skip_decision', action='store_true')
        parser.add_argument('--no_default_selected', action='store_true')
        parser.add_argument('--skip_state_only_decision', action='store_true', help='Do not draw the 4 extra decision plots that only contain samples from the matching state.')
        parser.add_argument('--state_decision_n', type=int, default=0, help='Rows per state-only decision plot. 0 means all rows from that state.')
        parser.add_argument('--state_filter', choices=['true_label', 'pred_label'], default='true_label', help='Which saved label decides the rows used in the 4 state-only decision plots.')
        parser.add_argument('--summary_order', default=','.join(SUMMARY_ORDER_RAW), help='Fixed summary order, raw sensor names. Decision order is still class-specific importance order.')
        args = parser.parse_args()
        root = find_v15_shap_root(args.shap_dir)
        sensor_order = [s for s in parse_csv_list(args.summary_order, str) if s]
        if not sensor_order:
            sensor_order = list(SUMMARY_ORDER_RAW)
        data = build_native_data_from_v15(root, sensor_order)
        out_dir = Path(args.out_dir) if args.out_dir else root / 'restyled_summary_decision_native_from_v15'
        ensure_dir(out_dir)
        apply_style(font=args.font, dpi=args.dpi, font_size=args.font_size)
        formats = parse_csv_list(args.formats, str)
        summary_xlim = parse_pair(args.summary_xlim, float) if str(args.summary_xlim).strip() else None
        decision_xlim = parse_pair(args.decision_xlim, float) if str(args.decision_xlim).strip() else None
        state_decision_xlim = parse_pair(args.state_decision_xlim, float) if str(args.state_decision_xlim).strip() else None
        X_disp, feature_names = rename_features(data['X_agg'])
        summary_order_raw = [f for f in sensor_order if f in data['X_agg'].columns]
        summary_order_disp = [FEATURE_NAME_MAP.get(f, f) for f in summary_order_raw]
        summary_col_idx = [X_disp.columns.get_loc(f) for f in summary_order_disp]
        class_names = list(data['class_names'])
        _display_rank = {name: i for i, name in enumerate(CLASS_ORDER_DEFAULT)}
        class_names = sorted(class_names, key=lambda c: (_display_rank.get(display_class_name(c), 999), str(c)))
        if args.class_name is not None:
            if args.class_name not in class_names:
                raise ValueError(f'class_name={args.class_name!r} not found. Available: {class_names}')
            class_names = [args.class_name]
        if not args.skip_decision and (not args.skip_state_only_decision):
            state_counts = {c: int(len(get_state_specific_indices(data, c, args.state_filter))) for c in class_names}
        for cname in class_names:
            shap_values = data['sv_by_class'][cname]
            top_feats_raw = data['importance_df'][cname].sort_values(ascending=False).index.astype(str).tolist()[:min(args.max_display, len(data['importance_df']))]
            top_feats_disp = [FEATURE_NAME_MAP.get(f, f) for f in top_feats_raw]
            if not args.skip_summary:
                shap_values_summary = shap_values[:, summary_col_idx]
                X_summary = X_disp.loc[:, summary_order_disp].copy()
                class_summary_xlim, class_summary_xtick_step = get_class_specific_summary_axis(args, cname, summary_xlim)
                plot_single_class_summary(shap_values=shap_values_summary, X=X_summary, feature_names=summary_order_disp, out_dir=out_dir, out_name=f'shap_summary_native__{safe_name(display_class_name(cname))}', class_name=display_class_name(cname), max_display=min(args.max_display, len(summary_order_disp)), width_mm=args.summary_width_mm, height_mm=args.summary_height_mm, formats=formats, dpi=args.dpi, font=args.font, font_size=args.font_size, point_size=args.point_size, xlim=class_summary_xlim, xtick_step=class_summary_xtick_step, xlabel=args.summary_xlabel)
            if not args.skip_decision:
                plot_single_class_decision(class_name=cname, shap_values=shap_values, X=X_disp, feature_names=feature_names, expected_value=float(data['expected_values'][cname]), feature_order=top_feats_disp, out_dir=out_dir, out_name=f'shap_decision_native__{safe_name(display_class_name(cname))}_n{args.decision_n}', decision_n=args.decision_n, seed=args.seed, decision_selected_df=data['decision_selected_df'], use_default_selected=not args.no_default_selected, width_mm=args.decision_width_mm, height_mm=args.decision_height_mm, formats=formats, dpi=args.dpi, font=args.font, font_size=args.font_size, line_width=args.decision_line_width, xlim=decision_xlim, xtick_step=args.decision_xtick_step, xlabel=args.decision_xlabel)
                if not args.skip_state_only_decision:
                    state_idx = get_state_specific_indices(data, cname, args.state_filter)
                    if len(state_idx) == 0:
                        pass
                    else:
                        X_state = X_disp.iloc[state_idx, :].reset_index(drop=True)
                        sv_state = shap_values[state_idx, :]
                        n_label = 'all' if int(args.state_decision_n) <= 0 else str(int(args.state_decision_n))
                        plot_single_class_decision(class_name=cname, shap_values=sv_state, X=X_state, feature_names=feature_names, expected_value=float(data['expected_values'][cname]), feature_order=top_feats_disp, out_dir=out_dir, out_name=f'shap_decision_native_state_only__{safe_name(display_class_name(cname))}_{args.state_filter}_n{n_label}', decision_n=args.state_decision_n, seed=args.seed, decision_selected_df=None, use_default_selected=False, width_mm=args.decision_width_mm, height_mm=args.decision_height_mm, formats=formats, dpi=args.dpi, font=args.font, font_size=args.font_size, line_width=args.decision_line_width, xlim=state_decision_xlim, xtick_step=args.state_decision_xtick_step, xlabel=args.decision_xlabel)
        if not args.skip_summary and (not args.skip_decision):
            plot_combined_figure3(data=data, class_names=class_names, X_disp=X_disp, summary_order_disp=summary_order_disp, summary_col_idx=summary_col_idx, args=args, out_dir=out_dir, formats=formats, summary_xlim_fallback=summary_xlim)
    main()

def run_all_in_one():
    import sys as _sys
    _base = Path(__file__).resolve().parent
    _out = _base / 'outputs_v15-1'
    _old_argv = list(_sys.argv)
    try:
        _sys.argv = ['classification', '--excel_dir', str(_base / 'out_1Hz-1'), '--excel_pattern', '*.xlsx', '--out', str(_out), '--shap_explain_n', '0', '--shap_decision_n', '0']
        classification_main()
    finally:
        _sys.argv = _old_argv
    _old_argv = list(_sys.argv)
    try:
        _sys.argv = ['shap_redraw', '--shap_dir', str(_out / 'shap_outputs'), '--out_dir', str(_out / 'shap_outputs' / 'Figure3_style'), '--summary_xlim=-3,6', '--summary_xtick_step', '3', '--rest_summary_xlim=-3,4.5', '--rest_summary_xtick_step', '3', '--warmup_summary_xlim=-3,4', '--warmup_summary_xtick_step', '3', '--exercise_summary_xlim=-2.5,4', '--exercise_summary_xtick_step', '2', '--recovery_summary_xlim=-2,3', '--recovery_summary_xtick_step', '1.5', '--decision_xlim=-6,7', '--decision_xtick_step', '4', '--formats', 'png,svg,pdf', '--dpi', '600', '--skip_state_only_decision']
        run_embedded_shap_redraw()
    finally:
        _sys.argv = _old_argv
if __name__ == '__main__':
    run_all_in_one()
