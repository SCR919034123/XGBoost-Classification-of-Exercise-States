from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from itertools import combinations
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def decode_hash_unicode_name(s: str) -> str:
    return re.sub('#U([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), str(s))

def subject_name_from_path(excel_path: str | Path) -> str:
    return decode_hash_unicode_name(Path(excel_path).stem)

def compute_sample_weights_regression(y: np.ndarray) -> np.ndarray:
    return np.ones(len(y), dtype=float)

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

def rpe_to_level(rpe: float) -> int:
    if rpe is None or (isinstance(rpe, float) and np.isnan(rpe)):
        return -1
    r = float(rpe)
    if 6 <= r <= 7:
        return 0
    if 8 <= r <= 10:
        return 1
    if 11 <= r <= 12:
        return 2
    if 13 <= r <= 14:
        return 3
    if 15 <= r <= 17:
        return 4
    if r >= 18:
        return 5
    return -1

def clip_rpe(x: float, lo: float=6.0, hi: float=20.0) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float('nan')
    return float(np.clip(float(x), lo, hi))
DEFAULT_RPE_BINS = [6, 8, 11, 13, 15, 18, 20]

def parse_rpe_bins(bin_str: str) -> List[float]:
    vals = [float(x.strip()) for x in str(bin_str).split(',') if str(x).strip()]
    if len(vals) < 2:
        raise ValueError('--fatigue_bins must contain at least two numeric edges, e.g. 6,8,11,13,15,18,20')
    vals = sorted(vals)
    if vals[0] > 6:
        raise ValueError('The first fatigue bin edge should usually start at 6 for Borg RPE.')
    return vals

def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-09:
        return str(int(round(x)))
    return f'{x:g}'

def level_labels_from_bins(bin_edges: List[float]) -> List[str]:
    labels = []
    for i in range(len(bin_edges) - 1):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if abs(lo - round(lo)) < 1e-09 and abs(hi - round(hi)) < 1e-09:
            lo_i = int(round(lo))
            hi_i = int(round(hi))
            if i == len(bin_edges) - 2:
                txt = f'{lo_i}-{hi_i}'
            else:
                txt = f'{lo_i}-{hi_i - 1}'
        else:
            txt = f'{_fmt_num(lo)}-{_fmt_num(hi)}'
        labels.append(f'L{i}({txt})')
    return labels

def rpe_to_level_by_bins(rpe: float, bin_edges: List[float]) -> int:
    if rpe is None or (isinstance(rpe, float) and np.isnan(rpe)):
        return -1
    r = float(rpe)
    if r < bin_edges[0] or r > bin_edges[-1]:
        return -1
    for i in range(len(bin_edges) - 1):
        lo = float(bin_edges[i])
        hi = float(bin_edges[i + 1])
        if i == len(bin_edges) - 2:
            if lo <= r <= hi:
                return i
        elif lo <= r < hi:
            return i
    return -1

def build_merged_1hz(excel_path: str | Path, out_csv: str | Path, sweat_cols: List[str], sweat_delay_sec: int=300) -> pd.DataFrame:
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
    tmin = min((df['time'].min() for df in sensor_dfs.values()))
    tmax = max((df['time'].max() for df in sensor_dfs.values()))
    timeline = pd.date_range(start=tmin.floor('s'), end=tmax.ceil('s'), freq='1s')
    merged = pd.DataFrame({'time': timeline}).set_index('time')
    for sh, df in sensor_dfs.items():
        d = df.set_index('time').sort_index()
        s = d[[sh]].resample('1s').mean()
        s = s.interpolate(method='time').ffill().bfill()
        merged = merged.join(s, how='left')
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
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding='utf-8-sig')
    return merged

def per_subject_sensor_zscore(df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in sensor_cols:
        x = pd.to_numeric(out[c], errors='coerce')
        arr = x.to_numpy(dtype=float)
        if len(arr) == 0 or np.all(~np.isfinite(arr)):
            out[c] = 0.0
            continue
        mu = np.nanmean(arr)
        sd = np.nanstd(arr)
        if not np.isfinite(mu):
            mu = 0.0
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        out[c] = (x - mu) / sd
        out[c] = out[c].fillna(0.0)
    return out

def ewma_features(x: np.ndarray, alpha=0.05) -> Tuple[float, float]:
    if len(x) == 0:
        return (0.0, 0.0)
    s = pd.Series(x)
    m = s.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    v = s.ewm(alpha=alpha, adjust=False).var().iloc[-1]
    if not np.isfinite(v):
        v = 0.0
    return (float(m), float(np.sqrt(v)))

def block_stats(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(mean=0.0, std=0.0, min=0.0, max=0.0, last=0.0, delta=0.0, slope=0.0)
    t = np.arange(len(x), dtype=float)
    slope = 0.0
    if len(x) >= 2:
        slope = np.polyfit(t, x, 1)[0]
    return dict(mean=float(np.mean(x)), std=float(np.std(x)), min=float(np.min(x)), max=float(np.max(x)), last=float(x[-1]), delta=float(x[-1] - x[0]), slope=float(slope))

def split_segments(arr: np.ndarray, n_segments: int) -> List[np.ndarray]:
    if n_segments <= 1:
        return [arr]
    L = len(arr)
    segs = []
    for i in range(n_segments):
        a = int(i * L / n_segments)
        b = int((i + 1) * L / n_segments)
        segs.append(arr[a:b])
    return segs

def make_sample_dataset_for_substage_vote(df_1hz: pd.DataFrame, subject_id: str, short_windows: List[int]=[60, 180], n_segments: int=3, ewma_alpha: float=0.05, sample_step_sec: int=5, history_sec: int=600) -> pd.DataFrame:
    df = df_1hz.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df[df['label'] != 'Unknown'].reset_index(drop=True)
    meta_cols = {'time', 'stage_raw', 'label', 'stage_key', 'rpe'}
    sensor_cols = [c for c in df.columns if c not in meta_cols]
    df = per_subject_sensor_zscore(df, sensor_cols)
    rows = []
    for skey, g in df.groupby('stage_key'):
        g = g.sort_values('time').reset_index(drop=True)
        if len(g) < 5:
            continue
        rpe_sub = float(np.nanmedian(g['rpe'].to_numpy(dtype=float)))
        if not np.isfinite(rpe_sub):
            rpe_sub = parse_rpe(skey)
        rpe_start = float(g['rpe'].iloc[0]) if np.isfinite(g['rpe'].iloc[0]) else rpe_sub
        idxs = np.arange(0, len(g), sample_step_sec, dtype=int)
        if len(idxs) == 0:
            idxs = np.array([len(g) - 1], dtype=int)
        for ii in idxs:
            t_center = g.loc[ii, 'time']
            label = g.loc[ii, 'label']
            rpe_now = float(g.loc[ii, 'rpe']) if np.isfinite(g.loc[ii, 'rpe']) else rpe_sub
            if not np.isfinite(rpe_now):
                continue
            rpe_now = clip_rpe(rpe_now, 6.0, 20.0)
            rpe_level = rpe_to_level(rpe_now)
            if rpe_level < 0:
                continue
            t0 = t_center - pd.Timedelta(seconds=history_sec)
            hist = df[(df['time'] <= t_center) & (df['time'] >= t0)].copy()
            if len(hist) < 30:
                continue
            feats = {'subject_id': subject_id, 't_center': t_center, 'stage_key': skey, 'label': label, 'rpe_value': float(rpe_now), 'rpe_level': int(rpe_level), 'rpe_substage': float(rpe_sub) if np.isfinite(rpe_sub) else float('nan'), 'rpe_delta': float(rpe_now - rpe_start) if np.isfinite(rpe_start) else 0.0}
            for col in sensor_cols:
                x_all = hist[col].to_numpy(dtype=float)
                st_all = block_stats(x_all)
                for k, v in st_all.items():
                    feats[f'{col}__hist__{k}'] = v
                m_ew, s_ew = ewma_features(x_all, alpha=ewma_alpha)
                feats[f'{col}__ewma_mean'] = m_ew
                feats[f'{col}__ewma_std'] = s_ew
                for w in short_windows:
                    x_w = x_all[-w:] if len(x_all) >= w else x_all
                    st_w = block_stats(x_w)
                    for k, v in st_w.items():
                        feats[f'{col}__w{w}__{k}'] = v
                if 60 in short_windows and 180 in short_windows:
                    for k in ['mean', 'slope', 'last']:
                        feats[f'{col}__diff60_180__{k}'] = feats.get(f'{col}__w60__{k}', 0.0) - feats.get(f'{col}__w180__{k}', 0.0)
                base_len = max(short_windows) if len(short_windows) > 0 else len(x_all)
                x_base = x_all[-base_len:] if len(x_all) >= base_len else x_all
                segs = split_segments(x_base, n_segments)
                for si, seg in enumerate(segs):
                    st_s = block_stats(seg)
                    for k, v in st_s.items():
                        feats[f'{col}__seg{si + 1}_{n_segments}__{k}'] = v
            rows.append(feats)
    return pd.DataFrame(rows)

def train_xgb_regressor(X, y, sample_weight=None, seed=42, n_estimators: int=700, learning_rate: float=0.025, max_depth: int=4) -> XGBRegressor:
    reg = XGBRegressor(objective='reg:squarederror', n_estimators=int(n_estimators), learning_rate=float(learning_rate), max_depth=int(max_depth), subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0, min_child_weight=2.0, gamma=0.0, tree_method='hist', random_state=seed)
    reg.fit(X, y, sample_weight=sample_weight)
    return reg

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str | Path, title: str):
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    row_sum[row_sum == 0] = 1.0
    cm_pct = cm / row_sum * 100.0
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, cmap='Reds', vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)')
    dark_thresh = 65.0
    red_num = '#d62728'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = float(cm_pct[i, j])
            txt_color = 'white' if pct >= dark_thresh else red_num
            ax.text(j, i, f'{count}\n({pct:.1f}%)', ha='center', va='center', fontsize=11, color=txt_color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_confusion_matrix_counts(cm: np.ndarray, labels: List[str], out_path: str | Path, title: str):
    cm = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('True level')
    ax.set_ylabel('Predicted level')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Count')
    vmax = max(1, int(cm.max()))
    thresh = vmax * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(j, i, f'{val}', ha='center', va='center', fontsize=11, color='white' if val >= thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_fatigue_partition_scatter(pred_df: pd.DataFrame, bin_edges: List[float], out_path: str | Path, title: str, *, rpe_lo: float=6.0, rpe_hi: float=20.0, fatigue_threshold: float=13.0):
    df = pred_df.copy()
    df = df[np.isfinite(pd.to_numeric(df['true_rpe'], errors='coerce')) & np.isfinite(pd.to_numeric(df['pred_rpe'], errors='coerce'))].copy()
    if len(df) == 0:
        return
    df['true_level'] = [rpe_to_level_by_bins(v, bin_edges) for v in df['true_rpe'].to_numpy(dtype=float)]
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    low_w = max(0.0, float(fatigue_threshold) - float(rpe_lo))
    high_w = max(0.0, float(rpe_hi) - float(fatigue_threshold))
    if low_w > 0:
        ax.add_patch(plt.Rectangle((rpe_lo, rpe_lo), low_w, low_w, facecolor='0.93', edgecolor='none', zorder=0))
    if high_w > 0:
        ax.add_patch(plt.Rectangle((fatigue_threshold, fatigue_threshold), high_w, high_w, facecolor='0.93', edgecolor='none', zorder=0))
    for edge in bin_edges:
        ax.axvline(edge, linestyle='--', linewidth=1.0, color='gray', alpha=0.6, zorder=1)
        ax.axhline(edge, linestyle='--', linewidth=1.0, color='gray', alpha=0.6, zorder=1)
    ax.plot([rpe_lo, rpe_hi], [rpe_lo, rpe_hi], linewidth=1.8, color='black', zorder=2)
    sc = ax.scatter(df['true_rpe'].to_numpy(dtype=float), df['pred_rpe'].to_numpy(dtype=float), c=df['true_level'].to_numpy(dtype=float), cmap='coolwarm', s=30, alpha=0.8, edgecolors='none', zorder=3)
    ax.set_xlim(rpe_lo, rpe_hi)
    ax.set_ylim(rpe_lo, rpe_hi)
    ax.set_xlabel('True RPE')
    ax.set_ylabel('Predicted RPE')
    ax.set_title(title)
    ax.text(rpe_lo + 0.3, rpe_lo + low_w - 0.6 if low_w > 1 else rpe_lo + 0.5, 'Non-fatigue', fontsize=11, va='top')
    ax.text(fatigue_threshold + 0.3, rpe_hi - 0.5, 'Fatigue', fontsize=11, va='top')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('True fatigue level')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def eval_substage_rpe_regression(ds: pd.DataFrame, feature_cols: List[str], label_order: List[str], reg: XGBRegressor, out_dir: Path, tag: str, *, level_edges: List[float], rpe_clip_lo: float=6.0, rpe_clip_hi: float=20.0, plot_band: float=1.0, report_level: bool=True):
    drop_cols = ['rpe_level', 'label', 'subject_id', 't_center', 'stage_key', 'rpe_value', 'rpe_substage', 'rpe_delta']
    X = ds.drop(columns=drop_cols, errors='ignore')
    X = X.reindex(columns=feature_cols, fill_value=0.0)
    y_true_sample = ds['rpe_value'].to_numpy(dtype=float)
    y_pred_sample = reg.predict(X).astype(float)
    y_pred_sample = np.clip(y_pred_sample, rpe_clip_lo, rpe_clip_hi)
    rows = []
    group_cols = ['subject_id', 'stage_key'] if 'subject_id' in ds.columns else ['stage_key']
    for group_key, g in ds.assign(y_true=y_true_sample, y_pred=y_pred_sample).groupby(group_cols, sort=False):
        if isinstance(group_key, tuple):
            sid, skey = group_key
        else:
            sid, skey = (str(g['subject_id'].iloc[0]) if 'subject_id' in g.columns else '', group_key)
        yt = float(np.nanmedian(g['y_true'].to_numpy(dtype=float)))
        yp = float(np.nanmedian(g['y_pred'].to_numpy(dtype=float)))
        if not np.isfinite(yt) or not np.isfinite(yp):
            continue
        rows.append({'subject_id': str(sid), 'stage_key': skey, 'stage_label': str(g['label'].iloc[0]) if 'label' in g.columns else '', 'true_rpe': yt, 'pred_rpe': yp, 'error': yp - yt, 'abs_error': abs(yp - yt), 'n_samples_in_stage': int(len(g))})
    pred_df = pd.DataFrame(rows)
    if len(pred_df) == 0:
        return {'mae_substage': float('nan'), 'rmse_substage': float('nan'), 'r2_substage': float('nan'), 'confidence_in_band': float('nan'), 'n_substages': 0, 'n_samples': int(len(ds))}
    y_true_sub = pred_df['true_rpe'].to_numpy(dtype=float)
    y_pred_sub = pred_df['pred_rpe'].to_numpy(dtype=float)
    mae = mean_absolute_error(y_true_sub, y_pred_sub)
    rmse = float(np.sqrt(mean_squared_error(y_true_sub, y_pred_sub)))
    r2 = r2_score(y_true_sub, y_pred_sub) if len(y_true_sub) >= 2 else float('nan')
    conf = float(np.mean(np.abs(y_pred_sub - y_true_sub) <= float(plot_band)))
    pred_df.to_csv(out_dir / f'substage_predictions_rpe_reg_{tag}.csv', index=False, encoding='utf-8-sig')
    out_xlsx = out_dir / f'pred_vs_true_rpe_{tag}.xlsx'
    metrics_df = pd.DataFrame([{'tag': tag, 'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2) if np.isfinite(r2) else float('nan'), 'band': float(plot_band), 'confidence_in_band': float(conf), 'n_substages': int(len(pred_df)), 'n_samples': int(len(ds)), 'rpe_clip_lo': float(rpe_clip_lo), 'rpe_clip_hi': float(rpe_clip_hi)}])
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
        pred_df.to_excel(w, index=False, sheet_name='pred_vs_true')
        metrics_df.to_excel(w, index=False, sheet_name='metrics')
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y_true_sub, y_pred_sub, facecolors='none', edgecolors='red', s=55)
    lo = float(rpe_clip_lo)
    hi = float(rpe_clip_hi)
    ax.plot([lo, hi], [lo, hi], linewidth=2)
    b = float(plot_band)
    ax.plot([lo, hi], [lo + b, hi + b], linestyle='--', linewidth=1.8)
    ax.plot([lo, hi], [lo - b, hi - b], linestyle='--', linewidth=1.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('True value (RPE)')
    ax.set_ylabel('Predicted value (RPE)')
    ax.text(0.05, 0.95, f'Confidence (|err|≤{b:g}) = {conf * 100:.1f}%\n$R^2$ score = {r2 * 100:.1f}%\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}', transform=ax.transAxes, va='top', fontsize=12)
    fig.tight_layout()
    out_png = out_dir / f'pred_vs_true_rpe_{tag}.png'
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    out_metrics = {'mae_substage': float(mae), 'rmse_substage': float(rmse), 'r2_substage': float(r2) if np.isfinite(r2) else float('nan'), 'confidence_in_band': float(conf), 'n_substages': int(len(pred_df)), 'n_samples': int(len(ds))}
    if report_level:
        y_true_level = np.array([rpe_to_level_by_bins(v, level_edges) for v in y_true_sub], dtype=int)
        y_pred_level = np.array([rpe_to_level_by_bins(v, level_edges) for v in y_pred_sub], dtype=int)
        ok = (y_true_level >= 0) & (y_pred_level >= 0)
        y_true_level = y_true_level[ok]
        y_pred_level = y_pred_level[ok]
        if len(y_true_level) > 0:
            macro_f1 = f1_score(y_true_level, y_pred_level, average='macro')
            micro_f1 = f1_score(y_true_level, y_pred_level, average='micro')
            cm = confusion_matrix(y_true_level, y_pred_level, labels=list(range(len(label_order))))
            plot_confusion_matrix(cm, label_order, out_dir / f'confusion_matrix_level_from_rpe_{tag}.png', title=f'Confusion Matrix (Level from Regression RPE, Substage-median) [{tag}]')
            report = classification_report(y_true_level, y_pred_level, labels=list(range(len(label_order))), target_names=label_order, digits=4)
            (out_dir / f'classification_report_level_from_rpe_{tag}.txt').write_text(report, encoding='utf-8')
            out_metrics['macro_f1_substage_level'] = float(macro_f1)
            out_metrics['micro_f1_substage_level'] = float(micro_f1)
    return out_metrics

def _safe_import_shap():
    try:
        import shap
        return shap
    except Exception as e:
        return None

def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)

def parse_int_list(s: Optional[str]) -> List[int]:
    if s is None or str(s).strip() == '':
        return []
    out = []
    for item in str(s).split(','):
        item = item.strip()
        if not item:
            continue
        v = int(item)
        if v > 0:
            out.append(v)
    return sorted(set(out))
BASE_PARAMS = ['LA', 'SR', 'SIC', 'GRS', 'HR', 'Temp', 'PH', 'AA', 'UA', 'GLU']

def aggregate_shap_by_base_feature(shap_values: np.ndarray, X: pd.DataFrame, base_params: List[str], *, agg: str='sum') -> Tuple[np.ndarray, pd.DataFrame]:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    X_df = X.copy()
    shap_out = {}
    X_out = {}
    for bp in base_params:
        cols = [c for c in X_df.columns if c == bp or c.startswith(f'{bp}__')]
        if len(cols) == 0:
            continue
        if agg == 'mean':
            shap_out[bp] = shap_df[cols].mean(axis=1)
            X_out[bp] = X_df[cols].mean(axis=1)
        else:
            shap_out[bp] = shap_df[cols].sum(axis=1)
            X_out[bp] = X_df[cols].mean(axis=1)
    if len(shap_out) == 0:
        return (shap_values, X_df)
    keys = list(shap_out.keys())
    shap_agg = np.vstack([shap_out[k] for k in keys]).T
    X_agg = pd.DataFrame({k: X_out[k] for k in keys})
    return (shap_agg, X_agg)

def make_shap_plots_rpe_regression(reg: XGBRegressor, X_train_full: pd.DataFrame, val_all_ds: pd.DataFrame, feature_cols: List[str], out_dir: Path, seed: int, drop_cols_train: List[str], shap_bg_n: int=500, shap_explain_n: int=0, shap_max_display: int=15, shap_decision_n: int=0, shap_decision_ns: Optional[List[int]]=None):
    shap = _safe_import_shap()
    if shap is None:
        return
    out_shap = ensure_dir(out_dir / 'shap_rpe_reg')
    X_val = val_all_ds.drop(columns=drop_cols_train, errors='ignore')
    X_val = X_val.reindex(columns=feature_cols, fill_value=0.0)
    X_bg = _sample_df(X_train_full, min(shap_bg_n, len(X_train_full)), seed)
    X_explain = _sample_df(X_val, int(shap_explain_n), seed)
    try:
        try:
            expl = shap.TreeExplainer(reg, data=X_bg, feature_perturbation='interventional')
        except TypeError:
            expl = shap.TreeExplainer(reg, data=X_bg)
        shap_vals_raw = expl.shap_values(X_explain)
        shap_vals_raw = np.asarray(shap_vals_raw)
        sv_agg, X_agg = aggregate_shap_by_base_feature(shap_values=shap_vals_raw, X=X_explain, base_params=BASE_PARAMS, agg='sum')
        base_val = expl.expected_value
        base_val_f = float(base_val) if not isinstance(base_val, (list, np.ndarray)) else float(np.asarray(base_val).ravel()[0])
        mean_abs_shap = np.mean(np.abs(sv_agg), axis=0)
        fi_df = pd.DataFrame({'feature': list(X_agg.columns), 'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=False)
        fi_df.to_csv(out_shap / 'shap_feature_importance_aggregated.csv', index=False, encoding='utf-8-sig')
        meta_cols = [c for c in ['subject_id', 't_center', 'stage_key', 'label', 'rpe_value', 'rpe_level', 'rpe_substage', 'rpe_delta'] if c in val_all_ds.columns]
        meta = val_all_ds.loc[X_explain.index, meta_cols].reset_index(drop=True).copy()
        meta.insert(0, 'sample_row_index', X_explain.index.to_numpy())
        try:
            meta['model_pred_rpe_raw'] = reg.predict(X_explain).astype(float)
        except Exception:
            pass
        shap_wide = meta.copy()
        for j, fn in enumerate(list(X_agg.columns)):
            shap_wide[f'SHAP_{fn}'] = sv_agg[:, j]
        shap_wide.to_csv(out_shap / 'shap_values_aggregated_samples.csv', index=False, encoding='utf-8-sig')
        value_wide = meta.copy()
        for fn in list(X_agg.columns):
            value_wide[f'VALUE_{fn}'] = X_agg[fn].to_numpy(dtype=float)
        value_wide.to_csv(out_shap / 'shap_feature_values_aggregated_samples.csv', index=False, encoding='utf-8-sig')
        long_parts = []
        base_meta_cols = [c for c in ['sample_row_index', 'subject_id', 't_center', 'stage_key', 'label', 'rpe_value', 'rpe_level', 'model_pred_rpe_raw'] if c in meta.columns]
        for j, fn in enumerate(list(X_agg.columns)):
            tmp = meta[base_meta_cols].copy()
            tmp['feature'] = fn
            tmp['shap_value'] = sv_agg[:, j]
            tmp['feature_value'] = X_agg[fn].to_numpy(dtype=float)
            long_parts.append(tmp)
        if len(long_parts) > 0:
            pd.concat(long_parts, ignore_index=True).to_csv(out_shap / 'shap_values_aggregated_long.csv', index=False, encoding='utf-8-sig')
        (out_shap / 'shap_base_value.json').write_text(json.dumps({'base_value': base_val_f, 'n_explained_samples': int(len(X_agg)), 'features': list(X_agg.columns)}, ensure_ascii=False, indent=2), encoding='utf-8')
        plt.figure()
        shap.summary_plot(sv_agg, X_agg, feature_names=list(X_agg.columns), show=False, max_display=min(shap_max_display, X_agg.shape[1]))
        plt.title('SHAP Summary (Regression: Predicted RPE) - Aggregated by Parameter')
        plt.tight_layout()
        plt.savefig(out_shap / 'shap_summary.png', dpi=300)
        plt.close()
        decision_ns = list(shap_decision_ns) if shap_decision_ns else []
        decision_ns = sorted(set([int(n) for n in decision_ns if int(n) > 0]))
        if len(decision_ns) == 0 and int(shap_decision_n) > 0:
            decision_ns = [int(shap_decision_n)]
        if len(decision_ns) == 0:
            decision_ns = [len(X_agg)]
        rng = np.random.RandomState(seed)
        idx = np.arange(len(X_agg), dtype=int)
        rng.shuffle(idx)
        for n_lines_requested in decision_ns:
            take = idx[:min(int(n_lines_requested), len(idx))]
            sv_mat = sv_agg[take, :]
            x_mat = X_agg.iloc[take, :]
            if len(take) <= 50:
                line_alpha = 0.8
            elif len(take) <= 150:
                line_alpha = 0.45
            elif len(take) <= 300:
                line_alpha = 0.25
            else:
                line_alpha = 0.15
            plt.figure(figsize=(10, 6))
            try:
                shap.decision_plot(base_value=base_val_f, shap_values=sv_mat, features=x_mat, feature_names=list(X_agg.columns), show=False, alpha=line_alpha, ignore_warnings=True)
            except TypeError:
                shap.decision_plot(base_value=base_val_f, shap_values=sv_mat, features=x_mat, feature_names=list(X_agg.columns), show=False, ignore_warnings=True)
            plt.title(f'SHAP Decision (Multi-sample, Regression) | n={len(take)}')
            plt.xlabel('Model output (Predicted RPE)')
            plt.tight_layout()
            out_png_n = out_shap / f'shap_decision_multisample_rpe_n{len(take)}.png'
            plt.savefig(out_png_n, dpi=300)
            if n_lines_requested == decision_ns[0]:
                plt.savefig(out_shap / 'shap_decision_multisample_rpe.png', dpi=300)
            plt.close()
    except Exception as e:
        pass
FEATURE_NAME_MAP = {'LA': 'LA', 'SR': 'SR', 'SIC': 'SIC', 'GRS': 'GSR', 'HR': 'HR', 'Temp': 'T', 'PH': 'pH', 'AA': 'AA', 'UA': 'UA', 'GLU': 'Glu'}

def load_saved_shap(shap_dir: Path):
    shap_values_path = shap_dir / 'shap_values_aggregated_samples.csv'
    feature_values_path = shap_dir / 'shap_feature_values_aggregated_samples.csv'
    base_value_path = shap_dir / 'shap_base_value.json'
    if not shap_values_path.exists():
        raise FileNotFoundError(f'Missing: {shap_values_path}')
    if not feature_values_path.exists():
        raise FileNotFoundError(f'Missing: {feature_values_path}')
    if not base_value_path.exists():
        raise FileNotFoundError(f'Missing: {base_value_path}')
    shap_df = pd.read_csv(shap_values_path, encoding='utf-8-sig')
    value_df = pd.read_csv(feature_values_path, encoding='utf-8-sig')
    shap_cols = [c for c in shap_df.columns if c.startswith('SHAP_')]
    features = [c.replace('SHAP_', '', 1) for c in shap_cols]
    value_cols = [f'VALUE_{f}' for f in features]
    shap_values = shap_df[shap_cols].to_numpy(dtype=float)
    X = value_df[value_cols].copy()
    X.columns = features
    with open(base_value_path, 'r', encoding='utf-8') as f:
        base_value = float(json.load(f)['base_value'])
    return (shap_values, X, features, base_value)

def set_all_fontsizes(fig, size=7):
    for text in fig.texts:
        text.set_fontsize(size)
        text.set_fontname('Arial')
    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=size)
        ax.xaxis.label.set_fontsize(size)
        ax.yaxis.label.set_fontsize(size)
        ax.title.set_fontsize(size)
        for text in ax.texts:
            text.set_fontsize(size)
            text.set_fontname('Arial')
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(size)

def apply_style(font='Arial', dpi=600):
    plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False, 'figure.dpi': 120, 'savefig.dpi': dpi, 'axes.linewidth': 0.5, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7})

def rename_features(X: pd.DataFrame, features):
    renamed = [FEATURE_NAME_MAP.get(f, f) for f in features]
    X2 = X.copy()
    X2.columns = renamed
    return (X2, renamed)

def save_fig(fig, out_base: Path, formats):
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f'.{fmt}'), bbox_inches='tight')
    plt.close(fig)

def plot_summary(shap_values, X, feature_names, out_dir: Path, max_display=10, width=67 / 25.4, height=52 / 25.4, formats=('png', 'svg')):
    font_size = 5
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=min(max_display, X.shape[1]), plot_size=(width, height))
    fig = plt.gcf()
    ax = max(fig.axes, key=lambda a: a.get_position().width)
    for collection in ax.collections:
        collection.set_alpha(1)
        collection.set_sizes([2])
    ax.set_axisbelow(True)
    for line in ax.lines:
        xdata = np.asarray(line.get_xdata())
        ydata = np.asarray(line.get_ydata())
        if len(ydata) > 1 and np.allclose(ydata, ydata[0]):
            line.set_color('0.55')
            line.set_linewidth(0.5)
            line.set_linestyle((0, (4, 4)))
            line.set_alpha(0.7)
            line.set_zorder(-2)
        elif len(xdata) > 1 and np.allclose(xdata, xdata[0]):
            line.set_color('0.8')
            line.set_linewidth(0.5)
            line.set_linestyle('-')
            line.set_alpha(1)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True, labeltop=False, direction='out', length=3, width=0.5, pad=3, labelsize=font_size, color='black', labelcolor='black')
    x_left = -4.0
    x_right = 2.0
    ax.set_xlim(x_left, x_right)
    x_step = 2
    xticks = np.arange(x_left, x_right + 1e-09, x_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{x:.0f}' for x in xticks])
    ax.tick_params(axis='y', labelsize=font_size, labelcolor='black', pad=2, length=0)
    for label in ax.get_yticklabels():
        label.set_fontsize(font_size)
        label.set_fontname('Arial')
        label.set_color('black')
        label.set_horizontalalignment('right')
    ax.set_title('')
    ax.set_xlabel('SHAP value contribution to predicted RPE', fontsize=font_size, labelpad=6, color='black')
    cbar_axes = [a for a in fig.axes if a is not ax]
    for cbar_ax in cbar_axes:
        cbar_pos = cbar_ax.get_position()
        cbar_width = 0.08
        cbar_left = 0.86
        cbar_ax.set_position([cbar_left, cbar_pos.y0, cbar_width, cbar_pos.height])
        cbar_ax.tick_params(labelsize=font_size, width=0.5, length=2, color='black', labelcolor='black')
        cbar_ax.set_ylabel('Feature value', fontsize=font_size, color='black', labelpad=3)
        for label in cbar_ax.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(font_size)
            label.set_color('black')
    fig.subplots_adjust(left=0.27, right=0.84, bottom=0.22, top=0.96)
    fig.canvas.draw()
    for label in ax.get_xticklabels():
        label.set_fontsize(7)
        label.set_fontname('Arial')
        label.set_color('black')
    for label in ax.get_yticklabels():
        label.set_fontsize(7)
        label.set_fontname('Arial')
        label.set_color('black')
    ax.xaxis.label.set_fontsize(7)
    ax.xaxis.label.set_fontname('Arial')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_fontsize(7)
    ax.yaxis.label.set_fontname('Arial')
    ax.yaxis.label.set_color('black')
    ax.title.set_fontsize(7)
    ax.title.set_text('')
    for other_ax in fig.axes:
        if other_ax is ax:
            continue
        for label in other_ax.get_xticklabels():
            label.set_fontsize(7)
            label.set_fontname('Arial')
            label.set_color('black')
        for label in other_ax.get_yticklabels():
            label.set_fontsize(7)
            label.set_fontname('Arial')
            label.set_color('black')
        other_ax.xaxis.label.set_fontsize(7)
        other_ax.yaxis.label.set_fontsize(7)
        other_ax.xaxis.label.set_fontname('Arial')
        other_ax.yaxis.label.set_fontname('Arial')
    set_all_fontsizes(fig, size=font_size)
    fig.canvas.draw()
    for t in fig.texts:
        pass
    for label in ax.get_yticklabels():
        pass
    save_fig(fig, out_dir / 'shap_summary_restyle', formats)

def plot_bar(shap_values, feature_names, out_dir: Path, max_display=10, width=6.5, height=4.2, formats=('png', 'svg')):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    plot_names = [feature_names[i] for i in order][::-1]
    plot_values = mean_abs[order][::-1]
    fig, ax = plt.subplots(figsize=(width, height))
    ax.barh(plot_names, plot_values)
    ax.set_xlabel('Mean absolute SHAP value')
    ax.set_title('Global Feature Importance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / 'shap_bar_restyle', formats)

def plot_decision(shap_values, X, feature_names, base_value, out_dir: Path, decision_ns=None, seed=42, width=47 / 25.4, height=57 / 25.4, formats=('png', 'svg')):
    if len(X) == 0:
        return
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    if decision_ns is None:
        decision_ns = []
    decision_ns = [int(n) for n in decision_ns if int(n) > 0]
    if len(decision_ns) == 0:
        decision_ns = [len(X)]
    x_start = 6
    x_end = 20
    x_step = 4
    for n in decision_ns:
        take = idx[:min(n, len(idx))]
        if len(take) <= 50:
            alpha = 0.8
        elif len(take) <= 150:
            alpha = 0.45
        elif len(take) <= 300:
            alpha = 0.25
        else:
            alpha = 0.15
        plt.figure(figsize=(width, height))
        try:
            shap.decision_plot(base_value=base_value, shap_values=shap_values[take, :], features=X.iloc[take, :], feature_names=feature_names, show=False, alpha=alpha, ignore_warnings=True, auto_size_plot=False, xlim=(x_start, x_end))
        except TypeError:
            shap.decision_plot(base_value=base_value, shap_values=shap_values[take, :], features=X.iloc[take, :], feature_names=feature_names, show=False, ignore_warnings=True, auto_size_plot=False, xlim=(x_start, x_end))
        fig = plt.gcf()
        fig.set_size_inches(width, height, forward=True)
        ax = plt.gca()
        ticks = np.arange(x_start, x_end + 1e-09, x_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{x:.0f}' for x in ticks])
        ticks = np.arange(x_start, x_end + 1e-09, x_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{x:.0f}' for x in ticks])
        font_size = 7
        path_linewidth = 0.5
        path_alpha = alpha
        for line in ax.lines:
            ydata = np.asarray(line.get_ydata())
            is_horizontal = len(ydata) > 1 and np.allclose(ydata, ydata[0])
            if is_horizontal:
                line.set_color('0.55')
                line.set_linewidth(0.5)
                line.set_linestyle((0, (4, 4)))
                line.set_alpha(0.7)
                line.set_zorder(-2)
            else:
                line.set_linewidth(path_linewidth)
                line.set_alpha(path_alpha)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True, labeltop=False, direction='out', length=1, width=0.5, pad=3, labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size, pad=2)
        for label in ax.get_yticklabels():
            label.set_fontsize(font_size)
            label.set_fontname('Arial')
            label.set_horizontalalignment('right')
        ax.set_xlabel('Model output: predicted RPE', fontsize=font_size, labelpad=6)
        fig.subplots_adjust(left=0.27, right=0.98, bottom=0.22, top=0.84)
        ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True, labeltop=False, direction='out', length=3, width=0.5, pad=3, labelsize=font_size, color='black', labelcolor='black')
        ax.set_xlabel('Model output: predicted RPE', fontsize=font_size, labelpad=6, color='black')
        ax.tick_params(axis='y', labelcolor='black')
        for label in ax.get_yticklabels():
            label.set_color('black')
        save_fig(fig, out_dir / f'shap_decision_restyle_n{len(take)}', formats)

def run_shap_restyle(shap_dir, max_display=10, decision_ns=None, font='Arial', dpi=600, formats='png,svg', seed=42):
    global shap
    shap = _safe_import_shap()
    if shap is None:
        return
    shap_dir = Path(shap_dir)
    out_dir = shap_dir / 'restyled'
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [x.strip().lower() for x in formats.split(',') if x.strip()]
    if decision_ns is None:
        decision_ns = []
    elif isinstance(decision_ns, str):
        decision_ns = [int(x.strip()) for x in decision_ns.split(',') if x.strip() and int(x.strip()) > 0]
    else:
        decision_ns = [int(x) for x in decision_ns if int(x) > 0]
    apply_style(font=font, dpi=dpi)
    shap_values, X, features, base_value = load_saved_shap(shap_dir)
    X_renamed, feature_names = rename_features(X, features)
    plot_summary(shap_values, X_renamed, feature_names, out_dir, max_display=max_display, formats=formats)
    plot_bar(shap_values, feature_names, out_dir, max_display=max_display, formats=formats)
    plot_decision(shap_values, X_renamed, feature_names, base_value, out_dir, decision_ns=decision_ns, seed=seed, formats=formats)

def main():
    ap = argparse.ArgumentParser(description='RPE regression with Warmup+Exercise targets and subject-level LOSO-CV.')
    ap.add_argument('--excel_dir', type=str, default='./out_1Hz-1', help='Directory containing *_1Hz.xlsx files.')
    ap.add_argument('--excel_pattern', type=str, default='*.xlsx', help='Excel glob pattern under --excel_dir.')
    ap.add_argument('--recursive', action='store_true', help='Search --excel_dir recursively.')
    ap.add_argument('--excels', nargs='*', default=None, help='Optional explicit Excel file list.')
    ap.add_argument('--out', default='./RPE_Prediction_Results')
    ap.add_argument('--no_cache', action='store_true', help='force rebuilding cached merged/sample datasets')
    ap.add_argument('--cv', choices=['loso', 'holdout'], default='loso', help='loso = leave-one-subject-out; holdout = use --val_subjects only.')
    ap.add_argument('--val_subjects', nargs='*', default=None, help='Optional held-out subject name(s). Match file stem, e.g. 刘兆研_1Hz. If provided with --cv loso, only these LOSO folds are run.')
    ap.add_argument('--max_folds', type=int, default=0, help='For speed testing. 1 runs only the first fold. Use 0 or -1 to run all folds.')
    ap.add_argument('--target_stages', type=str, default='Warmup,Exercise', help='Comma-separated stage labels used for RPE regression. Default: Warmup,Exercise')
    ap.add_argument('--sweat_delay', type=int, default=120, help='seconds, shift(-delay) for sweat cols')
    ap.add_argument('--sweat_cols', type=str, default='LA,PH,AA,UA,GLU')
    ap.add_argument('--short_windows', type=str, default='30,60,120,180,300')
    ap.add_argument('--n_segments', type=int, default=3)
    ap.add_argument('--ewma_alpha', type=float, default=0.05)
    ap.add_argument('--sample_step', type=int, default=5, help='seconds, sampling step inside each substage')
    ap.add_argument('--history_sec', type=int, default=900, help='use only recent history (sec)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_estimators', type=int, default=700)
    ap.add_argument('--learning_rate', type=float, default=0.025)
    ap.add_argument('--max_depth', type=int, default=4)
    ap.add_argument('--rpe_lo', type=float, default=6.0)
    ap.add_argument('--rpe_hi', type=float, default=20.0)
    ap.add_argument('--plot_band', type=float, default=1.0, help='band for confidence: |pred-true|<=band')
    ap.add_argument('--do_shap', action='store_true', help='enable SHAP plots for the first fold only')
    ap.add_argument('--do_global_shap', action='store_true', help='train one model on all subjects and generate overall SHAP figures')
    ap.add_argument('--shap_bg_n', type=int, default=300, help='background samples for SHAP')
    ap.add_argument('--shap_explain_n', type=int, default=0, help='samples to explain for SHAP; 0 means use all available samples')
    ap.add_argument('--shap_max_display', type=int, default=15, help='max features shown in SHAP plots')
    ap.add_argument('--shap_decision_n', type=int, default=0, help='how many samples(lines) in ONE decision plot; 0 means use all explained samples')
    ap.add_argument('--shap_decision_ns', type=str, default=None, help='optional comma-separated decision-plot sample counts; by default all explained samples are used')
    ap.add_argument('--fatigue_bins', type=str, default='6,8,11,13,15,18,20', help='RPE bin edges for fatigue-level plots/confusion matrix. Example: 6,8,11,13,15,18,20')
    ap.add_argument('--fatigue_threshold', type=float, default=13.0, help='RPE threshold separating non-fatigue and fatigue in the partition scatter plot')
    ap.add_argument('--disable_level_report', action='store_true', help='disable derived RPE-level confusion matrix')
    args = ap.parse_args()
    out_root = ensure_dir(args.out)
    cache_dir = ensure_dir(out_root / 'cache_subjects')
    fatigue_bins = parse_rpe_bins(args.fatigue_bins)
    label_order = level_labels_from_bins(fatigue_bins)
    sweat_cols = [x.strip() for x in args.sweat_cols.split(',') if x.strip()]
    short_windows = [int(x) for x in args.short_windows.split(',') if x.strip()]
    target_stages = [x.strip() for x in args.target_stages.split(',') if x.strip()]
    shap_decision_ns = parse_int_list(args.shap_decision_ns)
    excel_paths: List[Path] = []
    if args.excels:
        excel_paths.extend([Path(p) for p in args.excels])
    if args.excel_dir:
        root = Path(args.excel_dir)
        globber = root.rglob if args.recursive else root.glob
        excel_paths.extend(sorted(globber(args.excel_pattern)))
    excel_paths = sorted({p.resolve() for p in excel_paths if p.exists() and p.suffix.lower() in ['.xlsx', '.xls']})
    if len(excel_paths) == 0:
        raise ValueError('No Excel files found. Use --excel_dir/--excel_pattern or --excels.')
    subject_ds: Dict[str, pd.DataFrame] = {}
    for p in excel_paths:
        sid = subject_name_from_path(p)
        merged_csv = cache_dir / f'merged_1hz_{sid}.csv'
        ds_csv = cache_dir / f'dataset_samples_{sid}.csv'
        ds = None
        if not args.no_cache and ds_csv.exists() and (ds_csv.stat().st_size > 0):
            try:
                ds = pd.read_csv(ds_csv)
            except Exception as e:
                ds = None
        if ds is None:
            merged = build_merged_1hz(p, merged_csv, sweat_cols=sweat_cols, sweat_delay_sec=args.sweat_delay)
            ds = make_sample_dataset_for_substage_vote(merged, subject_id=sid, short_windows=short_windows, n_segments=args.n_segments, ewma_alpha=args.ewma_alpha, sample_step_sec=args.sample_step, history_sec=args.history_sec)
        if len(ds) > 0:
            ds = ds[ds['label'].astype(str).isin(target_stages)].copy()
            ds = ds[np.isfinite(pd.to_numeric(ds['rpe_value'], errors='coerce'))].copy()
            ds['rpe_value'] = np.clip(ds['rpe_value'].astype(float), args.rpe_lo, args.rpe_hi)
            ds['rpe_level'] = ds['rpe_value'].apply(lambda v: rpe_to_level_by_bins(v, fatigue_bins)).astype(int)
            ds = ds[ds['rpe_level'] >= 0].reset_index(drop=True)
        ds.to_csv(cache_dir / f'dataset_samples_{sid}.csv', index=False, encoding='utf-8-sig')
        if len(ds) == 0:
            continue
        subject_ds[sid] = ds
    subjects = sorted(subject_ds.keys())
    if len(subjects) < 2:
        raise ValueError(f'Need at least 2 valid subjects, but got {len(subjects)}: {subjects}')
    if args.cv == 'holdout':
        if not args.val_subjects:
            raise ValueError('--cv holdout requires --val_subjects')
        val_subjects = [decode_hash_unicode_name(s) for s in args.val_subjects]
        missing = [s for s in val_subjects if s not in subject_ds]
        if missing:
            raise ValueError(f'val_subjects not found after filtering: {missing}. Available: {subjects}')
        folds = [val_subjects]
    elif args.val_subjects:
        requested = [decode_hash_unicode_name(s) for s in args.val_subjects]
        missing = [s for s in requested if s not in subject_ds]
        if missing:
            raise ValueError(f'val_subjects not found after filtering: {missing}. Available: {subjects}')
        folds = [[s] for s in requested]
    else:
        folds = [[s] for s in subjects]
    if args.max_folds is not None and args.max_folds > 0:
        folds = folds[:int(args.max_folds)]
    else:
        pass
    summary_rows = []
    all_pred_dfs = []
    drop_cols_train = ['rpe_level', 'label', 'subject_id', 't_center', 'stage_key', 'rpe_value', 'rpe_substage', 'rpe_delta']
    shap_saved_once = False
    report_level = not args.disable_level_report
    for split_id, val_subjects in enumerate(folds, start=1):
        train_subjects = [s for s in subjects if s not in val_subjects]
        split_name = f"loso_{split_id:02d}_val_{'_'.join(val_subjects)}" if len(val_subjects) == 1 else f"holdout_{split_id:02d}_val_{'_'.join(val_subjects)}"
        out_dir = ensure_dir(out_root / split_name)
        train_ds = pd.concat([subject_ds[s] for s in train_subjects], ignore_index=True)
        val_ds_dict = {s: subject_ds[s].copy() for s in val_subjects}
        val_all = pd.concat(list(val_ds_dict.values()), ignore_index=True)
        X_train = train_ds.drop(columns=drop_cols_train, errors='ignore')
        X_train = X_train.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feature_cols = list(X_train.columns)
        y_train = train_ds['rpe_value'].to_numpy(dtype=float)
        y_train = np.clip(y_train, args.rpe_lo, args.rpe_hi)
        w = compute_sample_weights_regression(y_train)
        reg = train_xgb_regressor(X_train, y_train, sample_weight=w, seed=args.seed, n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth)
        reg.save_model(str(out_dir / 'model_rpe_reg.json'))
        if args.do_shap and (not shap_saved_once):
            make_shap_plots_rpe_regression(reg=reg, X_train_full=X_train, val_all_ds=val_all, feature_cols=feature_cols, out_dir=out_dir, seed=args.seed, drop_cols_train=drop_cols_train, shap_bg_n=args.shap_bg_n, shap_explain_n=args.shap_explain_n, shap_max_display=args.shap_max_display, shap_decision_n=args.shap_decision_n, shap_decision_ns=shap_decision_ns)
            shap_saved_once = True
        for sid in val_subjects:
            m = eval_substage_rpe_regression(val_ds_dict[sid], feature_cols=feature_cols, label_order=label_order, reg=reg, out_dir=out_dir, tag=f'val_{sid}', level_edges=fatigue_bins, rpe_clip_lo=args.rpe_lo, rpe_clip_hi=args.rpe_hi, plot_band=args.plot_band, report_level=report_level)
            metrics = {'split': split_name, 'val_subject': sid, **m}
            (out_dir / f'metrics_rpe_reg_{sid}.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        overall = eval_substage_rpe_regression(val_all, feature_cols=feature_cols, label_order=label_order, reg=reg, out_dir=out_dir, tag='val_ALL', level_edges=fatigue_bins, rpe_clip_lo=args.rpe_lo, rpe_clip_hi=args.rpe_hi, plot_band=args.plot_band, report_level=report_level)
        fold_pred_path = out_dir / 'substage_predictions_rpe_reg_val_ALL.csv'
        if fold_pred_path.exists():
            pdf = pd.read_csv(fold_pred_path)
            pdf['split'] = split_name
            all_pred_dfs.append(pdf)
        summary_rows.append({'split': split_name, 'val_subject': ','.join(val_subjects), 'train_subjects_n': len(train_subjects), 'mae_substage': overall.get('mae_substage', float('nan')), 'rmse_substage': overall.get('rmse_substage', float('nan')), 'r2_substage': overall.get('r2_substage', float('nan')), 'confidence_in_band': overall.get('confidence_in_band', float('nan')), 'band': float(args.plot_band), 'n_substages': overall.get('n_substages', 0), 'n_samples': overall.get('n_samples', 0), 'target_stages': ','.join(target_stages), 'short_windows': args.short_windows, 'n_segments': args.n_segments, 'ewma_alpha': args.ewma_alpha, 'sample_step': args.sample_step, 'history_sec': args.history_sec, 'sweat_delay': args.sweat_delay, 'sweat_cols': ','.join(sweat_cols), 'target': 'rpe_value (regression)'})
    if args.do_global_shap:
        all_ds_for_shap = pd.concat([subject_ds[s] for s in subjects], ignore_index=True)
        X_all_shap = all_ds_for_shap.drop(columns=drop_cols_train, errors='ignore')
        X_all_shap = X_all_shap.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_all_shap = np.clip(all_ds_for_shap['rpe_value'].to_numpy(dtype=float), args.rpe_lo, args.rpe_hi)
        reg_all_shap = train_xgb_regressor(X_all_shap, y_all_shap, sample_weight=compute_sample_weights_regression(y_all_shap), seed=args.seed, n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth)
        out_global_shap = ensure_dir(out_root / 'global_interpretation')
        reg_all_shap.save_model(str(out_global_shap / 'model_rpe_reg_all_subjects.json'))
        make_shap_plots_rpe_regression(reg=reg_all_shap, X_train_full=X_all_shap, val_all_ds=all_ds_for_shap, feature_cols=list(X_all_shap.columns), out_dir=out_global_shap, seed=args.seed, drop_cols_train=drop_cols_train, shap_bg_n=args.shap_bg_n, shap_explain_n=args.shap_explain_n, shap_max_display=args.shap_max_display, shap_decision_n=args.shap_decision_n, shap_decision_ns=shap_decision_ns)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / 'summary_rpe_reg_loso.csv', index=False, encoding='utf-8-sig')
    if len(all_pred_dfs) > 0:
        heldout_df = pd.concat(all_pred_dfs, ignore_index=True)
        heldout_df.to_csv(out_root / 'heldout_predictions_rpe_substage_all.csv', index=False, encoding='utf-8-sig')
        y_true = heldout_df['true_rpe'].to_numpy(dtype=float)
        y_pred = heldout_df['pred_rpe'].to_numpy(dtype=float)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else float('nan')
        conf = float(np.mean(np.abs(y_pred - y_true) <= float(args.plot_band)))
        metrics_all = {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2) if np.isfinite(r2) else float('nan'), 'confidence_in_band': float(conf), 'band': float(args.plot_band), 'n_substages': int(len(heldout_df)), 'n_folds_completed': int(len(folds)), 'target_stages': ','.join(target_stages)}
        (out_root / 'heldout_metrics_rpe_substage_all.json').write_text(json.dumps(metrics_all, ensure_ascii=False, indent=2), encoding='utf-8')
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(y_true, y_pred, facecolors='none', edgecolors='red', s=55)
        lo = float(args.rpe_lo)
        hi = float(args.rpe_hi)
        b = float(args.plot_band)
        ax.plot([lo, hi], [lo, hi], linewidth=2)
        ax.plot([lo, hi], [lo + b, hi + b], linestyle='--', linewidth=1.8)
        ax.plot([lo, hi], [lo - b, hi - b], linestyle='--', linewidth=1.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('True value (RPE)')
        ax.set_ylabel('Predicted value (RPE)')
        ax.set_title('Held-out RPE prediction across completed LOSO folds')
        ax.text(0.05, 0.95, f'Confidence (|err|≤{b:g}) = {conf * 100:.1f}%\n$R^2$ score = {r2 * 100:.1f}%\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}', transform=ax.transAxes, va='top', fontsize=12)
        fig.tight_layout()
        fig.savefig(out_root / 'heldout_pred_vs_true_rpe_substage_all.png', dpi=300)
        plt.close(fig)
        plot_fatigue_partition_scatter(heldout_df, fatigue_bins, out_root / 'heldout_fatigue_partition_map.png', title='Fatigue-level partition map (held-out predictions across completed LOSO folds)', rpe_lo=args.rpe_lo, rpe_hi=args.rpe_hi, fatigue_threshold=args.fatigue_threshold)
        heldout_df['true_level'] = [rpe_to_level_by_bins(v, fatigue_bins) for v in heldout_df['true_rpe'].to_numpy(dtype=float)]
        heldout_df['pred_level'] = [rpe_to_level_by_bins(v, fatigue_bins) for v in heldout_df['pred_rpe'].to_numpy(dtype=float)]
        heldout_ok = heldout_df[(heldout_df['true_level'] >= 0) & (heldout_df['pred_level'] >= 0)].copy()
        if len(heldout_ok) > 0:
            cm_all = confusion_matrix(heldout_ok['pred_level'].to_numpy(dtype=int), heldout_ok['true_level'].to_numpy(dtype=int), labels=list(range(len(label_order))))
            plot_confusion_matrix_counts(cm_all, label_order, out_root / 'heldout_confusion_matrix_levels_all.png', title='Overall confusion matrix of predicted vs. true fatigue levels')
            cm_df = pd.DataFrame(cm_all, index=[f'Pred_{x}' for x in label_order], columns=[f'True_{x}' for x in label_order])
            cm_df.to_csv(out_root / 'heldout_confusion_matrix_levels_all.csv', encoding='utf-8-sig')
    global_shap_dir = out_root / 'global_interpretation' / 'shap_rpe_reg'
    if global_shap_dir.exists():
        try:
            run_shap_restyle(shap_dir=global_shap_dir, max_display=10, decision_ns=None, font='Arial', dpi=600, formats='png,svg', seed=args.seed)
        except Exception as e:
            pass
    if len(all_pred_dfs) > 0:
        pass
if __name__ == '__main__':
    main()
