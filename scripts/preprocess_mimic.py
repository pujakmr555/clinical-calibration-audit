#!/usr/bin/env python3
"""MIMIC-IV → model-ready tabular dataset for mortality prediction.

Usage:
    python scripts/preprocess_mimic.py --mimic_dir /path/to/mimic-iv --output data/mimic_cohort.parquet

This script:
1. Selects adult first-ICU-stay cohort with ≥48h stay
2. Extracts vitals + labs from first 48h, aggregates to tabular features
3. Adds demographics (age, sex, ethnicity, insurance) for subgroup analysis
4. Outputs a single parquet file matching the pipeline's expected schema

IMPORTANT: Never commit the output file. data/ is gitignored.
"""
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Vital sign itemids in MIMIC-IV chartevents ──
VITAL_ITEMS = {
    220045: "hr",           # Heart Rate
    220050: "sbp",          # Systolic BP (arterial)
    220051: "dbp",          # Diastolic BP (arterial)
    220052: "map",          # Mean Arterial Pressure
    220179: "sbp_ni",       # Systolic BP (non-invasive)
    220180: "dbp_ni",       # Diastolic BP (non-invasive)
    220181: "map_ni",       # MAP (non-invasive)
    220210: "rr",           # Respiratory Rate
    220277: "spo2",         # SpO2
    223761: "temp_f",       # Temperature (F)
    223762: "temp_c",       # Temperature (C)
    220739: "gcs_eye",      # GCS - Eye
    223900: "gcs_verbal",   # GCS - Verbal
    223901: "gcs_motor",    # GCS - Motor
}

# ── Lab itemids in MIMIC-IV labevents ──
LAB_ITEMS = {
    50862: "albumin",
    50868: "aniongap",
    50882: "bicarbonate",
    50885: "bilirubin",
    50893: "calcium",
    50902: "chloride",
    50912: "creatinine",
    50931: "glucose",
    50960: "magnesium",
    50970: "phosphate",
    50971: "potassium",
    50983: "sodium",
    51006: "bun",
    51221: "hematocrit",
    51222: "hemoglobin",
    51248: "mcv",
    51249: "mchc",
    51265: "platelet",
    51277: "rdw",
    51279: "rbc",
    51300: "wbc",
    51301: "wbc_auto",
    50802: "pao2_bg",       # from blood gas
    50804: "paco2_bg",
    50813: "lactate",
    50818: "ph",
}

AGG_FUNCS = ["min", "max", "mean", "last"]


def load_cohort(mimic_dir):
    """Select adult, first ICU stay, ≥48h LOS, with known outcome."""
    print("Loading patients, admissions, icustays...")
    pts = pd.read_csv(os.path.join(mimic_dir, "hosp", "patients.csv.gz"),
                      usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"])
    adm = pd.read_csv(os.path.join(mimic_dir, "hosp", "admissions.csv.gz"),
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                               "race", "insurance"])
    icu = pd.read_csv(os.path.join(mimic_dir, "icu", "icustays.csv.gz"),
                      usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"])

    # Parse dates
    for col in ["admittime", "dischtime", "deathtime"]:
        adm[col] = pd.to_datetime(adm[col])
    for col in ["intime", "outtime"]:
        icu[col] = pd.to_datetime(icu[col])

    # First ICU stay per patient
    icu = icu.sort_values(["subject_id", "intime"]).drop_duplicates("subject_id", keep="first")

    # Merge
    df = icu.merge(adm, on=["subject_id", "hadm_id"], how="inner")
    df = df.merge(pts, on="subject_id", how="inner")

    # Adult + ≥48h
    df = df[df["anchor_age"] >= 18]
    df = df[df["los"] >= 2.0]  # los is in fractional days

    # Mortality label: died during THIS hospitalization
    df["mortality"] = (~df["deathtime"].isna()).astype(int)

    # Demographics for subgroup analysis
    df["age"] = df["anchor_age"]
    df["age_band"] = pd.cut(df["age"], bins=[17, 44, 64, 79, 200],
                            labels=["18-44", "45-64", "65-79", "80+"])
    df["sex"] = df["gender"].map({"F": "F", "M": "M"})

    # Simplify ethnicity
    eth_map = {}
    for e in df["race"].unique():
        eu = str(e).upper()
        if "WHITE" in eu:
            eth_map[e] = "White"
        elif "BLACK" in eu or "AFRICAN" in eu:
            eth_map[e] = "Black"
        elif "HISPANIC" in eu or "LATINO" in eu:
            eth_map[e] = "Hispanic"
        elif "ASIAN" in eu:
            eth_map[e] = "Asian"
        else:
            eth_map[e] = "Other"
    df["ethnicity_group"] = df["race"].map(eth_map)

    # Simplify insurance
    ins_map = {}
    for i in df["insurance"].unique():
        iu = str(i).upper()
        if "MEDICARE" in iu:
            ins_map[i] = "Medicare"
        elif "MEDICAID" in iu:
            ins_map[i] = "Medicaid"
        else:
            ins_map[i] = "Other"
    df["insurance_group"] = df["insurance"].map(ins_map)

    # 48h window end
    df["window_end"] = df["intime"] + pd.Timedelta(hours=48)

    cohort = df[["subject_id", "hadm_id", "stay_id", "intime", "window_end",
                 "age", "age_band", "sex", "ethnicity_group", "insurance_group", "mortality"]].copy()
    print(f"Cohort: {len(cohort)} stays, mortality rate: {cohort['mortality'].mean():.3f}")
    return cohort


def extract_vitals(mimic_dir, cohort):
    """Extract vital signs from chartevents (chunked — file is huge)."""
    print("Extracting vitals from chartevents (this takes a while)...")
    stay_windows = cohort.set_index("stay_id")[["intime", "window_end"]].to_dict("index")
    valid_items = set(VITAL_ITEMS.keys())
    valid_stays = set(cohort["stay_id"].values)

    records = []
    chart_path = os.path.join(mimic_dir, "icu", "chartevents.csv.gz")
    for chunk in pd.read_csv(chart_path, chunksize=2_000_000,
                             usecols=["stay_id", "itemid", "charttime", "valuenum"]):
        chunk = chunk[chunk["stay_id"].isin(valid_stays) & chunk["itemid"].isin(valid_items)]
        chunk = chunk.dropna(subset=["valuenum"])
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])

        # Filter to 48h window
        keep = []
        for _, row in chunk.iterrows():
            w = stay_windows.get(row["stay_id"])
            if w and w["intime"] <= row["charttime"] <= w["window_end"]:
                keep.append(row)
        if keep:
            records.append(pd.DataFrame(keep))
        print(f"  processed chunk, {sum(len(r) for r in records)} records so far")

    if not records:
        return pd.DataFrame()
    vitals = pd.concat(records, ignore_index=True)
    vitals["variable"] = vitals["itemid"].map(VITAL_ITEMS)

    # Convert Fahrenheit to Celsius
    mask_f = vitals["variable"] == "temp_f"
    vitals.loc[mask_f, "valuenum"] = (vitals.loc[mask_f, "valuenum"] - 32) * 5 / 9
    vitals.loc[mask_f, "variable"] = "temp"
    vitals.loc[vitals["variable"] == "temp_c", "variable"] = "temp"

    # Combine BP sources (prefer arterial, fallback to non-invasive)
    vitals.loc[vitals["variable"] == "sbp_ni", "variable"] = "sbp"
    vitals.loc[vitals["variable"] == "dbp_ni", "variable"] = "dbp"
    vitals.loc[vitals["variable"] == "map_ni", "variable"] = "map"

    print(f"Vitals: {len(vitals)} measurements across {vitals['stay_id'].nunique()} stays")
    return vitals[["stay_id", "variable", "valuenum", "charttime"]]


def extract_labs(mimic_dir, cohort):
    """Extract labs from labevents."""
    print("Extracting labs from labevents...")
    valid_items = set(LAB_ITEMS.keys())
    hadm_windows = cohort.set_index("hadm_id")[["intime", "window_end"]].to_dict("index")
    valid_hadms = set(cohort["hadm_id"].values)

    records = []
    lab_path = os.path.join(mimic_dir, "hosp", "labevents.csv.gz")
    for chunk in pd.read_csv(lab_path, chunksize=2_000_000,
                             usecols=["hadm_id", "itemid", "charttime", "valuenum"]):
        chunk = chunk[chunk["hadm_id"].isin(valid_hadms) & chunk["itemid"].isin(valid_items)]
        chunk = chunk.dropna(subset=["valuenum"])
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])

        keep = []
        for _, row in chunk.iterrows():
            w = hadm_windows.get(row["hadm_id"])
            if w and w["intime"] <= row["charttime"] <= w["window_end"]:
                keep.append(row)
        if keep:
            records.append(pd.DataFrame(keep))
        print(f"  processed chunk, {sum(len(r) for r in records)} records so far")

    if not records:
        return pd.DataFrame()
    labs = pd.concat(records, ignore_index=True)
    labs["variable"] = labs["itemid"].map(LAB_ITEMS)

    # Map hadm_id → stay_id
    h2s = cohort.set_index("hadm_id")["stay_id"].to_dict()
    labs["stay_id"] = labs["hadm_id"].map(h2s)
    labs = labs.dropna(subset=["stay_id"])
    labs["stay_id"] = labs["stay_id"].astype(int)

    # Combine WBC sources
    labs.loc[labs["variable"] == "wbc_auto", "variable"] = "wbc"

    print(f"Labs: {len(labs)} measurements across {labs['stay_id'].nunique()} stays")
    return labs[["stay_id", "variable", "valuenum", "charttime"]]


def aggregate_features(measurements, cohort):
    """Aggregate to tabular: (stay_id) × (variable × {min,max,mean,last})."""
    print("Aggregating features...")
    # Sort by time for 'last' aggregation
    measurements = measurements.sort_values(["stay_id", "variable", "charttime"])

    agg = measurements.groupby(["stay_id", "variable"])["valuenum"].agg(
        fmin="min", fmax="max", fmean="mean", flast="last"
    ).reset_index()

    # Pivot to wide format
    rows = []
    for stay_id, grp in agg.groupby("stay_id"):
        row = {"stay_id": stay_id}
        for _, r in grp.iterrows():
            v = r["variable"]
            row[f"f_{v}_min"] = r["fmin"]
            row[f"f_{v}_max"] = r["fmax"]
            row[f"f_{v}_mean"] = r["fmean"]
            row[f"f_{v}_last"] = r["flast"]
        rows.append(row)

    features = pd.DataFrame(rows)

    # Merge with cohort demographics + label
    df = cohort.merge(features, on="stay_id", how="inner")

    # Fill missing features with median (standard practice, state in paper)
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())

    print(f"Final dataset: {len(df)} stays × {len(feat_cols)} features")
    print(f"Mortality rate: {df['mortality'].mean():.3f}")
    print(f"Feature columns: {len(feat_cols)}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_dir", required=True, help="Path to mimic-iv root (contains hosp/ and icu/)")
    parser.add_argument("--output", default="data/mimic_cohort.parquet")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    cohort = load_cohort(args.mimic_dir)
    vitals = extract_vitals(args.mimic_dir, cohort)
    labs = extract_labs(args.mimic_dir, cohort)

    measurements = pd.concat([vitals, labs], ignore_index=True)
    df = aggregate_features(measurements, cohort)

    df.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(f"\nSummary:")
    print(f"  Stays: {len(df)}")
    print(f"  Features: {len([c for c in df.columns if c.startswith('f_')])}")
    print(f"  Mortality: {df['mortality'].sum()} ({df['mortality'].mean()*100:.1f}%)")
    print(f"  Age bands: {df['age_band'].value_counts().to_dict()}")
    print(f"  Sex: {df['sex'].value_counts().to_dict()}")
    print(f"  Ethnicity: {df['ethnicity_group'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
