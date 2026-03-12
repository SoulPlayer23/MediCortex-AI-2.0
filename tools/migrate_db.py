"""
Patient Database Seeder — Synthea-Powered (Multi-Source)
=========================================================
Downloads all three publicly available Synthea CSV datasets (Apache 2.0,
no credentials required), merges them, and upserts every patient into the
`patients` table — giving maximum demographic and clinical diversity.

Datasets downloaded
-------------------
  LATEST  — ~117  patients  (latest master snapshot)
  APR2020 — ~1171 patients  (April 2020 snapshot)
  NOV2021 — ~1163 patients  (November 2021 snapshot, richest clinical data)
  COVID19 — ~12352 patients (COVID-19 simulation module)
  ─────────────────────────────────────────────
  TOTAL   ≈ 14 800 unique patients

Synthea CSV files used per dataset
-----------------------------------
  patients.csv    — demographics (DOB, sex, race, ethnicity, address …)
  conditions.csv  — SNOMED-coded diagnoses
  medications.csv — RxNorm-coded drugs
  allergies.csv   — allergens
  observations.csv— LOINC vitals (category = vital-signs)

Run
---
    python -m tools.migrate_db          # seed all datasets (default)
    python -m tools.migrate_db --count 500   # limit to first N patients

Data source
-----------
    https://github.com/synthetichealth/synthea-sample-data
    License: Apache 2.0 (generated synthetic data, not real PHI)
"""

import argparse
import asyncio
import csv
import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import asyncpg
import httpx
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/medicortex",
).replace("postgresql+asyncpg://", "postgresql://")

_BASE = "https://raw.githubusercontent.com/synthetichealth/synthea-sample-data/main/downloads"
SYNTHEA_SOURCES = {
    "APR2020": f"{_BASE}/synthea_sample_data_csv_apr2020.zip",
    "NOV2021": f"{_BASE}/synthea_sample_data_csv_nov2021.zip",
    "COVID19": f"{_BASE}/10k_synthea_covid19_csv.zip",
}

# LOINC codes we care about for vitals
VITALS_LOINC = {
    "55284-4": "blood_pressure",      # Blood pressure panel (systolic+diastolic)
    "8480-6":  "systolic_bp",
    "8462-4":  "diastolic_bp",
    "8867-4":  "heart_rate",
    "8310-5":  "temperature",
    "9279-1":  "respiratory_rate",
    "2708-6":  "spo2",
    "29463-7": "weight",
    "39156-5": "bmi",
    "8302-2":  "height",
}


# ── Download + extract ────────────────────────────────────────────────────────

def _download_synthea_zip(label: str, url: str) -> bytes:
    print(f"📥 [{label}] {url}")
    with httpx.Client(follow_redirects=True, timeout=180) as client:
        resp = client.get(url)
        resp.raise_for_status()
    print(f"   {len(resp.content)/1_048_576:.1f} MB")
    return resp.content


def _extract_csv(zip_bytes: bytes, filename: str) -> List[Dict[str, str]]:
    """Return rows of a CSV file from within the zip as a list of dicts."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # The zip may nest files inside a subdirectory
        candidates = [n for n in zf.namelist() if n.endswith(f"/{filename}") or n == filename]
        if not candidates:
            raise FileNotFoundError(f"{filename} not found in Synthea zip")
        with zf.open(candidates[0]) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            return list(reader)


# ── Builders ──────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _age_from_dob(dob_str: str) -> Optional[int]:
    dob = _parse_date(dob_str)
    if not dob:
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _build_diagnoses(conditions: List[Dict]) -> List[Dict[str, Any]]:
    seen: set = set()
    out = []
    for row in conditions:
        code = row.get("CODE", "")
        desc = row.get("DESCRIPTION", "")
        key = (code, desc)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "condition": desc,
            "icd10":     code,
            "diagnosed": row.get("START", ""),
            "status":    "Active" if not row.get("STOP") else "Resolved",
        })
    return out


def _build_medications(medications: List[Dict]) -> List[Dict[str, Any]]:
    seen: set = set()
    out = []
    for row in medications:
        name = row.get("DESCRIPTION", "")
        code = row.get("CODE", "")
        key = (name, code)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "name":      name,
            "dosage":    "",          # Synthea CSV doesn't include dosage text
            "frequency": "",
            "rxnorm":    code,
            "start":     row.get("START", ""),
            "active":    not bool(row.get("STOP")),
        })
    return out


def _build_allergies(allergies: List[Dict]) -> List[str]:
    seen: set = set()
    out = []
    for row in allergies:
        desc = row.get("DESCRIPTION", "")
        if desc and desc not in seen:
            seen.add(desc)
            out.append(desc)
    return out


def _build_vitals_history(obs_by_encounter: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """
    Group observations by encounter date → one vitals snapshot per encounter.
    Returns list sorted by date descending (most recent first).
    """
    by_date: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for rows in obs_by_encounter.values():
        for row in rows:
            loinc = row.get("CODE", "")
            field = VITALS_LOINC.get(loinc)
            if not field:
                continue
            obs_date = row.get("DATE", "")[:10]   # keep YYYY-MM-DD only
            value    = row.get("VALUE", "")
            units    = row.get("UNITS", "")

            snapshot = by_date[obs_date]
            snapshot["date"] = obs_date

            display = f"{value} {units}".strip() if units else value

            if field == "systolic_bp":
                snapshot["_sys"] = value
            elif field == "diastolic_bp":
                snapshot["_dia"] = value
            elif field == "heart_rate":
                snapshot["heart_rate"] = f"{value} bpm"
            elif field == "temperature":
                snapshot["temperature"] = f"{value}°C"
            elif field == "respiratory_rate":
                snapshot["respiratory_rate"] = f"{value}/min"
            elif field == "spo2":
                snapshot["spo2"] = f"{value}%"
            elif field == "weight":
                snapshot["weight"] = f"{value} kg"
            elif field == "bmi":
                snapshot["bmi"] = value
            elif field == "height":
                snapshot["height"] = f"{value} cm"

    # Compose blood_pressure from sys/dia
    result = []
    for snap in by_date.values():
        sys_v = snap.pop("_sys", None)
        dia_v = snap.pop("_dia", None)
        if sys_v and dia_v:
            snap["blood_pressure"] = f"{sys_v}/{dia_v} mmHg"
        if len(snap) > 1:       # more than just the date key
            result.append(snap)

    result.sort(key=lambda s: s.get("date", ""), reverse=True)
    return result[:6]           # keep at most 6 snapshots per patient


def _last_visit(vitals: List[Dict]) -> Optional[date]:
    if not vitals:
        return None
    return _parse_date(vitals[0].get("date", ""))


# ── Main transform ────────────────────────────────────────────────────────────

def build_patient_records(
    patients_csv: List[Dict],
    conditions_map: Dict[str, List[Dict]],
    meds_map:       Dict[str, List[Dict]],
    allergies_map:  Dict[str, List[Dict]],
    obs_map:        Dict[str, List[Dict]],
    limit: Optional[int],
    id_prefix: str = "SYN",
) -> List[Dict[str, Any]]:
    """
    Join CSVs on patient UUID and return a list of patient dicts
    ready for DB insertion.  id_prefix differentiates datasets so
    patient_ids are unique across the merged table.
    """
    # Include all patients — hospitals keep records for deceased patients too.
    # Deceased flag is preserved in the address JSONB for reference.
    subset = patients_csv[:limit] if limit else patients_csv

    records = []
    for i, row in enumerate(subset, start=1):
        pid = row["Id"]
        dob  = row.get("BIRTHDATE", "")
        sex  = "Male" if row.get("GENDER") == "M" else "Female"
        vitals = _build_vitals_history(
            defaultdict(list, {pid: obs_map.get(pid, [])})
        )

        records.append({
            "patient_id":    f"PT-{id_prefix}-{i:05d}",
            "synthea_id":    pid,
            "full_name":     f"{row.get('FIRST', '')} {row.get('LAST', '')}".strip(),
            "date_of_birth": _parse_date(dob),
            "age":           _age_from_dob(dob),
            "sex":           sex,
            "blood_type":    None,           # not in Synthea CSV
            "race":          row.get("RACE", ""),
            "ethnicity":     row.get("ETHNICITY", ""),
            "marital_status":row.get("MARITAL", ""),
            "address": {
                "city":        row.get("CITY", ""),
                "state":       row.get("STATE", ""),
                "zip":         row.get("ZIP", ""),
                "deceased":    bool(row.get("DEATHDATE")),
                "death_date":  row.get("DEATHDATE") or None,
            },
            "diagnoses":    _build_diagnoses(conditions_map.get(pid, [])),
            "medications":  _build_medications(meds_map.get(pid, [])),
            "allergies":    _build_allergies(allergies_map.get(pid, [])),
            "vitals_history": vitals,
            "last_visit":   _last_visit(vitals),
        })

    return records


# ── DB seeder ─────────────────────────────────────────────────────────────────

UPSERT_SQL = """
    INSERT INTO patients (
        patient_id, full_name, date_of_birth, age, sex, blood_type,
        race, ethnicity, marital_status, address,
        diagnoses, medications, allergies, vitals_history, last_visit
    ) VALUES (
        $1, $2, $3, $4, $5, $6,
        $7, $8, $9, $10::jsonb,
        $11::jsonb, $12::jsonb, $13::jsonb, $14::jsonb, $15
    )
    ON CONFLICT (patient_id) DO UPDATE SET
        full_name      = EXCLUDED.full_name,
        date_of_birth  = EXCLUDED.date_of_birth,
        age            = EXCLUDED.age,
        sex            = EXCLUDED.sex,
        blood_type     = EXCLUDED.blood_type,
        race           = EXCLUDED.race,
        ethnicity      = EXCLUDED.ethnicity,
        marital_status = EXCLUDED.marital_status,
        address        = EXCLUDED.address,
        diagnoses      = EXCLUDED.diagnoses,
        medications    = EXCLUDED.medications,
        allergies      = EXCLUDED.allergies,
        vitals_history = EXCLUDED.vitals_history,
        last_visit     = EXCLUDED.last_visit,
        updated_at     = NOW()
"""


async def seed_patients(conn: asyncpg.Connection, records: List[Dict]) -> None:
    for p in records:
        await conn.execute(
            UPSERT_SQL,
            p["patient_id"],
            p["full_name"],
            p["date_of_birth"],
            p["age"],
            p["sex"],
            p.get("blood_type"),
            p.get("race"),
            p.get("ethnicity"),
            p.get("marital_status"),
            json.dumps(p.get("address", {})),
            json.dumps(p["diagnoses"]),
            json.dumps(p["medications"]),
            json.dumps(p["allergies"]),
            json.dumps(p["vitals_history"]),
            p.get("last_visit"),
        )
        dx_count  = len(p["diagnoses"])
        med_count = len(p["medications"])
        print(f"  ✔  {p['patient_id']}  {p['full_name']:<28}  "
              f"({dx_count} dx, {med_count} meds)")


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_zip(zip_bytes: bytes, label: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Download, parse and transform one Synthea zip into patient records."""
    print(f"📂 [{label}] Parsing CSVs …")
    patients_csv   = _extract_csv(zip_bytes, "patients.csv")
    conditions_csv = _extract_csv(zip_bytes, "conditions.csv")
    meds_csv       = _extract_csv(zip_bytes, "medications.csv")
    allergies_csv  = _extract_csv(zip_bytes, "allergies.csv")
    obs_csv        = _extract_csv(zip_bytes, "observations.csv")

    print(f"   patients:{len(patients_csv):>6}  conditions:{len(conditions_csv):>7}"
          f"  meds:{len(meds_csv):>7}  allergies:{len(allergies_csv):>5}"
          f"  observations:{len(obs_csv):>8}")

    conditions_map: Dict[str, List] = defaultdict(list)
    for row in conditions_csv:
        conditions_map[row["PATIENT"]].append(row)

    meds_map: Dict[str, List] = defaultdict(list)
    for row in meds_csv:
        meds_map[row["PATIENT"]].append(row)

    allergies_map: Dict[str, List] = defaultdict(list)
    for row in allergies_csv:
        allergies_map[row["PATIENT"]].append(row)

    obs_map: Dict[str, List] = defaultdict(list)
    for row in obs_csv:
        # Filter by CATEGORY when present; fall back to LOINC whitelist only
        # (APR2020 and COVID19 datasets omit the CATEGORY column entirely)
        category = row.get("CATEGORY", "")
        if not category or category == "vital-signs":
            obs_map[row["PATIENT"]].append(row)

    return build_patient_records(
        patients_csv, conditions_map, meds_map, allergies_map, obs_map,
        limit, id_prefix=label,
    )


async def migrate(count: Optional[int]) -> None:
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        grand_total = 0
        for label, url in SYNTHEA_SOURCES.items():
            zip_bytes = _download_synthea_zip(label, url)
            records   = _parse_zip(zip_bytes, label, count)
            print(f"\n🌱 [{label}] Seeding {len(records)} patients …\n")
            await seed_patients(conn, records)
            subtotal = await conn.fetchval(
                "SELECT COUNT(*) FROM patients WHERE patient_id LIKE $1",
                f"PT-{label}-%"
            )
            grand_total = await conn.fetchval("SELECT COUNT(*) FROM patients")
            print(f"\n   [{label}] {subtotal} rows in table  |  running total: {grand_total}")

        print(f"\n✅ All done — {grand_total} patient(s) total in table.")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed patients table from Synthea CSV data")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--count", type=int,
                       help="Limit seeding to N patients")
    group.add_argument("--all",   action="store_true", default=True,
                       help="Seed all available patients (default)")
    args = parser.parse_args()
    limit = args.count  # None when --all (default)
    asyncio.run(migrate(limit))


if __name__ == "__main__":
    main()
