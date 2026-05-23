"""
Few-shot ExampleData instances for LangExtract medical pre-extraction.

Each example uses synthetic (de-identified) clinical text. extraction_text
must be a verbatim substring of the example text for char-grounding to work.
"""

from langextract.core.data import ExampleData, Extraction

# ── Lab Report Examples ───────────────────────────────────────────────────────

LAB_EXAMPLES = [
    ExampleData(
        text=(
            "CBC WITH DIFFERENTIAL\n"
            "WBC: 14.2 K/uL [H] (ref 4.5-11.0)\n"
            "RBC: 3.98 M/uL [L] (ref 4.20-5.80)\n"
            "Hemoglobin: 8.4 g/dL [L] (ref 13.5-17.5)\n"
            "Hematocrit: 26.1 % [L] (ref 40.0-54.0)\n"
            "Platelets: 412 K/uL [H] (ref 150-400)\n"
            "Specimen: Venous blood"
        ),
        extractions=[
            Extraction(
                extraction_class="lab_result",
                extraction_text="WBC: 14.2 K/uL [H] (ref 4.5-11.0)",
                attributes={
                    "lab_test_name": "WBC",
                    "value": "14.2",
                    "unit": "K/uL",
                    "reference_range": "4.5-11.0",
                    "flag": "H",
                    "specimen_type": "Venous blood",
                },
            ),
            Extraction(
                extraction_class="lab_result",
                extraction_text="Hemoglobin: 8.4 g/dL [L] (ref 13.5-17.5)",
                attributes={
                    "lab_test_name": "Hemoglobin",
                    "value": "8.4",
                    "unit": "g/dL",
                    "reference_range": "13.5-17.5",
                    "flag": "L",
                    "specimen_type": "Venous blood",
                },
            ),
            Extraction(
                extraction_class="lab_result",
                extraction_text="Platelets: 412 K/uL [H] (ref 150-400)",
                attributes={
                    "lab_test_name": "Platelets",
                    "value": "412",
                    "unit": "K/uL",
                    "reference_range": "150-400",
                    "flag": "H",
                    "specimen_type": "Venous blood",
                },
            ),
        ],
    ),
    ExampleData(
        text=(
            "METABOLIC PANEL\n"
            "Sodium: 138 mEq/L (ref 136-145)\n"
            "Potassium: 3.2 mEq/L [L] (ref 3.5-5.1)\n"
            "Creatinine: 2.8 mg/dL [H] (ref 0.7-1.3)\n"
            "eGFR: 22 mL/min/1.73m2 [Critical Low]\n"
            "Glucose: 312 mg/dL [Critical High] (ref 70-99)\n"
            "Specimen: Serum"
        ),
        extractions=[
            Extraction(
                extraction_class="lab_result",
                extraction_text="Creatinine: 2.8 mg/dL [H] (ref 0.7-1.3)",
                attributes={
                    "lab_test_name": "Creatinine",
                    "value": "2.8",
                    "unit": "mg/dL",
                    "reference_range": "0.7-1.3",
                    "flag": "H",
                    "specimen_type": "Serum",
                },
            ),
            Extraction(
                extraction_class="lab_result",
                extraction_text="eGFR: 22 mL/min/1.73m2 [Critical Low]",
                attributes={
                    "lab_test_name": "eGFR",
                    "value": "22",
                    "unit": "mL/min/1.73m2",
                    "reference_range": "",
                    "flag": "Critical",
                    "specimen_type": "Serum",
                },
            ),
            Extraction(
                extraction_class="lab_result",
                extraction_text="Glucose: 312 mg/dL [Critical High] (ref 70-99)",
                attributes={
                    "lab_test_name": "Glucose",
                    "value": "312",
                    "unit": "mg/dL",
                    "reference_range": "70-99",
                    "flag": "Critical",
                    "specimen_type": "Serum",
                },
            ),
        ],
    ),
]

# ── Radiology Report Examples ─────────────────────────────────────────────────

RADIOLOGY_EXAMPLES = [
    ExampleData(
        text=(
            "CHEST X-RAY PA AND LATERAL\n"
            "FINDINGS: The cardiac silhouette is mildly enlarged. "
            "There is a 2.3 cm opacity in the right lower lobe consistent "
            "with consolidation. No pleural effusion is identified on the left. "
            "Mild bilateral interstitial markings are present.\n"
            "IMPRESSION: 1. Right lower lobe consolidation, possibly pneumonia. "
            "2. Cardiomegaly, mild. Clinical correlation recommended."
        ),
        extractions=[
            Extraction(
                extraction_class="radiology_finding",
                extraction_text="2.3 cm opacity in the right lower lobe consistent with consolidation",
                attributes={
                    "finding": "opacity consistent with consolidation",
                    "anatomic_location": "lower lobe",
                    "laterality": "right",
                    "severity": "moderate",
                    "impression_line": "Right lower lobe consolidation, possibly pneumonia.",
                },
            ),
            Extraction(
                extraction_class="radiology_finding",
                extraction_text="cardiac silhouette is mildly enlarged",
                attributes={
                    "finding": "cardiomegaly",
                    "anatomic_location": "cardiac silhouette",
                    "laterality": "bilateral",
                    "severity": "mild",
                    "impression_line": "Cardiomegaly, mild.",
                },
            ),
        ],
    ),
    ExampleData(
        text=(
            "MRI BRAIN WITHOUT CONTRAST\n"
            "FINDINGS: There is a 1.8 cm T2 hyperintense lesion in the left "
            "frontal white matter with surrounding vasogenic edema. "
            "No midline shift. The posterior fossa structures are intact. "
            "No acute infarct identified.\n"
            "IMPRESSION: Left frontal white matter lesion with edema — "
            "differential includes high-grade glioma vs metastasis."
        ),
        extractions=[
            Extraction(
                extraction_class="radiology_finding",
                extraction_text="1.8 cm T2 hyperintense lesion in the left frontal white matter with surrounding vasogenic edema",
                attributes={
                    "finding": "T2 hyperintense lesion with vasogenic edema",
                    "anatomic_location": "frontal white matter",
                    "laterality": "left",
                    "severity": "significant",
                    "impression_line": "Left frontal white matter lesion with edema — differential includes high-grade glioma vs metastasis.",
                },
            ),
        ],
    ),
]

# ── Discharge Summary Examples ────────────────────────────────────────────────

DISCHARGE_EXAMPLES = [
    ExampleData(
        text=(
            "DISCHARGE SUMMARY\n"
            "DIAGNOSIS: Type 2 Diabetes Mellitus, uncontrolled. "
            "Hypertensive urgency.\n"
            "PROCEDURES: IV insulin drip, continuous cardiac monitoring.\n"
            "MEDICATIONS ON DISCHARGE:\n"
            "Metformin 1000 mg oral twice daily with meals for diabetes management.\n"
            "Lisinopril 10 mg oral once daily for hypertension.\n"
            "Aspirin 81 mg oral once daily for cardiovascular prophylaxis.\n"
            "FOLLOW-UP: Primary care in 1 week."
        ),
        extractions=[
            Extraction(
                extraction_class="medication",
                extraction_text="Metformin 1000 mg oral twice daily with meals for diabetes management",
                attributes={
                    "medication_name": "Metformin",
                    "dosage": "1000 mg",
                    "route": "oral",
                    "frequency": "twice daily",
                    "duration": "",
                    "indication": "diabetes management",
                    "diagnosis": "Type 2 Diabetes Mellitus",
                    "procedure": "",
                },
            ),
            Extraction(
                extraction_class="medication",
                extraction_text="Lisinopril 10 mg oral once daily for hypertension",
                attributes={
                    "medication_name": "Lisinopril",
                    "dosage": "10 mg",
                    "route": "oral",
                    "frequency": "once daily",
                    "duration": "",
                    "indication": "hypertension",
                    "diagnosis": "Hypertensive urgency",
                    "procedure": "",
                },
            ),
        ],
    ),
    ExampleData(
        text=(
            "ADMISSION DIAGNOSIS: Community-acquired pneumonia, severe.\n"
            "HOSPITAL COURSE: Patient treated with IV antibiotics.\n"
            "DISCHARGE MEDICATIONS:\n"
            "Amoxicillin-clavulanate 875/125 mg oral twice daily for 7 days "
            "to complete pneumonia treatment.\n"
            "Azithromycin 250 mg oral once daily for 5 days for atypical coverage.\n"
            "Prednisone 40 mg oral once daily for 5 days for inflammation.\n"
            "PROCEDURES: Chest physiotherapy, incentive spirometry."
        ),
        extractions=[
            Extraction(
                extraction_class="medication",
                extraction_text="Amoxicillin-clavulanate 875/125 mg oral twice daily for 7 days to complete pneumonia treatment",
                attributes={
                    "medication_name": "Amoxicillin-clavulanate",
                    "dosage": "875/125 mg",
                    "route": "oral",
                    "frequency": "twice daily",
                    "duration": "7 days",
                    "indication": "pneumonia treatment",
                    "diagnosis": "Community-acquired pneumonia",
                    "procedure": "",
                },
            ),
            Extraction(
                extraction_class="medication",
                extraction_text="Prednisone 40 mg oral once daily for 5 days for inflammation",
                attributes={
                    "medication_name": "Prednisone",
                    "dosage": "40 mg",
                    "route": "oral",
                    "frequency": "once daily",
                    "duration": "5 days",
                    "indication": "inflammation",
                    "diagnosis": "Community-acquired pneumonia",
                    "procedure": "Chest physiotherapy, incentive spirometry",
                },
            ),
        ],
    ),
]
