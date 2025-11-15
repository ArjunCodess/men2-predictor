import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

STUDY_NAME_MAP = {
    "study_1": "JCEM Case Reports (2025)",
    "study_2": "JCEM (2016) RET Exon 7 Deletion",
    "study_3": "Thyroid Journal (2016)",
    "study_4": "European Journal of Endocrinology (2006)",
    "study_5": "Laryngoscope (2021) MEN2A Penetrance",
    "study_6": "JCEM (2018) Homozygous RET K666N",
    "study_7": "Oncotarget (2015) RET S891A FMTC",
    "study_8": "AJCR (2022) Calcitonin-Negative MTC",
    "study_9": "JCEM (2022) ctDNA Cohort",
    "study_10": "Genes (2022) RET c.1901G>A Family",
    "study_11": "BMC Pediatrics (2020) MEN2B Case",
    "study_12": "Annales d'Endocrinologie (2015) RET Y791F",
    "study_13": "Surgery Today (2014) RET S891A Pheo",
    "study_14": "Annals of Surgery (2014) MEN2B Cohort",
    "ret_k666n_homozygous_2018": "JCEM (2018) Homozygous RET K666N",
    "ret_s891a_fmtc_ca_2015": "Oncotarget (2015) RET S891A FMTC"
}

AMINO_ACID_MAP = {
    "ala": "A",
    "arg": "R",
    "asn": "N",
    "asp": "D",
    "cys": "C",
    "gln": "Q",
    "glu": "E",
    "gly": "G",
    "his": "H",
    "ile": "I",
    "leu": "L",
    "lys": "K",
    "met": "M",
    "phe": "F",
    "pro": "P",
    "ser": "S",
    "thr": "T",
    "trp": "W",
    "tyr": "Y",
    "val": "V"
}


def amino_acid_to_one(code):
    """convert 3-letter amino acid shorthand into 1-letter"""
    if not code:
        return ""
    cleaned = str(code).strip().lower()
    if len(cleaned) == 1:
        return cleaned.upper()
    return AMINO_ACID_MAP.get(cleaned, cleaned[0].upper())


def normalize_variant_code(variant_text):
    """normalize variant notation such as p.Cys634Tyr into C634Y"""
    if not variant_text:
        return None
    text = str(variant_text).strip()
    text = text.replace("p.", "")
    text = text.replace("P.", "")
    del_match = re.match(r"([A-Za-z]{1,3})(\d+)_([A-Za-z]{1,3})(\d+)(del.*)", text, re.IGNORECASE)
    if del_match:
        start, pos1, end, pos2, suffix = del_match.groups()
        suffix = suffix.lower()
        return f"{amino_acid_to_one(start)}{pos1}_{amino_acid_to_one(end)}{pos2}{suffix}"
    match = re.match(r"([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})", text)
    if match:
        start, pos, end = match.groups()
        return f"{amino_acid_to_one(start)}{pos}{amino_acid_to_one(end)}"
    return text.upper().replace(" ", "")


def parse_ret_variant_string(raw_value):
    """extract a normalized RET variant from free text"""
    if not raw_value:
        return None
    text = str(raw_value).strip()
    # remove gene label if present
    if " " in text and text.upper().startswith("RET"):
        text = text.split(" ", 1)[1]
    complex_match = re.search(r"([A-Z]\d+_[A-Z]\d+del(?:ins[A-Z]+)?)", text)
    if complex_match:
        return complex_match.group(1)
    simple_match = re.search(r"([A-Z]\d+[A-Z])", text)
    if simple_match:
        return simple_match.group(1)
    return normalize_variant_code(text)


def parse_age_range(range_text):
    """parse an age range string like '25-77' into numeric bounds"""
    if not range_text or "-" not in str(range_text):
        return None
    parts = str(range_text).replace(" ", "").split("-")
    try:
        low = float(parts[0])
        high = float(parts[1])
        if np.isnan(low) or np.isnan(high):
            return None
        return low, high
    except (ValueError, TypeError):
        return None


def generate_age_sequence(count, age_stats):
    """generate deterministic ages based on available study summary statistics"""
    if count <= 0:
        return []
    age_stats = age_stats or {}
    age_range = parse_age_range(age_stats.get("range"))
    mean_age = age_stats.get("mean")
    if age_range and age_range[0] != age_range[1]:
        low, high = age_range
        values = np.linspace(low, high, count)
    elif mean_age is not None:
        values = np.full(count, float(mean_age))
    else:
        values = np.full(count, 45.0)
    return [float(val) for val in values]


def generate_gender_sequence(count, sex_distribution):
    """assign genders following the reported male/female distribution"""
    if count <= 0:
        return []
    sex_distribution = sex_distribution or {}
    total = sex_distribution.get("male", 0) + sex_distribution.get("female", 0)
    if total <= 0:
        return ["Female"] * count
    male_ratio = sex_distribution.get("male", 0) / total
    target_males = int(round(male_ratio * count))
    target_males = min(max(target_males, 0), count)
    male_remaining = target_males
    female_remaining = count - target_males
    genders = []
    for idx in range(count):
        remaining = count - idx
        male_share = male_remaining / remaining if remaining else 0
        female_share = female_remaining / remaining if remaining else 0
        if male_share >= female_share and male_remaining > 0:
            genders.append("Male")
            male_remaining -= 1
        elif female_remaining > 0:
            genders.append("Female")
            female_remaining -= 1
        else:
            genders.append("Female")
    return genders


def convert_biochemical_entries(entries, test_key="test", value_key="value"):
    """normalize heterogeneous biochemical entry formats"""
    converted = []
    if not entries:
        return converted
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized = {
            "test": entry.get(test_key) or entry.get("analyte"),
            "value": entry.get(value_key) or entry.get("result") or entry.get("value"),
            "unit": entry.get("unit"),
            "reference_range": entry.get("reference_range") or entry.get("reference")
        }
        converted.append(normalized)
    return converted


def normalize_yes_no(value, default="No"):
    """standardize yes/no style flags"""
    if value is None:
        return default
    if isinstance(value, str):
        val = value.strip().lower()
        if not val:
            return default
        if val in {"yes", "y", "true", "1", "present"}:
            return "Yes"
        if val in {"no", "n", "false", "0", "absent"}:
            return "No"
        return value
    if isinstance(value, (int, float)):
        return "Yes" if value else "No"
    return value


def build_study2_patients(study):
    """convert RET exon 7 deletion case report"""
    variant = parse_ret_variant_string((study.get("variant_info") or {}).get("protein")) or "E505_G506del"
    patients = []
    for record in study.get("patient_data", []):
        mtc_info = record.get("mtc", {})
        mtc_biochem = mtc_info.get("biochemistry") or {}
        basal_calcitonin = mtc_biochem.get("basal_calcitonin", {})
        stimulated_calcitonin = mtc_biochem.get("pentagastrin_stimulated_calcitonin", {})
        bio_entries = []
        if basal_calcitonin:
            bio_entries.append({
                "test": "calcitonin",
                "value": basal_calcitonin.get("value"),
                "unit": basal_calcitonin.get("unit"),
                "reference_range": basal_calcitonin.get("normal")
            })
        if stimulated_calcitonin:
            bio_entries.append({
                "test": "stimulated_calcitonin",
                "value": stimulated_calcitonin.get("value"),
                "unit": stimulated_calcitonin.get("unit"),
                "reference_range": stimulated_calcitonin.get("normal")
            })
        patient = {
            "patient_id": f"{study.get('study_id', 'study_2')}_{record.get('patient_id', 'proband')}",
            "age": record.get("age_mtc") or record.get("age_pheo"),
            "gender": record.get("sex"),
            "relationship": "Proband",
            "ret_variant": variant,
            "men2_syndrome": "Yes",
            "mtc_diagnosis": "Yes",
            "pheochromocytoma": "Yes",
            "hyperparathyroidism": "No",
            "family_history_mtc": "No",
            "calcitonin_level": basal_calcitonin.get("value"),
            "calcitonin_normal_range": basal_calcitonin.get("normal"),
            "cea_level": None,
            "thyroid_ultrasound": (mtc_info.get("ultrasound") or {}).get("nodule_size"),
            "family_screening": "Yes" if study.get("family_screening") else "No"
        }
        if bio_entries:
            patient["biochemical_data"] = bio_entries
        patients.append(patient)
    return patients


def build_study14_patients(study):
    """convert Annals of Surgery 2014 MEN2B cohort"""
    patients = []

    def relationship_for(record):
        if record.get("relationship"):
            return record["relationship"]
        if str(record.get("diagnosis_group", "")).lower() == "inherited":
            return "Family member"
        return "Proband"

    for record in study.get("patient_data", []):
        patient = {
            "patient_id": f"{study.get('study_id', 'study_14')}_{record.get('patient_id', 'patient')}",
            "age": record.get("age") or record.get("age_at_diagnosis"),
            "gender": record.get("gender"),
            "relationship": relationship_for(record),
            "ret_variant": parse_ret_variant_string(record.get("ret_variant")) or "M918T",
            "men2_syndrome": normalize_yes_no(record.get("men2_syndrome"), default="Yes"),
            "mtc_diagnosis": normalize_yes_no(record.get("mtc_diagnosis"), default="No"),
            "pheochromocytoma": normalize_yes_no(record.get("pheochromocytoma"), default="No"),
            "hyperparathyroidism": normalize_yes_no(record.get("hyperparathyroidism"), default="No"),
            "family_history_mtc": normalize_yes_no(
                record.get("family_history_mtc"),
                default="Yes" if str(record.get("diagnosis_group", "")).lower() == "inherited" else "No"
            ),
            "calcitonin_level": record.get("calcitonin_level"),
            "calcitonin_normal_range": record.get("calcitonin_normal_range"),
            "thyroid_ultrasound": record.get("thyroid_ultrasound"),
            "family_screening": "Yes" if str(record.get("diagnosis_trigger", "")).lower() == "family screening" else "No",
            "c_cell_hyperplasia": record.get("c_cell_hyperplasia"),
            "men2b_diagnosis": record.get("men2b_diagnosis"),
            "diagnosis_trigger": record.get("diagnosis_trigger"),
            "diagnosis_group": record.get("diagnosis_group")
        }

        if patient["men2_syndrome"] not in {"Yes", "No"}:
            patient["men2_syndrome"] = normalize_yes_no(record.get("men2b_diagnosis"), default="Yes")

        if patient["mtc_diagnosis"] not in {"Yes", "No"} and record.get("mtc_stage"):
            patient["mtc_diagnosis"] = "Yes"

        bio_entries = []
        for key, label in [
            ("calcitonin_preoperative_basal", "calcitonin_preoperative_basal"),
            ("calcitonin_preoperative_stimulated", "calcitonin_preoperative_stimulated"),
            ("calcitonin_postoperative_basal", "calcitonin_postoperative_basal"),
            ("calcitonin_postoperative_stimulated", "calcitonin_postoperative_stimulated"),
        ]:
            value = record.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            bio_entries.append({
                "test": label,
                "value": numeric_value,
                "unit": "pg/mL",
                "reference_range": record.get("calcitonin_normal_range")
            })

        if bio_entries:
            patient["biochemical_data"] = bio_entries

        patients.append(patient)

    return patients


def prepare_study8_patients(study):
    """convert study 8 structure into standardized patient records"""
    patient_data = study.get("patient_data")
    if not isinstance(patient_data, dict):
        return []
    patient = patient_data.copy()
    patient.setdefault("patient_id", f"{study.get('study_id', 'study_8')}_proband")
    patient.setdefault("relationship", "Proband")
    variant_info = (study.get("variant_info") or {}).get("ret_variant") or {}
    patient["ret_variant"] = parse_ret_variant_string(variant_info.get("protein"))
    patient.setdefault("mtc_diagnosis", "Yes")
    patient.setdefault("men2_syndrome", "Yes")
    patient.setdefault("pheochromocytoma", "No")
    patient.setdefault("hyperparathyroidism", "No")
    patient.setdefault("family_history_mtc", "Yes")
    bio_entries = convert_biochemical_entries(patient.pop("biochemical_results", []))
    if bio_entries:
        patient["biochemical_data"] = bio_entries
        calcitonin_entry = next((entry for entry in bio_entries if entry.get("test") and "calcitonin" in entry["test"].lower()), None)
        cea_entry = next((entry for entry in bio_entries if entry.get("test") and "cea" in entry["test"].lower()), None)
        if calcitonin_entry:
            patient.setdefault("calcitonin_level", calcitonin_entry.get("value"))
            if calcitonin_entry.get("reference_range"):
                patient["calcitonin_normal_range"] = calcitonin_entry["reference_range"]
        if cea_entry:
            patient.setdefault("cea_level", cea_entry.get("value"))
    ultrasound = None
    for imaging in patient.get("imaging", []):
        if isinstance(imaging, dict) and imaging.get("modality", "").lower() == "ultrasound":
            ultrasound = imaging.get("finding")
            break
    if ultrasound:
        patient["thyroid_ultrasound"] = ultrasound
        nodule_count = ultrasound.lower().count("nodule")
        if nodule_count:
            patient["thyroid_nodule_count"] = max(nodule_count, patient.get("thyroid_nodule_count", 0))
    tumor_sizes = (patient.get("pathology") or {}).get("tumor_sizes_cm")
    if tumor_sizes and isinstance(tumor_sizes, list):
        patient["thyroid_nodule_count"] = max(len(tumor_sizes), patient.get("thyroid_nodule_count", 0))
    return [patient]


def build_study9_patients(study):
    """convert ctDNA cohort cases into patient-level rows"""
    cases = study.get("case_data", [])
    ret_cases = [case for case in cases if str(case.get("mutation", "")).upper().startswith("RET")]
    if not ret_cases:
        return []
    cohort_info = study.get("cohort_info", {})
    age_sequence = generate_age_sequence(len(ret_cases), cohort_info.get("age_stats", {}))
    gender_sequence = generate_gender_sequence(len(ret_cases), cohort_info.get("sex_distribution", {}))
    patients = []
    for idx, case in enumerate(ret_cases):
        patient = {
            "patient_id": f"{study.get('study_id', 'study_9')}_case_{case.get('case_id')}",
            "age": age_sequence[idx] if idx < len(age_sequence) else cohort_info.get("age_stats", {}).get("mean"),
            "gender": gender_sequence[idx] if idx < len(gender_sequence) else "Female",
            "relationship": "Sporadic case",
            "ret_variant": parse_ret_variant_string(case.get("mutation")),
            "mtc_diagnosis": "Yes",
            "men2_syndrome": "No",
            "pheochromocytoma": "No",
            "hyperparathyroidism": "No",
            "family_history_mtc": "No",
            "calcitonin_level": case.get("preop_ct"),
            "calcitonin_normal_range": "0-7.5 pg/mL",
            "cea_level": case.get("preop_cea"),
            "thyroid_ultrasound": "Not reported",
            "study_id": study.get("study_id", "study_9")
        }
        bio_values = {}
        if case.get("preop_ct") is not None:
            bio_values["calcitonin_pg_per_ml"] = case.get("preop_ct")
        if case.get("preop_cea") is not None:
            bio_values["CEA_ng_per_ml"] = case.get("preop_cea")
        if bio_values:
            patient["biochemical_values"] = bio_values
        patients.append(patient)
    return patients


def build_study10_patients(study):
    """convert familial study into patient records"""
    variant_info = (study.get("variant_info") or {}).get("ret_variant") or {}
    base_variant = parse_ret_variant_string(variant_info.get("protein"))
    patients = []
    for record in study.get("patients", []):
        genetics = record.get("genetics")
        has_ret = False
        if isinstance(genetics, dict):
            for key, value in genetics.items():
                if "ret" in key.lower() and str(value).lower() in {"present", "positive", "heterozygous"}:
                    has_ret = True
        elif isinstance(genetics, str):
            has_ret = "ret" in genetics.lower() and any(term in genetics.lower() for term in ["present", "positive"])
        if not has_ret:
            continue
        diag_list = [str(item).lower() for item in record.get("diagnoses", [])]
        bio_data = record.get("biochemical_data", [])
        calcitonin_high = any("calcitonin" in str(item).lower() and "elevated" in str(item).lower() for item in bio_data)
        cea_high = any("cea" in str(item).lower() and "elevated" in str(item).lower() for item in bio_data)
        patient = {
            "patient_id": f"{study.get('study_id', 'study_10')}_{record.get('id', len(patients) + 1)}",
            "age": record.get("age_at_presentation") or record.get("age_at_first_pheo") or record.get("age") or record.get("age_of_death"),
            "gender": record.get("sex"),
            "relationship": str(record.get("role", "Relative")).title(),
            "ret_variant": base_variant,
            "men2_syndrome": "Yes",
            "family_history_mtc": "Yes",
            "mtc_diagnosis": "Yes" if any("medullary thyroid carcinoma" in diag for diag in diag_list) else "Suspected (elevated calcitonin)",
            "pheochromocytoma": "Yes" if any("pheochromocytoma" in diag for diag in diag_list) else "No",
            "hyperparathyroidism": "Yes" if any("hyperparathyroidism" in diag for diag in diag_list) else "No",
            "calcitonin_level": "elevated" if calcitonin_high else record.get("calcitonin_level"),
            "cea_level": "elevated" if cea_high else record.get("cea_level"),
            "thyroid_ultrasound": "; ".join(record.get("imaging", [])),
            "family_screening": "Yes"
        }
        patients.append(patient)
    return patients


def build_study11_patients(study):
    """convert MEN2B case into patient record"""
    variant = parse_ret_variant_string(((study.get("variant_info") or {}).get("ret_variant") or {}).get("protein"))
    patients = []
    for record in study.get("patients", []):
        if str(record.get("id")).lower() != "proband":
            continue
        lab_entries = convert_biochemical_entries(record.get("biochemical_data", []), test_key="analyte")
        calcitonin_entry = next((entry for entry in lab_entries if entry.get("test") and "calcitonin" in entry["test"].lower()), None)
        cea_entry = next((entry for entry in lab_entries if entry.get("test") and "cea" in entry["test"].lower()), None)
        patient = {
            "patient_id": f"{study.get('study_id', 'study_11')}_proband",
            "age": record.get("age"),
            "gender": record.get("sex"),
            "relationship": "Proband",
            "ret_variant": variant,
            "men2_syndrome": "Yes",
            "mtc_diagnosis": "Yes",
            "pheochromocytoma": "No",
            "hyperparathyroidism": "No",
            "family_history_mtc": "No",
            "biochemical_data": lab_entries,
            "calcitonin_level": calcitonin_entry.get("value") if calcitonin_entry else None,
            "cea_level": cea_entry.get("value") if cea_entry else None,
            "thyroid_ultrasound": "; ".join(record.get("imaging", []))
        }
        patients.append(patient)
    return patients


def build_study12_patients(study):
    """convert Annales d'Endocrinologie Y791F report"""
    variant = parse_ret_variant_string(((study.get("variant_info") or {}).get("ret_variant") or {}).get("protein"))
    patients = []
    for record in study.get("patients", []):
        if str(record.get("id")).lower() != "proband":
            continue
        labs = record.get("labs", {})
        patient = {
            "patient_id": f"{study.get('study_id', 'study_12')}_proband",
            "age": record.get("age"),
            "gender": record.get("sex"),
            "relationship": "Proband",
            "ret_variant": variant or "Y791F",
            "men2_syndrome": "Yes",
            "mtc_diagnosis": "No",
            "pheochromocytoma": "Yes",
            "hyperparathyroidism": "No",
            "family_history_mtc": "No",
            "calcitonin_level": labs.get("calcitonin_basal") or labs.get("calcitonin_calcium_stimulated"),
            "calcitonin_normal_range": "<10 ng/L",
            "cea_level": None,
            "thyroid_ultrasound": (record.get("thyroid_workup") or {}).get("ultrasound"),
            "biochemical_values": {"calcitonin_pg_per_ml": labs.get("calcitonin_basal")} if labs.get("calcitonin_basal") is not None else None
        }
        if patient["biochemical_values"] is None:
            patient.pop("biochemical_values")
        patients.append(patient)
    return patients


def build_study13_patients(study):
    """convert Surgery Today S891A report"""
    variant = parse_ret_variant_string(((study.get("variant_info") or {}).get("ret_variant") or {}).get("protein"))
    patients = []
    for record in study.get("patients", []):
        record_id = str(record.get("id"))
        if record_id == "proband":
            labs = record.get("labs", {})
            bio_values = {}
            calcitonin_values = []
            if labs.get("basal_calcitonin_pg_ml") is not None:
                calcitonin_values.append(labs.get("basal_calcitonin_pg_ml"))
            if labs.get("stimulated_calcitonin_pg_ml") is not None:
                calcitonin_values.append(labs.get("stimulated_calcitonin_pg_ml"))
            if calcitonin_values:
                bio_values["calcitonin_pg_per_ml"] = calcitonin_values
            patient = {
                "patient_id": f"{study.get('study_id', 'study_13')}_proband",
                "age": record.get("age"),
                "gender": record.get("sex"),
                "relationship": "Proband",
                "ret_variant": variant or "S891A",
                "men2_syndrome": "Yes",
                "mtc_diagnosis": "No",
                "pheochromocytoma": "Yes",
                "hyperparathyroidism": "Yes",
                "family_history_mtc": "Yes",
                "calcitonin_level": labs.get("basal_calcitonin_pg_ml"),
                "cea_level": None,
                "calcitonin_normal_range": "0-7.5 pg/mL",
                "thyroid_ultrasound": (record.get("follow_up") or {}).get("thyroid_ultrasound_preop"),
                "biochemical_values": bio_values if bio_values else None
            }
            if patient["biochemical_values"] is None:
                patient.pop("biochemical_values")
            patients.append(patient)
        elif record_id == "son_27":
            thyroid_info = record.get("thyroid", {})
            patient = {
                "patient_id": f"{study.get('study_id', 'study_13')}_son27",
                "age": record.get("age"),
                "gender": record.get("sex"),
                "relationship": "Son",
                "ret_variant": variant or "S891A",
                "men2_syndrome": "Yes",
                "mtc_diagnosis": "No",
                "pheochromocytoma": "No",
                "hyperparathyroidism": "No",
                "family_history_mtc": "Yes",
                "calcitonin_level": thyroid_info.get("basal_calcitonin"),
                "calcitonin_normal_range": "0-7.5 pg/mL",
                "cea_level": None,
                "thyroid_ultrasound": thyroid_info.get("ultrasound")
            }
            patients.append(patient)
    return patients


def extract_patients_from_study(study):
    """extract patient objects regardless of original study schema"""
    study_id = study.get("study_id")
    if study_id == "study_2":
        return build_study2_patients(study)
    if study_id == "study_8":
        return prepare_study8_patients(study)
    if study_id == "study_9":
        return build_study9_patients(study)
    if study_id == "study_10":
        return build_study10_patients(study)
    if study_id == "study_11":
        return build_study11_patients(study)
    if study_id == "study_12":
        return build_study12_patients(study)
    if study_id == "study_13":
        return build_study13_patients(study)
    if study_id == "study_14":
        return build_study14_patients(study)
    raw_patients = study.get("patient_data")
    if isinstance(raw_patients, dict):
        patient_copy = raw_patients.copy()
        patient_copy.setdefault("patient_id", f"{study_id}_patient")
        return [patient_copy]
    if isinstance(raw_patients, list):
        return raw_patients
    generic_patients = study.get("patients")
    if isinstance(generic_patients, list):
        return generic_patients
    return []


def get_study_display_name(study_id):
    """map internal identifiers to human-readable names"""
    return STUDY_NAME_MAP.get(study_id, study_id)


def normalize_patient_record(patient):
    """harmonize field names across studies before dataframe creation"""
    record = patient.copy()

    if 'age' not in record:
        if 'age_at_diagnosis' in record:
            record['age'] = record['age_at_diagnosis']
        elif 'age_at_first_symptoms' in record:
            record['age'] = record['age_at_first_symptoms']

    if 'relationship' not in record and record.get('relationship_in_family'):
        record['relationship'] = record['relationship_in_family']

    if 'gender' not in record and record.get('sex'):
        sex_value = str(record['sex']).strip().lower()
        if sex_value in {'female', 'f'}:
            record['gender'] = 'Female'
        elif sex_value in {'male', 'm'}:
            record['gender'] = 'Male'

    genotype_field = record.get('genotype')
    if not record.get('ret_variant') and genotype_field:
        genotype_values = genotype_field if isinstance(genotype_field, list) else [genotype_field]
        for genotype in genotype_values:
            match = re.search(r'([A-Z]\d+[A-Z])', str(genotype))
            if match:
                record['ret_variant'] = match.group(1)
                break

    if 'mtc_diagnosis' not in record and record.get('medullary_thyroid_carcinoma_present'):
        record['mtc_diagnosis'] = record['medullary_thyroid_carcinoma_present']

    if 'pheochromocytoma' not in record and record.get('pheochromocytoma_present'):
        record['pheochromocytoma'] = record['pheochromocytoma_present']

    if 'hyperparathyroidism' not in record and record.get('primary_hyperparathyroidism_present'):
        record['hyperparathyroidism'] = record['primary_hyperparathyroidism_present']

    if not record.get('biochemical_data') and isinstance(record.get('biochemical_results'), list):
        record['biochemical_data'] = convert_biochemical_entries(record['biochemical_results'])

    if not record.get('biochemical_values') and isinstance(record.get('biochemical_data'), list):
        bio_values = {}

        def append_measurement(key, raw_value, normal_value=None):
            values = parse_numeric_measurements(raw_value, normal_value=normal_value)
            if not values:
                return
            if key not in bio_values:
                bio_values[key] = values if len(values) > 1 else values[0]
            else:
                existing = bio_values[key]
                if not isinstance(existing, list):
                    existing = [existing]
                existing.extend(values)
                bio_values[key] = existing

        for entry in record['biochemical_data']:
            test_name = str(entry.get('test') or '').lower()
            raw_value = entry.get('value')
            if not test_name or raw_value is None:
                continue
            if isinstance(raw_value, str):
                stripped = raw_value.strip()
                if not stripped or stripped.lower() in {'unknown', 'nid'}:
                    continue
                raw_value = stripped
            elif isinstance(raw_value, list) and not raw_value:
                continue
            if 'calcitonin' in test_name:
                append_measurement('calcitonin_pg_per_ml', raw_value, normal_value=0.2)
            elif 'cea' in test_name:
                append_measurement('CEA_ng_per_ml', raw_value)

        if bio_values:
            record['biochemical_values'] = bio_values

    return record


def parse_numeric_measurements(value, normal_value=None):
    """convert heterogeneous numeric inputs (scalar/list/str) into float list"""
    results = []
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return results

    if isinstance(value, dict):
        for entry in value.values():
            results.extend(parse_numeric_measurements(entry, normal_value))
        return [float(v) for v in results if v is not None]

    if isinstance(value, list):
        for entry in value:
            results.extend(parse_numeric_measurements(entry, normal_value))
        return [float(v) for v in results if v is not None]

    if isinstance(value, (int, float)):
        return [float(value)]

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return results
        lowered = cleaned.lower()
        missing_tokens = ['unknown', 'na', 'n/a', 'not available']
        if any(token in lowered for token in missing_tokens):
            return results
        normal_tokens = ['undetectable', 'normal', 'not detected', 'not specified', 'nid']
        if any(token in lowered for token in normal_tokens):
            if normal_value is not None:
                return [float(normal_value)]
            return results
        matches = re.findall(r'(\d+\.?\d*)', cleaned.replace(',', ' '))
        return [float(m) for m in matches]

    return results


def summarize_measurements(values):
    """summarize numeric list using median"""
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return float(np.median(cleaned))


def build_biomarker_pairs(df):
    """collect all observed calcitonin and CEA pairs"""
    required_cols = ['patient_id', 'study_id', 'calcitonin_level_numeric', 'cea_level_numeric']
    if not set(required_cols).issubset(df.columns):
        return pd.DataFrame(columns=required_cols)

    observed = df.dropna(subset=['calcitonin_level_numeric', 'cea_level_numeric']).copy()
    if observed.empty:
        return pd.DataFrame(columns=required_cols)
    return observed[required_cols]


def fit_biomarker_regression(pairs_df):
    """fit simple linear regression of CEA on calcitonin"""
    if pairs_df.empty or pairs_df['cea_level_numeric'].nunique() == 0:
        return {'slope': 0.05, 'intercept': 0.5, 'residual_std': 0.1, 'samples': 0}

    if len(pairs_df) == 1:
        return {
            'slope': 0.05,
            'intercept': float(pairs_df['cea_level_numeric'].iloc[0]),
            'residual_std': 0.1,
            'samples': len(pairs_df)
        }

    model = LinearRegression()
    X = pairs_df[['calcitonin_level_numeric']]
    y = pairs_df['cea_level_numeric']
    model.fit(X, y)
    preds = model.predict(X)
    residual_std = float(np.std(y - preds)) if len(y) > 1 else 0.1

    return {
        'slope': float(model.coef_[0]),
        'intercept': float(model.intercept_),
        'residual_std': max(residual_std, 0.05),
        'samples': len(pairs_df)
    }


def estimate_cea_from_calcitonin(calcitonin_value, regression_params, rng=None):
    """estimate cea using linear fit with gaussian noise"""
    rng = rng or np.random.default_rng()
    slope = regression_params.get('slope', 0.05) if regression_params else 0.05
    intercept = regression_params.get('intercept', 0.5) if regression_params else 0.5
    residual_std = regression_params.get('residual_std', 0.3) if regression_params else 0.3
    if calcitonin_value is None or np.isnan(calcitonin_value):
        calcitonin_value = 5.0
    estimate = slope * float(calcitonin_value) + intercept
    noise = rng.normal(0, residual_std)
    return float(max(0.0, estimate + noise))


def predictive_mean_matching(observed_series, predicted_series, n_neighbors=5):
    """apply predictive mean matching so imputed values reuse observed donors"""
    observed = observed_series.dropna()
    adjusted = predicted_series.copy()
    if observed.empty:
        return adjusted.clip(lower=0.0)

    donors = observed.values
    rng = np.random.default_rng(42)
    missing_indices = observed_series[observed_series.isna()].index

    for idx in missing_indices:
        pred_val = predicted_series.loc[idx]
        distances = np.abs(donors - pred_val)
        donor_count = min(n_neighbors, len(donors))
        nearest = np.argsort(distances)[:donor_count]
        choice = rng.choice(nearest)
        adjusted.loc[idx] = float(max(0.0, donors[choice]))

    adjusted.loc[observed.index] = observed
    return adjusted


def run_mice_pmm_imputation(df, observed_pairs):
    """execute MICE (IterativeImputer) followed by PMM adjustment"""
    info = {
        'observed_before': int(df['cea_level_numeric'].notna().sum()),
        'missing_before': int(df['cea_level_numeric'].isna().sum())
    }

    if info['missing_before'] == 0 or observed_pairs.empty:
        df['cea_imputed_flag'] = df['cea_level_numeric'].isna().astype(int)
        info['missing_after'] = info['missing_before']
        info['strategy'] = 'skipped' if observed_pairs.empty else 'not_required'
        info['mice_iterations'] = 0
        return df, info

    imputer = IterativeImputer(
        random_state=42,
        sample_posterior=True,
        max_iter=20,
        min_value=0.0,
        imputation_order='ascending'
    )
    imputer.fit(observed_pairs[['calcitonin_level_numeric', 'cea_level_numeric']])

    transformed = imputer.transform(df[['calcitonin_level_numeric', 'cea_level_numeric']])
    predicted_cea = pd.Series(transformed[:, 1], index=df.index)
    adjusted = predictive_mean_matching(df['cea_level_numeric'], predicted_cea)
    missing_mask = df['cea_level_numeric'].isna()
    df.loc[missing_mask, 'cea_level_numeric'] = adjusted.loc[missing_mask]
    df['cea_level_numeric'] = df['cea_level_numeric'].clip(lower=0.0)
    df['cea_imputed_flag'] = missing_mask.astype(int)

    info['missing_after'] = int(df['cea_level_numeric'].isna().sum())
    info['strategy'] = 'mice_pmm'
    info['mice_iterations'] = getattr(imputer, 'n_iter_', None)
    return df, info


def save_biomarker_summary(summary_path, correlation_value, observed_pairs, imputation_info, regression_params):
    """persist textual summary of biomarker processing"""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("BIOMARKER INTEGRATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        study_count = observed_pairs['study_id'].nunique() if not observed_pairs.empty else 0
        f.write(f"Observed correlation sample size: {len(observed_pairs)} pairs across {study_count} studies\n")
        if np.isnan(correlation_value):
            f.write("Pearson correlation: insufficient data\n")
        else:
            f.write(f"Pearson correlation (calcitonin vs CEA): {correlation_value:.4f}\n")
        f.write(f"Regression slope: {regression_params.get('slope'):.4f}\n")
        f.write(f"Regression intercept: {regression_params.get('intercept'):.4f}\n")
        f.write(f"Residual std: {regression_params.get('residual_std'):.4f}\n")
        if not observed_pairs.empty:
            f.write("Pair contributions by study:\n")
            for study_id, count in observed_pairs['study_id'].value_counts().items():
                f.write(f"- {get_study_display_name(study_id)}: {count} pairs\n")
        f.write("\nMICE + PMM Imputation\n")
        f.write(f"Observed CEA values before: {imputation_info['observed_before']}\n")
        f.write(f"Missing CEA values before: {imputation_info['missing_before']}\n")
        f.write(f"Strategy: {imputation_info['strategy']}\n")
        f.write(f"MICE iterations: {imputation_info.get('mice_iterations', 0)}\n")
        f.write(f"Missing after imputation: {imputation_info['missing_after']}\n")


def save_biomarker_plot(observed_pairs_df, df):
    """plot correlation and imputed distributions"""
    if observed_pairs_df.empty:
        return

    os.makedirs('charts', exist_ok=True)
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_studies = observed_pairs_df['study_id'].unique()
    cmap = plt.get_cmap('tab20', max(len(unique_studies), 1))
    for idx, study_id in enumerate(unique_studies):
        subset = observed_pairs_df[observed_pairs_df['study_id'] == study_id]
        ax.scatter(
            subset['calcitonin_level_numeric'],
            subset['cea_level_numeric'],
            label=get_study_display_name(study_id),
            color=cmap(idx),
            s=60
        )

    imputed_df = df[df['cea_imputed_flag'] == 1]

    if not imputed_df.empty:
        ax.scatter(
            imputed_df['calcitonin_level_numeric'],
            imputed_df['cea_level_numeric'],
            label='Imputed (MICE+PMM)',
            color='tab:green',
            alpha=0.6
        )

    ax.set_xlabel('Calcitonin (pg/mL)')
    ax.set_ylabel('CEA (ng/mL)')
    ax.set_title('Calcitonin vs CEA Across Studies')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join('charts', 'calcitonin_cea_relationship.png'), dpi=200)
    plt.close(fig)

def create_paper_dataset():
    """extract key data points from research paper and convert to structured dataframe"""

    # load data from json files in raw data folder
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

    # load study data
    studies = []
    study_files = [
        'study_1.json',
        'study_2.json',
        'study_3.json',
        'study_4.json',
        'study_5.json',
        'study_6.json',
        'study_7.json',
        'study_8.json',
        'study_9.json',
        'study_10.json',
        'study_11.json',
        'study_12.json',
        'study_13.json',
        'study_14.json'
    ]
    for study_file in study_files:
        with open(os.path.join(dataset_dir, study_file), 'r', encoding='utf-8') as f:
            studies.append(json.load(f))

    # load literature data
    with open(os.path.join(dataset_dir, 'literature_data.json'), 'r', encoding='utf-8') as f:
        literature_data = json.load(f)

    # load mutation characteristics
    with open(os.path.join(dataset_dir, 'mutation_characteristics.json'), 'r', encoding='utf-8') as f:
        mutation_characteristics = json.load(f)

    # combine into paper_data structure
    paper_data = {
        "studies": studies,
        "literature_data": literature_data,
        "mutation_characteristics": mutation_characteristics
    }
    # combine patient data from all studies
    all_patients = []
    source_id_counter = 0
    
    for study in paper_data["studies"]:
        study_id = study["study_id"]
        study_patients = extract_patients_from_study(study)
        for patient in study_patients:
            patient_copy = normalize_patient_record(patient)
            patient_copy['study_id'] = study_id
            patient_copy['source_id'] = source_id_counter
            all_patients.append(patient_copy)
            source_id_counter += 1
    
    # create structured dataframe from combined patient data
    df = pd.DataFrame(all_patients)
    
    # remove duplicates based on patient_id
    print(f"Before deduplication: {len(df)} patients")
    df = df.drop_duplicates(subset=['study_id', 'patient_id'], keep='first')
    print(f"After deduplication: {len(df)} patients")

    # ensure required columns exist with sensible defaults
    default_string_columns = {
        'family_screening': 'No',
        'family_history_mtc': 'No',
        'c_cell_hyperplasia': 'No',
        'c_cell_disease_suspected': 'No',
        'men2_syndrome': 'No',
        'pheochromocytoma': 'No',
        'hyperparathyroidism': 'No',
        'calcitonin_level': '',
        'cea_level': '',
        'thyroid_ultrasound': '',
        'relationship': 'Proband'
    }
    for col, default in default_string_columns.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    if 'thyroid_nodule_count' not in df.columns:
        df['thyroid_nodule_count'] = 0
    df['thyroid_nodule_count'] = pd.to_numeric(df['thyroid_nodule_count'], errors='coerce').fillna(0)

    if 'gender' not in df.columns:
        df['gender'] = 'Female'
    df['gender'] = df['gender'].fillna('Female')

    if 'age' not in df.columns:
        df['age'] = np.nan
    df['age'] = pd.to_numeric(df.get('age'), errors='coerce')
    if df['age'].isna().all():
        df['age'] = df['age'].fillna(45.0)
    else:
        df['age'] = df['age'].fillna(df['age'].median())

    # feature engineering

    # Extract RET variant from ret_variant field
    # Studies 1-3 patients are all K666N carriers (variant not explicitly stored in JSON)
    # Study 4 has explicit variant field for each patient
    df['ret_variant'] = df['ret_variant'].fillna('K666N')

    # Create RET risk level mapping based on ATA guidelines
    # Level 1 (Moderate): L790F, Y791F, V804M, S891A, K666N
    # Level 2 (High): C618S, C630R, C620Y
    # Level 3 (Highest): C634R, C634Y, C634W
    ret_risk_mapping = {
        'K666N': 1,
        'L790F': 1,
        'Y791F': 1,
        'V804M': 1,
        'S891A': 1,
        'C618S': 2,
        'C630R': 2,
        'C620Y': 2,
        'C634R': 3,
        'C634Y': 3,
        'C634W': 3,
        'C634S': 3,
        'M918T': 3,
        'A883F': 3,
        'C620W': 2,
        'C630G': 2,
        'E505_G506del': 2,
        'E632_C634del': 3,
        'E632_L633del': 3,
        'V899_E902del': 2,
        'D898_E901del': 2
    }
    df['ret_risk_level'] = df['ret_variant'].map(ret_risk_mapping).fillna(1).astype(int)
    
    # convert calcitonin levels to numeric (supporting multiple study formats)
    def extract_calcitonin_numeric(row):
        # Study 4 explicit numeric column
        if 'calcitonin_preoperative_basal' in row and pd.notna(row.get('calcitonin_preoperative_basal')):
            return float(row['calcitonin_preoperative_basal'])

        calcitonin_str = str(row.get('calcitonin_level', '')).lower()
        if 'undetectable' in calcitonin_str or 'k/k' in calcitonin_str:
            return 0.2  # minimal detectable

        match = re.search(r'(\d+\.?\d*)', calcitonin_str)
        if match:
            return float(match.group(1))

        bio_values = row.get('biochemical_values', {}) or {}
        if isinstance(bio_values, dict):
            measurements = parse_numeric_measurements(bio_values.get('calcitonin_pg_per_ml'), normal_value=0.2)
            summary = summarize_measurements(measurements)
            if summary is not None:
                return summary

        return 0.0

    df['calcitonin_level_numeric'] = df.apply(extract_calcitonin_numeric, axis=1)

    def extract_cea_numeric(row):
        values = []
        cea_raw = row.get('cea_level')
        if pd.notna(cea_raw):
            values.extend(parse_numeric_measurements(cea_raw))

        bio_values = row.get('biochemical_values', {}) or {}
        if isinstance(bio_values, dict):
            values.extend(parse_numeric_measurements(bio_values.get('CEA_ng_per_ml')))

        summary = summarize_measurements(values)
        return summary

    df['cea_level_numeric'] = df.apply(extract_cea_numeric, axis=1)

    biomarker_pairs_df = build_biomarker_pairs(df)
    correlation_value = np.nan
    if not biomarker_pairs_df.empty and biomarker_pairs_df['cea_level_numeric'].nunique() > 1:
        correlation_value = biomarker_pairs_df['calcitonin_level_numeric'].corr(biomarker_pairs_df['cea_level_numeric'])

    regression_input = biomarker_pairs_df[['calcitonin_level_numeric', 'cea_level_numeric']] \
        if not biomarker_pairs_df.empty else biomarker_pairs_df
    regression_params = fit_biomarker_regression(regression_input)
    df, imputation_info = run_mice_pmm_imputation(df, biomarker_pairs_df)
    df['cea_elevated'] = (df['cea_level_numeric'].fillna(0) > 5).astype(int)

    summary_path = Path('results') / 'biomarker_ceaimputation_summary.txt'
    save_biomarker_summary(summary_path, correlation_value, biomarker_pairs_df, imputation_info, regression_params)
    save_biomarker_plot(biomarker_pairs_df, df)

    if np.isnan(correlation_value):
        print("Calcitonin<->CEA correlation: insufficient paired data across studies.")
    else:
        pair_count = len(biomarker_pairs_df)
        study_count = biomarker_pairs_df['study_id'].nunique()
        print(f"Calcitonin<->CEA correlation (n={pair_count} pairs across {study_count} studies): {correlation_value:.3f}")
    print(f"CEA imputation strategy: {imputation_info['strategy']} "
          f"(missing {imputation_info['missing_before']} -> {imputation_info['missing_after']})")
    
    # calcitonin elevated flag - handle different normal ranges
    def determine_calcitonin_elevated(row):
        calcitonin_level = str(row.get('calcitonin_level', '')).lower()
        calcitonin_numeric = row['calcitonin_level_numeric']
        if pd.isna(calcitonin_numeric):
            calcitonin_numeric = 0.0
        gender = row.get('gender', 0)  # 0=Female, 1=Male

        # handle special cases
        if 'undetectable' in calcitonin_level or 'normal' in calcitonin_level:
            return 0
        if 'not screened' in calcitonin_level or 'unknown' in calcitonin_level or 'not evaluated' in calcitonin_level:
            return 0
        if 'not elevated' in calcitonin_level or 'within normal' in calcitonin_level:
            return 0
        if any(token in calcitonin_level for token in ['elevated', 'high', 'raised']):
            if 'not elevated' not in calcitonin_level and 'non-elevated' not in calcitonin_level:
                return 1

        # extract normal range from calcitonin_normal_range
        normal_range = str(row.get('calcitonin_normal_range', ''))

        # Study 4 uses gender-specific ranges: Males <10.3, Females <4.3
        if 'Males <10.3' in normal_range or 'Females <4.3' in normal_range:
            threshold = 10.3 if gender == 1 else 4.3
            return 1 if calcitonin_numeric > threshold else 0
        # Studies 1-3 ranges
        elif '0-5.1' in normal_range or '0.0-5.1' in normal_range:
            return 1 if calcitonin_numeric > 5.1 else 0
        elif '0-7.5' in normal_range or '0.0-7.5' in normal_range:
            return 1 if calcitonin_numeric > 7.5 else 0
        else:
            # default to 7.5 if range unclear
            return 1 if calcitonin_numeric > 7.5 else 0

    df['calcitonin_elevated'] = df.apply(determine_calcitonin_elevated, axis=1)
    
    # gender encoding (0=female, 1=male)
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    if df['gender'].isna().any():
        gender_mode = df['gender'].mode()
        default_gender = int(gender_mode.iloc[0]) if not gender_mode.empty else 0
        df['gender'] = df['gender'].fillna(default_gender)
    df['gender'] = df['gender'].astype(int)
    
    # thyroid nodules
    df['thyroid_nodules_present'] = df['thyroid_ultrasound'].str.contains('nodules', case=False, na=False).astype(int)
    df['multiple_nodules'] = df['thyroid_nodule_count'].fillna(0) > 1
    df['multiple_nodules'] = df['multiple_nodules'].astype(int)
    
    # family history of MTC (based on relationship, family screening, and explicit family_history_mtc field)
    relationship_family = df['relationship'].isin([
        'Sister', 'Father', 'Paternal grandmother', 'Proband\'s daughter',
        'Sister\'s son', 'Mother', 'Brother', 'Daughter', 'Son'
    ])
    family_screening_yes = df['family_screening'].fillna('No').str.lower() == 'yes'
    explicit_family_history = df['family_history_mtc'].fillna('No').str.lower() == 'yes'
    
    df['family_history_mtc'] = (relationship_family | family_screening_yes | explicit_family_history).astype(int)
    
    # MTC diagnosis (target variable)
    df['mtc_diagnosis'] = df['mtc_diagnosis'].map({
        'No': 0, 
        'Yes': 1, 
        'Suspected (declined workup)': 0,
        'Suspected (elevated calcitonin)': 0,
        'Not screened': 0
    }).fillna(0).astype(int)
    
    # C-cell disease (includes MTC and C-cell hyperplasia/suspected)
    df['c_cell_disease'] = ((df['mtc_diagnosis'] == 1) | 
                           (df['c_cell_hyperplasia'] == 'Yes') | 
                           (df['c_cell_disease_suspected'] == 'Yes')).astype(int)
    
    # MEN2 syndrome (currently all No in paper data)
    df['men2_syndrome'] = df['men2_syndrome'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
    
    # other clinical features
    df['pheochromocytoma'] = df['pheochromocytoma'].map({'No': 0, 'Yes': 1, 'Not screened': 0}).fillna(0).astype(int)
    df['hyperparathyroidism'] = df['hyperparathyroidism'].map({'No': 0, 'Yes': 1, 'Not screened': 0}).fillna(0).astype(int)
    
    # age groups for stratification
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['young', 'middle', 'elderly', 'very_elderly'])
    
    # other flags (post-outcome/leaky features intentionally not included in model)
    
    # select final columns for ML
    final_columns = [
        'source_id', 'study_id', 'age', 'gender', 'ret_variant', 'ret_risk_level',
        'calcitonin_elevated', 'calcitonin_level_numeric',
        'cea_level_numeric', 'cea_elevated', 'cea_imputed_flag',
        'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc', 'mtc_diagnosis',
        'c_cell_disease', 'men2_syndrome', 'pheochromocytoma', 'hyperparathyroidism', 'age_group'
    ]
    
    df_final = df[final_columns].copy()

    # save paper-only dataset
    df_final.to_csv('data/processed/ret_multivariant_training_data.csv', index=False)

    # create expanded dataset with literature cases
    expanded_df = create_expanded_dataset(df_final, paper_data, regression_params)

    # save expanded dataset
    expanded_df.to_csv('data/processed/ret_multivariant_expanded_training_data.csv', index=False)
    
    return df_final, expanded_df

def create_expanded_dataset(paper_df, paper_data, regression_params=None):
    """create expanded dataset with literature cases"""

    regression_params = regression_params or {'slope': 0.05, 'intercept': 0.5, 'residual_std': 0.3, 'samples': 0}

    # literature MTC diagnosis ages
    lit_ages = paper_data['literature_data']['mtc_diagnosis_ages']

    # create synthetic cases based on literature
    np.random.seed(42)  # for reproducibility
    rng = np.random.default_rng(42)

    # Get variant distribution from paper_df to create realistic synthetic cases
    variant_distribution = paper_df['ret_variant'].value_counts(normalize=True).to_dict()
    risk_level_distribution = paper_df['ret_risk_level'].value_counts(normalize=True).to_dict()

    synthetic_cases = []
    for age in lit_ages:
        # Sample variant based on distribution in real data
        ret_variant = np.random.choice(list(variant_distribution.keys()),
                                       p=list(variant_distribution.values()))
        ret_risk_level = paper_df[paper_df['ret_variant'] == ret_variant]['ret_risk_level'].iloc[0]

        # create case with MTC
        case = {
            'age': age,
            'gender': np.random.choice([0, 1]),  # random gender
            'ret_variant': ret_variant,
            'ret_risk_level': ret_risk_level,
            'calcitonin_elevated': 1,
            'calcitonin_level_numeric': np.random.uniform(15, 60),  # elevated calcitonin
            'thyroid_nodules_present': 1,
            'multiple_nodules': np.random.choice([0, 1]),
            'family_history_mtc': 1,
            'mtc_diagnosis': 1,
            'c_cell_disease': 1,
            'men2_syndrome': 0,  # rare in heterozygous carriers
            'pheochromocytoma': 0,
            'hyperparathyroidism': 0,
            'age_group': 'middle' if 30 <= age <= 50 else 'elderly' if age > 50 else 'young',
            'declined_evaluation': 0,
            'underwent_surgery': 1
        }
        case['cea_level_numeric'] = estimate_cea_from_calcitonin(case['calcitonin_level_numeric'], regression_params, rng)
        case['cea_elevated'] = int(case['cea_level_numeric'] > 5)
        case['cea_imputed_flag'] = 1
        synthetic_cases.append(case)

    # create additional control cases (non-MTC carriers)
    control_cases = []
    for i in range(len(lit_ages) * 2):  # 2:1 control to case ratio
        age = np.random.choice(paper_df['age'].tolist() + lit_ages)

        # Sample variant based on distribution
        ret_variant = np.random.choice(list(variant_distribution.keys()),
                                       p=list(variant_distribution.values()))
        ret_risk_level = paper_df[paper_df['ret_variant'] == ret_variant]['ret_risk_level'].iloc[0]

        control = {
            'age': age,
            'gender': np.random.choice([0, 1]),
            'ret_variant': ret_variant,
            'ret_risk_level': ret_risk_level,
            'calcitonin_elevated': 0,
            'calcitonin_level_numeric': np.random.uniform(0, 7.5),  # normal calcitonin
            'thyroid_nodules_present': np.random.choice([0, 1], p=[0.7, 0.3]),  # lower nodule rate
            'multiple_nodules': 0,  # controls less likely to have multiple nodules
            'family_history_mtc': np.random.choice([0, 1], p=[0.6, 0.4]),
            'mtc_diagnosis': 0,
            'c_cell_disease': 0,
            'men2_syndrome': 0,
            'pheochromocytoma': 0,
            'hyperparathyroidism': 0,
            'age_group': 'middle' if 30 <= age <= 50 else 'elderly' if age > 50 else 'young',
            'declined_evaluation': 0,
            'underwent_surgery': 0
        }
        control['cea_level_numeric'] = estimate_cea_from_calcitonin(control['calcitonin_level_numeric'], regression_params, rng)
        control['cea_elevated'] = int(control['cea_level_numeric'] > 5)
        control['cea_imputed_flag'] = 1
        control_cases.append(control)
    
    # combine all cases
    expanded_df = pd.concat([paper_df, pd.DataFrame(synthetic_cases), pd.DataFrame(control_cases)], ignore_index=True)

    # fill any residual CEA gaps using regression estimate
    missing_cea_mask = expanded_df['cea_level_numeric'].isna()
    if missing_cea_mask.any():
        expanded_df.loc[missing_cea_mask, 'cea_level_numeric'] = expanded_df.loc[missing_cea_mask, 'calcitonin_level_numeric'] \
            .apply(lambda val: estimate_cea_from_calcitonin(val, regression_params, rng))
        expanded_df.loc[missing_cea_mask, 'cea_imputed_flag'] = 1

    expanded_df['cea_level_numeric'] = expanded_df['cea_level_numeric'].astype(float)
    expanded_df['cea_elevated'] = (expanded_df['cea_level_numeric'] > 5).astype(int)
    expanded_df['cea_imputed_flag'] = expanded_df['cea_imputed_flag'].fillna(0).astype(int)

    # remove duplicates based on key features (age, gender, calcitonin_level_numeric, ret_variant)
    print(f"Before expanded dataset deduplication: {len(expanded_df)} patients")
    expanded_df = expanded_df.drop_duplicates(subset=['age', 'gender', 'calcitonin_level_numeric', 'ret_variant'], keep='first')
    print(f"After expanded dataset deduplication: {len(expanded_df)} patients")

    return expanded_df

def print_dataset_info(df1, df2):
    """print dataset shapes and target distribution summaries"""
    print("=" * 60)
    print("DATASET CREATION SUMMARY")
    print("=" * 60)
    print(f"paper-only dataset shape: {df1.shape}")
    print(f"expanded dataset shape: {df2.shape}")
    print()
    
    # study breakdown
    print("STUDY BREAKDOWN:")
    study_counts = df1['study_id'].value_counts()
    for study_id, count in study_counts.items():
        print(f"- {get_study_display_name(study_id)}: {count} patients")
    print()

    # variant breakdown
    print("RET VARIANT BREAKDOWN:")
    variant_counts = df1['ret_variant'].value_counts()
    for variant, count in variant_counts.items():
        risk_level = df1[df1['ret_variant'] == variant]['ret_risk_level'].iloc[0]
        risk_label = {1: 'Moderate', 2: 'High', 3: 'Highest'}[risk_level]
        print(f"- {variant} (Risk Level {risk_level} - {risk_label}): {count} patients")
    print()
    
    print("PAPER-ONLY DATASET TARGET DISTRIBUTION:")
    print(f"MTC diagnosis cases: {df1['mtc_diagnosis'].sum()}/{len(df1)} ({df1['mtc_diagnosis'].mean():.1%})")
    print(f"C-cell disease cases: {df1['c_cell_disease'].sum()}/{len(df1)} ({df1['c_cell_disease'].mean():.1%})")
    print(f"MEN2 syndrome cases: {df1['men2_syndrome'].sum()}/{len(df1)} ({df1['men2_syndrome'].mean():.1%})")
    print(f"Pheochromocytoma cases: {df1['pheochromocytoma'].sum()}/{len(df1)} ({df1['pheochromocytoma'].mean():.1%})")
    print(f"Hyperparathyroidism cases: {df1['hyperparathyroidism'].sum()}/{len(df1)} ({df1['hyperparathyroidism'].mean():.1%})")
    print()
    
    print("EXPANDED DATASET TARGET DISTRIBUTION:")
    print(f"MTC diagnosis cases: {df2['mtc_diagnosis'].sum()}/{len(df2)} ({df2['mtc_diagnosis'].mean():.1%})")
    print(f"C-cell disease cases: {df2['c_cell_disease'].sum()}/{len(df2)} ({df2['c_cell_disease'].mean():.1%})")
    print(f"MEN2 syndrome cases: {df2['men2_syndrome'].sum()}/{len(df2)} ({df2['men2_syndrome'].mean():.1%})")
    print()
    
    print("KEY FEATURES SUMMARY:")
    print(f"Average age: {df2['age'].mean():.1f} years")
    print(f"Gender distribution (M/F): {df2[df2['gender']==1]['gender'].count()}/{df2[df2['gender']==0]['gender'].count()}")
    print(f"Calcitonin elevated rate: {df2['calcitonin_elevated'].mean():.1%}")
    print(f"Thyroid nodules rate: {df2['thyroid_nodules_present'].mean():.1%}")
    print("=" * 60)

if __name__ == "__main__":
    paper_df, expanded_df = create_paper_dataset()
    print_dataset_info(paper_df, expanded_df)
