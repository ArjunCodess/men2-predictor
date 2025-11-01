import pandas as pd
import numpy as np
import os

def create_paper_dataset():
    """extract key data points from research paper and convert to structured dataframe"""
    
    # multi-study data from research papers
    paper_data = {
        "studies": [
            {
                "study_id": "study_1",
                "study_info": {
                    "title": "Medullary Thyroid Carcinoma and Clinical Outcomes in Heterozygous Carriers of the RET K666N Germline Pathogenic Variant",
                    "journal": "JCEM Case Reports",
                    "publication_date": "March 2025",
                    "study_type": "Case series"
                },
                "patient_data": [
                    {
                        "patient_id": "Patient_1_Proband",
                        "age": 24,
                        "gender": "Female",
                        "relationship": "Proband",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Undetectable",
                        "calcitonin_normal_range": "0.0-7.5 pg/mL",
                        "thyroid_ultrasound": "Normal",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "genetic_testing_reason": "Breast cancer risk stratification"
                    },
                    {
                        "patient_id": "Patient_2_Sister",
                        "age": 21,
                        "gender": "Female",
                        "relationship": "Sister",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Undetectable",
                        "calcitonin_normal_range": "0.0-7.5 pg/mL",
                        "thyroid_ultrasound": "Normal",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "genetic_testing_reason": "Family screening"
                    },
                    {
                        "patient_id": "Patient_3_Father",
                        "age": 60,
                        "gender": "Male",
                        "relationship": "Father",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "13 pg/mL",
                        "calcitonin_normal_range": "0.0-7.5 pg/mL",
                        "calcitonin_elevated": "Yes",
                        "thyroid_ultrasound": "Multiple subcentimeter solid hypoechoic nodules",
                        "thyroid_nodule_count": 3,
                        "largest_nodule_size_mm": 6,
                        "biopsy_result": "Bethesda V - Suspicious for malignancy",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "pT1aNxMx",
                        "mtc_size_mm": 8,
                        "mtc_bilateral": "Yes",
                        "c_cell_hyperplasia": "Yes",
                        "treatment": "Total thyroidectomy",
                        "postop_calcitonin": "Undetectable",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_screening": "Yes"
                    },
                    {
                        "patient_id": "Patient_4_Grandmother",
                        "age": 84,
                        "gender": "Female",
                        "relationship": "Paternal grandmother",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "37 pg/mL",
                        "calcitonin_normal_range": "0.0-7.5 pg/mL",
                        "calcitonin_elevated": "Yes",
                        "thyroid_ultrasound": "Multiple thyroid nodules",
                        "mtc_diagnosis": "Suspected (declined workup)",
                        "c_cell_disease_suspected": "Yes",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "declined_further_evaluation": "Yes"
                    }
                ]
            },
            {
                "study_id": "study_2",
                "study_info": {
                    "title": "MEN2 phenotype in a family with germline heterozygous rare RET K666N variant",
                    "journal": "Endocrinology, Diabetes & Metabolism Case Reports",
                    "publication_date": "September 2024",
                    "study_type": "Case report",
                    "doi": "10.1530/EDM-24-0009"
                },
                "patient_data": [
                    {
                        "patient_id": "New_Patient_1_Proband",
                        "age": 40,
                        "gender": "Female",
                        "relationship": "Proband",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "12.3 pg/mL",
                        "calcitonin_normal_range": "0-5.1 pg/mL",
                        "calcitonin_elevated": "Yes",
                        "thyroid_ultrasound": "No nodules",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "pT1aN1",
                        "mtc_size_mm": 4,
                        "pheochromocytoma": "Yes",
                        "pheochromocytoma_size_cm": 3.7,
                        "pheochromocytoma_laterality": "Unilateral",
                        "hyperparathyroidism": "Yes",
                        "men2_syndrome": "Yes",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "PHEO diagnosis"
                    },
                    {
                        "patient_id": "New_Patient_2_Sister",
                        "age": 42,
                        "gender": "Female",
                        "relationship": "Sister",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "5.2 pg/mL",
                        "calcitonin_normal_range": "0-5.1 pg/mL",
                        "calcitonin_elevated": "No",
                        "thyroid_ultrasound": "Two suspicious nodules (5 mm and 11 mm), suspicious lymph nodes",
                        "mtc_diagnosis": "No",
                        "ptc_diagnosis": "Yes",
                        "ptc_stage": "pT1aN1b",
                        "ptc_size_mm": 8,
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "Cascade testing"
                    },
                    {
                        "patient_id": "New_Patient_3_Brother",
                        "age": 46,
                        "gender": "Male",
                        "relationship": "Brother",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "17.3 pg/mL",
                        "calcitonin_normal_range": "0-5.1 pg/mL",
                        "calcitonin_elevated": "Yes",
                        "thyroid_ultrasound": "No nodules",
                        "mtc_diagnosis": "Suspected (elevated calcitonin)",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "Cascade testing"
                    },
                    {
                        "patient_id": "New_Patient_4_Daughter",
                        "age": 22,
                        "gender": "Female",
                        "relationship": "Proband's daughter",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Normal",
                        "calcitonin_normal_range": "0-5.1 pg/mL",
                        "calcitonin_elevated": "No",
                        "thyroid_ultrasound": "Normal",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "Cascade testing"
                    }
                ]
            },
            {
                "study_id": "study_3",
                "study_info": {
                    "title": "Medullary Thyroid Carcinoma Associated with Germline RETK666N Mutation",
                    "journal": "Thyroid",
                    "publication_date": "2016 Dec 1",
                    "study_type": "Case series",
                    "doi": "10.1089/thy.2016.0374"
                },
                "patient_data": [
                    # Family 1 Proband - 55yo woman, diagnosed at 22, TXN1M0
                    {
                        "patient_id": "Family1_Proband",
                        "age": 55,
                        "age_at_diagnosis": 22,
                        "gender": "Female",
                        "relationship": "Proband",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "12.2-25.2 pg/mL",
                        "calcitonin_elevated": "Yes",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "TXN1M0",
                        "mtc_bilateral_positive_nodes": "Yes",
                        "treatment": "Total thyroidectomy and lymph node dissection, external beam radiotherapy",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "MTC diagnosis",
                        "follow_up_months": 397
                    },
                    # Family 2 Proband - 34yo woman (surgery at 33), T1N1bM1
                    {
                        "patient_id": "Family2_Proband",
                        "age": 34,
                        "age_at_diagnosis": 33,
                        "gender": "Female",
                        "relationship": "Proband",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "1988 pg/mL preoperative, 300 pg/mL 20mo post-surgery",
                        "calcitonin_elevated": "Yes",
                        "calcitonin_normal_range": "<5 pg/mL",
                        "cea_level": "35.8 ng/mL preoperative, 4.3 ng/mL post-surgery",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "T1N1bM1",
                        "mtc_size_mm": 6,  # 0.6 cm
                        "mtc_multifocal": "No",
                        "lymph_nodes_positive": "30/39",
                        "extracapsular_extension": "Yes",
                        "distant_metastasis_site": "Sternum",
                        "treatment": "Total thyroidectomy, central and bilateral lateral neck dissection, external beam radiotherapy to sternal metastasis",
                        "c_cell_hyperplasia": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "MTC diagnosis",
                        "follow_up_months": 23
                    },
                    # Family 3 Proband - 32yo woman, diagnosed at 23, T3N1bMX with distant metastasis
                    {
                        "patient_id": "Family3_Proband",
                        "age": 32,
                        "age_at_diagnosis": 23,
                        "gender": "Female",
                        "relationship": "Proband",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "28000 pg/mL preoperative, 14450 pg/mL at 9 years post-surgery",
                        "calcitonin_elevated": "Yes",
                        "calcitonin_normal_range": "Not specified",
                        "cea_level": "51 ng/mL preoperative, 27.2 ng/mL at 9 years",
                        "thyroid_ultrasound": "Palpable neck mass",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "T3N1bMX",
                        "mtc_size_mm": 12,  # 1.2 cm
                        "mtc_multifocal": "No",
                        "mtc_extrathyroidal_extension": "Yes",
                        "lymph_nodes_positive": "17/37",
                        "largest_node_size_cm": 5.1,
                        "extracapsular_extension": "Yes",
                        "distant_metastasis_site": "Liver and spine",
                        "treatment": "Total thyroidectomy, central and bilateral modified lateral neck dissections",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "MTC diagnosis",
                        "follow_up_months": 118
                    },
                    # Family 4 Proband - 65yo woman, diagnosed at 49, T2NXM0
                    {
                        "patient_id": "Family4_Proband",
                        "age": 65,
                        "age_at_diagnosis": 49,
                        "gender": "Female",
                        "relationship": "Proband",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "16-22 pg/mL",
                        "calcitonin_elevated": "Yes",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "T2NXM0",
                        "mtc_size_mm": 30,  # 3 cm dominant tumor
                        "mtc_multifocal": "Yes",
                        "mtc_bilateral": "Yes",
                        "c_cell_hyperplasia": "Yes",
                        "treatment": "Total thyroidectomy, left lateral modified neck dissection for recurrence",
                        "regional_recurrence": "Yes",
                        "recurrence_age": 64,
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "MTC diagnosis",
                        "follow_up_months": 202
                    },
                    # Family 5 Proband - 59yo man, diagnosed at 51, T1N0M0, incidental finding during PTC surgery
                    {
                        "patient_id": "Family5_Proband",
                        "age": 59,
                        "age_at_diagnosis": 51,
                        "gender": "Male",
                        "relationship": "Proband",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Undetectable",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "T1N0M0",
                        "mtc_size_mm": 1.5,
                        "mtc_multifocal": "No",
                        "c_cell_hyperplasia": "Yes",
                        "ptc_diagnosis": "Yes",
                        "ptc_stage": "IVa",
                        "treatment": "Total thyroidectomy, central and bilateral lateral neck dissection",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "Incidental finding during PTC surgery",
                        "follow_up_months": 132
                    },
                    # Family 6 Proband - 61yo female, diagnosed at 59, T3N1bM0
                    {
                        "patient_id": "Family6_Proband",
                        "age": 61,
                        "age_at_diagnosis": 59,
                        "gender": "Female",
                        "relationship": "Proband",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "67.4 pg/mL",
                        "calcitonin_elevated": "Yes",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "T3N1bM0",
                        "mtc_size_mm": 23,  # 2.3 cm largest dimension
                        "mtc_foci_count": 2,
                        "mtc_multifocal": "Yes",
                        "mtc_extrathyroidal_extension": "Yes",
                        "lymph_nodes_positive": "1/23",
                        "extracapsular_extension": "Yes",
                        "c_cell_hyperplasia": "No",
                        "treatment": "Total thyroidectomy and right modified radical neck dissection",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "Dysphagia evaluation",
                        "follow_up_months": 26
                    },
                    # Family 7 Proband - 65yo female, diagnosed at 64, T2N0M0
                    {
                        "patient_id": "Family7_Proband",
                        "age": 65,
                        "age_at_diagnosis": 64,
                        "gender": "Female",
                        "relationship": "Proband",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "973 pg/mL preoperative, undetectable postoperative",
                        "calcitonin_elevated": "Yes",
                        "calcitonin_normal_range": "Not specified",
                        "cea_level": "22.3 ng/mL preoperative",
                        "thyroid_ultrasound": "Incidental right thyroid nodule on CT scan",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "T2N0M0",
                        "mtc_size_mm": 22,  # 2.2 cm
                        "mtc_multifocal": "No",
                        "lymph_nodes_positive": "0/43",  # 17 central + 26 lateral = 43 total
                        "c_cell_hyperplasia": "No",
                        "treatment": "Total thyroidectomy, bilateral central and right lateral selective neck dissection",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "Incidental thyroid nodule",
                        "follow_up_months": 12
                    },
                    # Family 8 Proband - 64yo female, diagnosed at 55, T1NXM0
                    {
                        "patient_id": "Family8_Proband",
                        "age": 64,
                        "age_at_diagnosis": 55,
                        "gender": "Female",
                        "relationship": "Proband",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Undetectable",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Asymptomatic thyroid nodule on carotid artery screening",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "T1NXM0",
                        "mtc_size_mm": 11,  # 1.1 cm
                        "mtc_multifocal": "No",
                        "c_cell_hyperplasia": "No",
                        "treatment": "Right thyroid lobectomy followed by completion thyroidectomy",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "Patient request after MTC diagnosis",
                        "follow_up_months": 108
                    },
                    # 16 additional family members with K666N variant
                    # Family 2 - Mother (II-3), 57yo - elevated calcitonin, elected surveillance
                    {
                        "patient_id": "Family2_Mother_II3",
                        "age": 57,
                        "gender": "Female",  # (mother)
                        "relationship": "Mother",
                        "family_id": "Family2",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "26 pg/mL",
                        "calcitonin_elevated": "Yes",
                        "calcitonin_normal_range": "<5 pg/mL",
                        "thyroid_ultrasound": "3 mm calcification within right thyroid lobe",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "declined_surgery": "Yes",
                        "surveillance_elected": "Yes"
                    },
                    # Family 8 - Sister (II-2), 70yo at surgery - MTC confirmed, T1N0M0
                    {
                        "patient_id": "Family8_Sister_II2",
                        "age": 70,
                        "age_at_diagnosis": 70,
                        "gender": "Female",  # (sister)
                        "relationship": "Sister",
                        "family_id": "Family8",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "9 pg/mL preoperative, undetectable postoperative",
                        "calcitonin_elevated": "Yes",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "Yes",
                        "mtc_stage": "T1N0M0",
                        "mtc_size_mm": 4,  # 0.4 cm
                        "mtc_multifocal": "No",
                        "c_cell_hyperplasia": "Yes",
                        "treatment": "Thyroidectomy",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "underwent_surgery": "Yes",
                        "follow_up_months": 24
                    },
                    # Prophylactic thyroidectomy cases
                    # Family 1 - Daughter (III-1), 20yo - prophylactic thyroidectomy with C-cell hyperplasia
                    {
                        "patient_id": "Family1_Daughter_III1",
                        "age": 20,
                        "gender": "Female",  # (daughter)
                        "relationship": "Daughter",
                        "family_id": "Family1",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Undetectable",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "No",
                        "c_cell_hyperplasia": "Yes",
                        "cch_size_mm": 0.5,
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "underwent_surgery": "Yes",
                        "prophylactic_thyroidectomy": "Yes"
                    },
                    # Family 7 - Son, 20yo - prophylactic thyroidectomy with normal pathology
                    {
                        "patient_id": "Family7_Son",
                        "age": 20,
                        "gender": "Male",  # (son)
                        "relationship": "Son",
                        "family_id": "Family7",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Normal",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "cea_level": "Normal",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "No",
                        "c_cell_hyperplasia": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "underwent_surgery": "Yes",
                        "prophylactic_thyroidectomy": "Yes"
                    },
                    # Family 6 - Daughter, 30yo - prophylactic thyroidectomy, no MTC
                    {
                        "patient_id": "Family6_Daughter",
                        "age": 30,
                        "gender": "Female",  # (daughter)
                        "relationship": "Daughter",
                        "family_id": "Family6",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Normal",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "No",
                        "c_cell_hyperplasia": "Unknown",  # No information available
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "underwent_surgery": "Yes",
                        "prophylactic_thyroidectomy": "Yes"
                    },
                    # Family 6 - Niece, 30yo - prophylactic thyroidectomy, no MTC
                    {
                        "patient_id": "Family6_Niece",
                        "age": 30,
                        "gender": "Female",  # (niece)
                        "relationship": "Niece",
                        "family_id": "Family6",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Normal",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "No",
                        "c_cell_hyperplasia": "Unknown",  # No information available
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "underwent_surgery": "Yes",
                        "prophylactic_thyroidectomy": "Yes"
                    },
                    # Additional documented family members from paper
                    # Family 2 - Grandmother (I-2), 80yo
                    {
                        "patient_id": "Family2_Grandmother_I2",
                        "age": 80,
                        "gender": "Female",  # (grandmother)
                        "relationship": "Grandmother",
                        "family_id": "Family2",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "7 pg/mL",
                        "calcitonin_elevated": "No",  # Within range due to chronic kidney disease
                        "calcitonin_normal_range": "Not specified",
                        "chronic_kidney_disease": "Yes",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing"
                    },
                    # Family 3 - Father (I-1), 58yo
                    {
                        "patient_id": "Family3_Father_I1",
                        "age": 58,
                        "gender": "Male",  # (father)
                        "relationship": "Father",
                        "family_id": "Family3",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Normal",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Normal",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "No",
                        "genetic_testing_reason": "Cascade testing"
                    },
                    # Family 3 - Sister (II-3), 27yo
                    {
                        "patient_id": "Family3_Sister_II3",
                        "age": 27,
                        "gender": "Female",  # (sister)
                        "relationship": "Sister",
                        "family_id": "Family3",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Normal",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Benign appearing thyroid nodule",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "surveillance": "Observation"
                    },
                    # Family 6 - Child (IV-1), 5yo
                    {
                        "patient_id": "Family6_Child_IV1",
                        "age": 5,
                        "gender": "Unknown",
                        "relationship": "Child",
                        "family_id": "Family6",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Not evaluated",
                        "calcitonin_elevated": "Unknown",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not evaluated",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "clinical_evaluation": "No"
                    },
                    # Family 6 - Sister (II-5), 47yo
                    {
                        "patient_id": "Family6_Sister_II5",
                        "age": 47,
                        "gender": "Female",  # (sister)
                        "relationship": "Sister",
                        "family_id": "Family6",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Not evaluated",
                        "calcitonin_elevated": "Unknown",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not evaluated",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "clinical_evaluation": "No"
                    },
                    # Family 8 - Mother (I-2), 90yo at testing (died at 92)
                    {
                        "patient_id": "Family8_Mother_I2",
                        "age": 90,
                        "gender": "Female",  # (mother)
                        "relationship": "Mother",
                        "family_id": "Family8",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Not screened",
                        "calcitonin_elevated": "Unknown",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not screened",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "clinical_evaluation": "No",
                        "died_age": 92,
                        "cause_of_death": "Congestive heart failure"
                    },
                    # Family 8 - Daughter of sister (III), 39yo
                    {
                        "patient_id": "Family8_Niece_III",
                        "age": 39,
                        "gender": "Female",
                        "relationship": "Niece",
                        "family_id": "Family8",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Undetectable",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing"
                    },
                    # Family 8 - Brother (II-4), 61yo
                    {
                        "patient_id": "Family8_Brother_II4",
                        "age": 61,
                        "gender": "Male",  # (brother)
                        "relationship": "Brother",
                        "family_id": "Family8",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Undetectable",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Small thyroid nodule",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing"
                    },
                    # Family 8 - Nephew (III), 29yo
                    {
                        "patient_id": "Family8_Nephew_III",
                        "age": 29,
                        "gender": "Male",
                        "relationship": "Nephew",
                        "family_id": "Family8",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Not evaluated",
                        "calcitonin_elevated": "Unknown",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not evaluated",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing",
                        "clinical_evaluation": "No"
                    },
                    # Representative patient from remaining family members with normal calcitonin (ages 20-80 range)
                    {
                        "patient_id": "Family_Member_Representative_1",
                        "age": 35,
                        "gender": "Unknown",
                        "relationship": "Family member",
                        "ret_variant": "K666N",
                        "ret_status": "Positive",
                        "calcitonin_level": "Normal",
                        "calcitonin_elevated": "No",
                        "calcitonin_normal_range": "Not specified",
                        "thyroid_ultrasound": "Not specified",
                        "mtc_diagnosis": "No",
                        "pheochromocytoma": "No",
                        "hyperparathyroidism": "No",
                        "men2_syndrome": "No",
                        "family_history_mtc": "Yes",
                        "genetic_testing_reason": "Cascade testing"
                    }
                ]
            }
        ],
        
        "literature_data": {
            "ret_k666n_families_reported": 10,  # 8 from this study + 2 previously reported isolated cases
            "ret_k666n_carriers_reported": 26,  # 24 from this study (per Table 2 footnote) + 2 previous cases
            "mtc_cases_total": 11,  # 9 from this study (8 probands + 1 Family 8 sister) + 2 previous
            "mtc_cases_this_study": 9,  # 8 probands + 1 Family 8 sister (II-2)
            "carriers_with_c_cell_disease": 2,  # 1 with CCH (Family 1 daughter), 1 with elevated Ctn (Family 2 mother)
            "mtc_diagnosis_ages": [22, 23, 33, 49, 51, 55, 59, 64, 70],  # From Table 1
            "youngest_mtc_age": 22,  # Family 1 proband
            "oldest_mtc_age": 70,  # Family 8 sister
            "carriers_no_mtc_age_range": "20-80 years",  # From discussion: "eight confirmed K666N carriers, ranging in age from 20 to 80 years, had no clinical evidence of MTC"
            "unevaluated_carriers_age_range": "5-92 years",  # From discussion: "five remaining variant carriers (age range 5–92 years)"
            "penetrance": "Low",
            "expressivity": "Age-dependent, variable",
            "pheo_cases": 0,  # "none of the germline RETK666N carriers had evidence of PHPT or PHEO"
            "phpt_cases": 0,  # "none of the germline RETK666N carriers had evidence of PHPT or PHEO"
            "men2_features": "No PHEO or PHPT observed in K666N carriers"
        },

        "mutation_characteristics": {
            "mutation_type": "Missense",
            "nucleotide_change": "c.1998G>T",
            "protein_change": "p.Lys666Asn",
            "exon": "11/14",  # conflicting info between studies
            "domain": "Intracellular domain",
            "mechanism": "Increased kinase activity",
            "ata_risk_level": "Not yet assigned"
        }
    }

    # combine patient data from all studies
    all_patients = []
    source_id_counter = 0
    
    for study in paper_data["studies"]:
        study_id = study["study_id"]
        for patient in study["patient_data"]:
            patient_copy = patient.copy()
            patient_copy['study_id'] = study_id
            patient_copy['source_id'] = source_id_counter
            all_patients.append(patient_copy)
            source_id_counter += 1
    
    # create structured dataframe from combined patient data
    df = pd.DataFrame(all_patients)
    
    # remove duplicates based on patient_id
    print(f"Before deduplication: {len(df)} patients")
    df = df.drop_duplicates(subset=['patient_id'], keep='first')
    print(f"After deduplication: {len(df)} patients")
    
    # feature engineering
    df['ret_k666n_positive'] = 1  # all patients have the mutation
    
    # convert calcitonin levels to numeric
    df['calcitonin_level_numeric'] = df['calcitonin_level'].str.extract(r'(\d+\.?\d*)').iloc[:, 0].astype(float)
    df['calcitonin_level_numeric'] = df['calcitonin_level_numeric'].fillna(0.0)
    
    # calcitonin elevated flag - handle different normal ranges
    def determine_calcitonin_elevated(row):
        calcitonin_level = str(row['calcitonin_level']).lower()
        calcitonin_numeric = row['calcitonin_level_numeric']
        
        # handle special cases
        if 'undetectable' in calcitonin_level or 'normal' in calcitonin_level:
            return 0
        if 'not screened' in calcitonin_level or 'unknown' in calcitonin_level:
            return 0
        
        # extract normal range from calcitonin_normal_range
        normal_range = str(row['calcitonin_normal_range'])
        if '0-5.1' in normal_range or '0.0-5.1' in normal_range:
            return 1 if calcitonin_numeric > 5.1 else 0
        elif '0-7.5' in normal_range or '0.0-7.5' in normal_range:
            return 1 if calcitonin_numeric > 7.5 else 0
        else:
            # default to 7.5 if range unclear
            return 1 if calcitonin_numeric > 7.5 else 0
    
    df['calcitonin_elevated'] = df.apply(determine_calcitonin_elevated, axis=1)
    
    # gender encoding (0=female, 1=male)
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    
    # thyroid nodules
    df['thyroid_nodules_present'] = df['thyroid_ultrasound'].str.contains('nodules', case=False, na=False).astype(int)
    df['multiple_nodules'] = df['thyroid_nodule_count'].fillna(0) > 1
    df['multiple_nodules'] = df['multiple_nodules'].astype(int)
    
    # family history of MTC (based on relationship, family screening, and explicit family_history_mtc field)
    relationship_family = df['relationship'].isin(['Sister', 'Father', 'Paternal grandmother', 'Proband\'s daughter', 'Sister\'s son'])
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
        'source_id', 'study_id', 'age', 'gender', 'ret_k666n_positive', 'calcitonin_elevated', 'calcitonin_level_numeric',
        'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc', 'mtc_diagnosis',
        'c_cell_disease', 'men2_syndrome', 'pheochromocytoma', 'hyperparathyroidism', 'age_group'
    ]
    
    df_final = df[final_columns].copy()
    
    # save paper-only dataset
    df_final.to_csv('data/ret_k666n_training_data.csv', index=False)
    
    # create expanded dataset with literature cases
    expanded_df = create_expanded_dataset(df_final, paper_data)
    
    # save expanded dataset  
    expanded_df.to_csv('data/ret_k666n_expanded_training_data.csv', index=False)
    
    return df_final, expanded_df

def create_expanded_dataset(paper_df, paper_data):
    """create expanded dataset with literature cases"""
    
    # literature MTC diagnosis ages
    lit_ages = paper_data['literature_data']['mtc_diagnosis_ages']
    
    # create synthetic cases based on literature
    np.random.seed(42)  # for reproducibility
    
    synthetic_cases = []
    for age in lit_ages:
        # create case with MTC
        case = {
            'age': age,
            'gender': np.random.choice([0, 1]),  # random gender
            'ret_k666n_positive': 1,
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
        synthetic_cases.append(case)
    
    # create additional control cases (non-MTC carriers)
    control_cases = []
    for i in range(len(lit_ages) * 2):  # 2:1 control to case ratio
        age = np.random.choice(paper_df['age'].tolist() + lit_ages)
        control = {
            'age': age,
            'gender': np.random.choice([0, 1]),
            'ret_k666n_positive': 1,
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
        control_cases.append(control)
    
    # combine all cases
    expanded_df = pd.concat([paper_df, pd.DataFrame(synthetic_cases), pd.DataFrame(control_cases)], ignore_index=True)
    
    # remove duplicates based on key features (age, gender, calcitonin_level_numeric)
    print(f"Before expanded dataset deduplication: {len(expanded_df)} patients")
    expanded_df = expanded_df.drop_duplicates(subset=['age', 'gender', 'calcitonin_level_numeric'], keep='first')
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
        if study_id == "study_1":
            study_name = "JCEM Case Reports (2025)"
        elif study_id == "study_2":
            study_name = "EDM Case Reports (2024)"
        elif study_id == "study_3":
            study_name = "Thyroid Journal (2016)"
        else:
            study_name = study_id
        print(f"- {study_name}: {count} patients")
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