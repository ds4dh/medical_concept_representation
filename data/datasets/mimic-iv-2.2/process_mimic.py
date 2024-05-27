import os
import json
from tqdm import tqdm
from multiprocessing import Pool
from load_hosp_data import (
    load_admission_data,
    load_patient_data,
    load_location_data,
    load_diagnosis_data,
    load_procedure_data,
    load_medication_data,
    load_labevent_data,
)
from mimic_utils import (
    get_patient_data,
    get_admission_data,
    get_admission_labels,
    DIR_MIMIC_IV,
    PATIENT_PARAMS,
    ADMISSION_PARAMS,
    LOCATION_PARAMS,
    DIAGNOSIS_PARAMS,
    PROCEDURE_PARAMS,
    MEDICATION_PARAMS,
    LABEVENT_PARAMS,
)

OUTPUT_DIR = os.path.join(DIR_MIMIC_IV, 'datasets_full')
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
assert TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0 < 0.001
N_CPUS_USED = 16  # min(8, max(1, os.cpu_count() // 2))
DEBUG = False  # False
if DEBUG: OUTPUT_DIR += '_debug'


def main():
    # Load all relevant data
    data = {
        'patients': load_patient_data(os.path.join(DIR_MIMIC_IV, 'hosp')),
        'admissions': load_admission_data(os.path.join(DIR_MIMIC_IV, 'hosp')),
        'locations': load_location_data(os.path.join(DIR_MIMIC_IV, 'hosp')),
        'diagnoses': load_diagnosis_data(os.path.join(DIR_MIMIC_IV, 'hosp')),
        'procedures': load_procedure_data(os.path.join(DIR_MIMIC_IV, 'hosp')),
        'medications': load_medication_data(os.path.join(DIR_MIMIC_IV, 'hosp')),
        'labevents': load_labevent_data(os.path.join(DIR_MIMIC_IV, 'hosp')),
    }
    
    # Build splits based on patient ids
    subject_ids = data['patients'].subject_id.values
    train_index = int(len(subject_ids) * TRAIN_RATIO)
    valid_index = int(len(subject_ids) * (TRAIN_RATIO + VALID_RATIO))
    split_subject_id = {
        'train': subject_ids[:train_index],
        'valid': subject_ids[train_index:valid_index],
        'test': subject_ids[valid_index:]
    }
    
    # Generate all datasets based on patient data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in ['train', 'valid', 'test']:
        file_path = os.path.join(OUTPUT_DIR, '%s.json' % split)
        if os.path.exists(file_path): os.remove(file_path)
        iter_args = [(file_path, data, s) for s in split_subject_id[split]]
        if not DEBUG:
            print("Writing patient data (loop takes some time to start)")
            with Pool(processes=N_CPUS_USED) as pool:
                list(tqdm(
                    pool.imap(write_data_for_one_subject, iter_args, chunksize=1000),
                    total=len(iter_args),
                    desc='Building %s set' % split,
                    position=1,  # 0 (default) is not printed correctly in docker
                ))
            print("Finished writing patient data")
        else:  # run on a single CPU if using debug mode
            for arg in tqdm(iter_args, desc='Building %s set' % split):
                write_data_for_one_subject(arg)


def write_data_for_one_subject(args):
    # Parse arguments
    file_path, data, subject_id = args
    
    # Get all relevant data for one patient
    patient = get_patient_data(data, subject_id, **PATIENT_PARAMS)
    admissions = get_patient_data(data, subject_id, **ADMISSION_PARAMS)
    locations = get_patient_data(data, subject_id, **LOCATION_PARAMS)
    diagnoses = get_patient_data(data, subject_id, **DIAGNOSIS_PARAMS)
    procedures = get_patient_data(data, subject_id, **PROCEDURE_PARAMS)
    medications = get_patient_data(data, subject_id, **MEDICATION_PARAMS)
    labevents = get_patient_data(data, subject_id, **LABEVENT_PARAMS)
    
    # Generate sentence for each admission
    for admission_id in admissions.hadm_id:
        # Get all relevant data for one admission
        adm = get_admission_data(admissions, admission_id)
        loc = get_admission_data(locations, admission_id)
        dia = get_admission_data(diagnoses, admission_id)
        pro = get_admission_data(procedures, admission_id)
        med = get_admission_data(medications, admission_id)
        lab = get_admission_data(labevents, admission_id, {'flag': 'abnormal'})
        dem, lbl = get_admission_labels(patient, admissions, adm)
        
        # Build admission sentence (and sort using different time flags)
        t0 = adm.admittime.iloc[0]
        sub_tokens = ['SUB_%s' % subject_id, 'ADM_%s' % admission_id]
        dem_tokens = ['DEM_%s' % l for l in dem]
        lbl_tokens = ['LBL_%s' % l for l in lbl]
        dia_tokens = ['DIA_%s' % d for d in dia['icd_code'].values]
        loc_tokens = [(t - t0, 'LOC_%s' % word)
               for v, t in loc[['careunit', 'intime']].values for word in v]
        pro_tokens = [(t - t0, 'PRO_%s' % v)
               for v, t in pro[['icd_code', 'chartdate']].values]
        med_tokens = [(t - t0, 'MED_%s' % v) 
               for v, t in med[['gsn', 'starttime']].values]
        lab_tokens = [(t - t0, 'LAB_%s' % v)
               for v, t in lab[['itemid', 'charttime']].values]
        sorted_tokens = [s[1] for s in sorted(
            loc_tokens + pro_tokens + med_tokens + lab_tokens,
            key=lambda t: t[0])]
        
        # Build the patient sequence and append it to the correct json file
        seq = sub_tokens + lbl_tokens + dem_tokens + dia_tokens + sorted_tokens
        with open(file_path, 'a') as file:
            file.write(json.dumps(seq) + '\n')


if __name__ == '__main__':
    main()
