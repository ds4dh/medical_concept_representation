import os
import pandas as pd


def load_admission_data(path_mimic_iv_hosp_dir: str,
                        ) -> dict[str, pd.DataFrame]:
    """ Load data for patient admissions
    """
    print('Loading data for patient admissions')
    df_admissions = pd.read_csv(
        os.path.join(path_mimic_iv_hosp_dir, 'admissions.csv.gz'),
        usecols=[
            'subject_id',
            'hadm_id',
            'admission_location',  # useful? or use df_locations?
            'discharge_location',  # useful? or use df_locations?
            'race',
            'admittime',
            'deathtime',  # should use dod instead?
            'dischtime',
        ],
        dtype={
            'subject_id': pd.Int32Dtype(),
            'hadm_id': pd.Int32Dtype(),
            'admission_location': pd.StringDtype(),
            'discharge_location': pd.StringDtype(),
            'race': pd.StringDtype(),
        },
        parse_dates=['admittime', 'dischtime', 'deathtime'],
    )  # .sort_values(['subject_id', 'hadm_id'])
    return df_admissions


def load_patient_data(path_mimic_iv_hosp_dir: str,
                      ) -> dict[str, pd.DataFrame]:
    """ Load data for patient demographics
    """
    print('Loading data for patient demographics')
    df_patients = pd.read_csv(
        os.path.join(path_mimic_iv_hosp_dir, 'patients.csv.gz'),
        usecols=[
            'subject_id',
            'gender',
            'anchor_age',
            'anchor_year',
            'dod',  # should use deathtime or dod?
        ],
        dtype={
            'subject_id': pd.Int32Dtype(),
            'hadm_id': pd.Int32Dtype(),
            'gender': pd.StringDtype(),
            'anchor_age': pd.Int16Dtype(),
            'anchore_year': pd.Int16Dtype,
        },
        parse_dates=['dod'],
    )  # .sort_values(['subject_id', 'hadm_id'])
    return df_patients


def load_location_data(path_mimic_iv_hosp_dir: str,
                       ) -> dict[str, pd.DataFrame]:
    """ Load data for patient locations
    """
    print('Loading data for patient locations')
    df_locations = pd.read_csv(
        os.path.join(path_mimic_iv_hosp_dir, 'transfers.csv.gz'),
        usecols=[
            'subject_id',
            'hadm_id',
            'careunit',
            'eventtype',
            'intime',
        ],
        dtype={
            'subject_id': pd.Int32Dtype(),
            'hadm_id': pd.Int32Dtype(),
            'careunit': pd.StringDtype(),
        },
        parse_dates=['intime'],
    )
    return df_locations


def load_diagnosis_data(path_mimic_iv_hosp_dir: str,
                        ) -> dict[str, pd.DataFrame]:
    """ Load data for patient diagnoses
    """
    print('Loading data for diagnoses')
    df_diagnoses = pd.read_csv(
        os.path.join(path_mimic_iv_hosp_dir, 'diagnoses_icd.csv.gz'),
        usecols=[
            'subject_id',
            'hadm_id',
            'icd_code',
            'icd_version',
            'seq_num',  # useful for sorting?
        ],
        dtype={
            'subject_id': pd.Int32Dtype(),
            'hadm_id': pd.Int32Dtype(),
            'icd_code': pd.StringDtype(),
            'icd_version': pd.Int16Dtype(),
            'seq_num': pd.Int16Dtype(),
        },
    )  # .sort_values(['subject_id', 'hadm_id'])
    return df_diagnoses


def load_procedure_data(path_mimic_iv_hosp_dir: str,
                        ) -> dict[str, pd.DataFrame]:
    """ Load data for patient procedures
    """
    # Hosp data for procedure to icd conversion
    print('Loading data for procedures')
    df_procedures = pd.read_csv(
        os.path.join(path_mimic_iv_hosp_dir, 'procedures_icd.csv.gz'),
        usecols=[
            'subject_id',
            'hadm_id',
            'icd_code',
            'icd_version',
            'seq_num',
            'chartdate',
        ],
        dtype={
            'subject_id': pd.Int32Dtype(),
            'hadm_id': pd.Int32Dtype(),
            'icd_code': pd.StringDtype(),
            'icd_version': pd.Int16Dtype(),
            'seq_num': pd.Int16Dtype(),
        },
        parse_dates=['chartdate'],
    )  # .sort_values(['subject_id', 'hadm_id'])
    return df_procedures


def load_medication_data(path_mimic_iv_hosp_dir: str,
                         ) -> dict[str, pd.DataFrame]:
    """ Load data for patient medications
    """
    # Hosp data for prescriptions (this step takes about 1 minute)
    print('Loading data for medication (takes about 1 minute)')
    df_prescription_chunks = pd.read_csv(
        os.path.join(path_mimic_iv_hosp_dir, 'prescriptions.csv.gz'),
        usecols=[
            'subject_id',
            'hadm_id',
            'ndc',
            'gsn',
            'starttime',
        ],
        dtype={
            'subject_id': pd.Int32Dtype(),
            'hadm_id': pd.Int32Dtype(),
            'ndc': pd.StringDtype(),
            'gsn': pd.StringDtype(),
        },
        parse_dates=['starttime'],
        chunksize=1000000,
        iterator=True,
    )
    df_prescriptions = pd.concat(df_prescription_chunks, ignore_index=True)
    df_prescriptions = df_prescriptions.replace({'ndc': {'0': pd.NA}})
    # df_prescriptions = df_prescriptions.sort_values(['subject_id', 'hadm_id'])
    return df_prescriptions


def load_labevent_data(path_mimic_iv_hosp_dir: str,
                       ) -> dict[str, pd.DataFrame]:
    """ Load data for lab events
    """
    # Hosp data for lab events (this step takes about 10 minutes)
    print('Loading data for lab events (takes about 10 minute)')
    df_labevent_chunks = pd.read_csv(
        os.path.join(path_mimic_iv_hosp_dir, 'labevents.csv'),
        usecols=[
            'subject_id',
            'hadm_id',
            'itemid',
            'flag',
            'charttime',
        ],
        dtype={
            'subject_id': pd.Int32Dtype(),
            'hadm_id': pd.Int32Dtype(),
            'flag': pd.StringDtype(),
        },
        parse_dates=['charttime'],
        chunksize=1000000,
        iterator=True,
    )
    df_labevents = pd.concat(df_labevent_chunks, ignore_index=True)
    # df_labevents = df_labevents.sort_values(['subject_id', 'hadm_id'])
    return df_labevents
