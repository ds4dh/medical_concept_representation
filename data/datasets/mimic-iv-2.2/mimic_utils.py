import os
import numpy as np
import pandas as pd
from typing import Union, Callable


# Useful variables
DIR_MIMIC_IV = os.path.abspath(os.path.join('data', 'datasets', 'mimic-iv-2.2'))
MAP_DIR = os.path.join(DIR_MIMIC_IV, 'maps')
ZERO_DAYS = np.timedelta64(0, 'D')
SEVEN_DAYS = np.timedelta64(7, 'D')
THIRTY_DAYS = np.timedelta64(30, 'D')

# All mappings to transform mimic-iv data entries into computer-readable tokens
DIAPRO_MAP = pd.read_csv(
    os.path.join(MAP_DIR, 'icd10_to_icd9.csv'),
    usecols=['icd9', 'icd10'],
    dtype={'icd9': pd.StringDtype(), 'icd10': pd.StringDtype()}
)
DIA_MAP = pd.read_csv(
    os.path.join(MAP_DIR, 'icd9_to_icd10_cm.csv'),
    usecols=['icd9cm', 'icd10cm'],
    dtype={'icd9cm': pd.StringDtype(), 'icd10cm': pd.StringDtype()}
)
PRO_MAP = pd.read_csv(
    os.path.join(MAP_DIR, 'icd9_to_icd10_pcs.csv'),
    usecols=['icd9pcs', 'icd10pcs'],
    dtype={'icd9pcs': pd.StringDtype(), 'icd10pcs': pd.StringDtype()}
)
ETH_MAP = pd.read_csv(
    os.path.join(MAP_DIR, 'ethnicity.csv'),
    usecols=['race', 'racelabel'],
    dtype={'race': pd.StringDtype(), 'race': pd.StringDtype()}
)
LOC_MAP = pd.read_csv(
    os.path.join(MAP_DIR, 'location.csv'),
    usecols=['location', 'locationlabel'],
    dtype={'location': pd.StringDtype(), 'locationlabel': pd.StringDtype()}
)
MED_MAP = pd.read_csv(
    os.path.join(MAP_DIR, 'gsn_to_atc_to_ndc.csv'),
    usecols=['gsn', 'ndc', 'atc'],
    dtype={
        'gsn': pd.StringDtype(),
        'ndc': pd.StringDtype(),
        'atc': pd.StringDtype()
    }
)
LAB_MAP = pd.read_csv(
    os.path.join(DIR_MIMIC_IV, 'hosp', 'd_labitems.csv.gz'),
    usecols=['itemid', 'label'],
    dtype={'itemid': pd.StringDtype(), 'label': pd.StringDtype()},
)

# Transform map dataframes into useable dictionaries
ETH_MAP = dict(zip(ETH_MAP.race, ETH_MAP.racelabel))
LOC_MAP = dict(zip(LOC_MAP.location, LOC_MAP.locationlabel))
DIAPRO_MAP = dict(zip(DIAPRO_MAP.icd9, DIAPRO_MAP.icd10))
DIA_MAP['temp'] = DIA_MAP.loc[:, 'icd9cm']
DIA_MAP.temp[DIA_MAP.icd10cm == 'NoDx'] =\
    DIA_MAP.temp[DIA_MAP.icd10cm == 'NoDx'].replace(DIAPRO_MAP)
DIA_MAP.icd10cm[DIA_MAP.icd10cm == 'NoDx'] =\
    DIA_MAP.pop('temp')[DIA_MAP.icd10cm == 'NoDx']
DIA_MAP = dict(zip(DIA_MAP.icd9cm, DIA_MAP.icd10cm))
PRO_MAP['temp'] = PRO_MAP.loc[:, 'icd9pcs']
PRO_MAP.temp[PRO_MAP.icd10pcs == 'NoPCS'] =\
    PRO_MAP.temp[PRO_MAP.icd10pcs == 'NoPCS'].replace(DIAPRO_MAP)
PRO_MAP.icd10pcs[PRO_MAP.icd10pcs == 'NoPCS'] =\
    PRO_MAP.pop('temp')[PRO_MAP.icd10pcs == 'NoPCS']
PRO_MAP = dict(zip(PRO_MAP.icd9pcs, PRO_MAP.icd10pcs))
MED_MAP['ndc'] = MED_MAP['ndc'].apply(lambda s: s.rjust(11, '0'))
MED_MAP_G2A = dict(zip(MED_MAP.gsn, MED_MAP.atc))
MED_MAP_N2A = dict(zip(MED_MAP.ndc, MED_MAP.atc))
LAB_MAP = dict(zip(LAB_MAP.itemid, LAB_MAP.label))

# Parameters to read mimic-iv dataset, for each type of tokens
ADMISSION_PARAMS = {
    'data_key': 'admissions',
    'fields_to_sort': ['admittime'],
    'fields_to_get': [
        'hadm_id',
        'admittime',
        'dischtime',
        'deathtime',
        'admission_location',
        'race'
    ],
    'mappings': {'race': ETH_MAP},
}
PATIENT_PARAMS = {
    'data_key': 'patients',
    'fields_to_get': ['gender', 'anchor_age', 'anchor_year', 'dod'],
}
LOCATION_PARAMS = {
    'data_key': 'locations',
    'fields_to_get': ['hadm_id', 'careunit', 'intime'],
    'mappings': {'careunit': LOC_MAP},
    'applies': {'careunit': lambda s: s.split('-')}
}
DIAGNOSIS_PARAMS = {
    'data_key': 'diagnoses',
    'fields_to_get': ['hadm_id', 'icd_code'],
    'fields_to_sort': ['hadm_id', 'seq_num'],
    # 'fields_to_unify': ['icd_code'],
    'mappings': {'icd_code': DIA_MAP},
    'selects': {'icd_code': ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                             'V', 'W', 'X', 'Y', 'Z', 'U')},
}
PROCEDURE_PARAMS = {
    'data_key': 'procedures',
    'fields_to_get': ['hadm_id', 'icd_code', 'chartdate'],
    # 'fields_to_unify': ['icd_code'],
    'mappings': {'icd_code': PRO_MAP},
    'selects': {'icd_code': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                             'B', 'C', 'D', 'F', 'G', 'H', 'X')},
}
MEDICATION_PARAMS = {
    'data_key': 'medications',
    'fields_to_get': ['hadm_id', 'gsn', 'ndc', 'starttime'],
    # 'fields_to_unify': ['gsn'],  # actually atc but we don't update column name
    'fields_to_dropna': [{'gsn': 'ndc'}],
    'mappings': {'gsn': MED_MAP_G2A, 'ndc': MED_MAP_N2A},
    'selects': {'gsn': ('A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P',
                        'R', 'S', 'V')},
}
LABEVENT_PARAMS = {
    'data_key': 'labevents',
    'fields_to_get': ['hadm_id', 'itemid', 'flag', 'charttime'],
    # 'fields_to_unify': ['itemid'],
}


def get_patient_data(data: dict[pd.DataFrame],
                     subject_id: int,
                     data_key: str,
                     fields_to_get: list[str],
                     fields_to_sort: list[str]=[],
                     fields_to_unify: list[str]=[],
                     fields_to_dropna: list[Union[str, dict[str, str]]]=[],
                     mappings: dict[str, dict[str, str]]={},
                     applies: dict[str, Callable]={},
                     selects: dict[str, tuple[str]]={},
                     ) -> pd.DataFrame:
    """ Get specific data for a given subject
    """
    # Get subject data for a given data key
    specific_data = data[data_key]
    subject_bool = (specific_data.subject_id == subject_id)
    subject_data = specific_data[subject_bool]

    # Sort rows according to some fields (one by one)
    if len(fields_to_sort) > 0:
        subject_data = subject_data.sort_values(fields_to_sort)
    
    # Remap values if required (before or after NaNs replacement???)
    subject_fields = subject_data[fields_to_get]

    if data_key == 'locations':
        import pdb; pdb.set_trace()
        
    subject_fields = subject_fields.replace(mappings)
    
    # Handle NaNs if required (if field is a dict, replace NaN key by value)
    for field in fields_to_dropna:
        if isinstance(field, dict):  # there must be a cleaner way!
            assert len(field) == 1
            replacement = subject_fields.pop(list(field.values())[0])
            subject_fields[field.keys()].fillna(replacement, inplace=True)
        subject_fields.dropna(subset=field, inplace=True)
        
    # Remove duplicates while preserving order
    for field in fields_to_unify:
        subject_fields.drop_duplicates(subset=field, keep='first', inplace=True)
    
    # Filter out rows that do not start with any of the given strings
    for key, keep_starts in selects.items():
        filter_bool = subject_fields[key].str.startswith(keep_starts)
        subject_fields = subject_fields[filter_bool]
        
    # Apply a function if required
    for key, fn in applies.items():
        subject_fields[key] = subject_fields[key].apply(fn)
    
    # Return required subject fields
    return subject_fields


def get_admission_data(patient_data: pd.DataFrame,
                       admission_id: int,
                       additional_match_dict: dict={},
                       ) -> pd.DataFrame:
    """ Get specific data for a given patient's admission
    """
    admission_data = patient_data[patient_data.hadm_id == admission_id]
    for k, v in additional_match_dict.items():
        admission_data = admission_data[admission_data[k] == v]
    return admission_data


def get_admission_labels(patient: pd.DataFrame,
                         admissions: pd.DataFrame,
                         admission: pd.DataFrame,
                         ) -> list[str]:
    """ Initialize an admission sentence for a given patient with prediction
        labels and demographic information
    """
    mortality_label = find_mortality_label(admission, patient)
    readmission_label = find_readmission_label(admission, admissions)
    length_of_stay_label = find_length_of_stay_label(admission)
    gender_label = find_gender_label(patient)
    ethnicity_label = find_ethnicity_label(admission)
    age_label = find_age_label(admission, patient)
    return (gender_label + ethnicity_label + age_label,
            mortality_label + readmission_label + length_of_stay_label)
           
           
def find_gender_label(patient: pd.DataFrame) -> list:
    """ Indentify the gender of a patient
    """
    return [patient.gender.values[0]]


def find_age_label(admission: pd.DataFrame,
                   patient: pd.DataFrame
                   ) -> list:
    """ Identify the age label of a patient
    """
    age = admission.admittime.dt.year
    age -= patient.anchor_year.values
    age += patient.anchor_age.values
    age = age.values[0]
    assert age > 0, 'Age %s is not valid' % age
    if age < 21:
        return ['YOUNG']
    elif age < 50:
        return ['ADULT']
    elif age < 65:
        return ['MIDDLE']
    else:
        return ['ELDER']
    
    
def find_ethnicity_label(admission: pd.DataFrame) -> list:
    """ Identify the ethnicity label of an admission's patient
    """
    return admission.race.values[0].split('-')


def find_mortality_label(admission: pd.DataFrame,
                         patient: pd.DataFrame
                         ) -> list:
    """ Identify whether a patient passed aways during an admission or less than
        seven days after admission discharge time
    """
    if not any(admission.deathtime.isnull()):
        return ['DEAD']
    relative_dod = patient.dod.values - admission.dischtime.values
    if relative_dod.astype('timedelta64[D]') < THIRTY_DAYS:
        return ['DEAD']
    return ['ALIVE']


def find_readmission_label(admission: pd.DataFrame,
                           admissions: pd.DataFrame
                           )-> list:
    """ Identify whether a patient was readmitted less than thirty days after
        the discharge time of a given admission
    """
    time_differences = admissions.admittime.values - admission.dischtime.values
    if any([ZERO_DAYS < t < THIRTY_DAYS for t in time_differences]):
        return ['READM']
    else:
        return ['AWAY']  # there must be a better word


def find_length_of_stay_label(admission: pd.DataFrame) -> list:
    """ Identify whether a patient's admission lasted more than 7 days
    """
    duration = admission.dischtime.values - admission.admittime.values
    if duration > SEVEN_DAYS:
        return ['LONG']
    else:
        return ['SHORT']
    