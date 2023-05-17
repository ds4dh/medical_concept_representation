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

    
    # # Hosp data for diagnose codes description
    # print('Loading data for diagnose codes description')
    # df_diagnoses_descr = pd.read_csv(
    #     os.path.join(path_mimic_iv_hosp_dir, 'd_icd_diagnoses.csv.gz'),
    #     dtype={
    #         'icd_code': pd.StringDtype(),
    #         'icd_version': 'int8',
    #     }
    # )
    
    # # Hosp data for procedure descriptions
    # print('Loading data for procedure descriptions')
    # df_procedures_descr = pd.read_csv(
    #     os.path.join(path_mimic_iv_hosp_dir, 'd_icd_procedures.csv.gz'),
    #     usecols=['icd_code', 'icd_version'],
    #     dtype={
    #         'icd_code': pd.StringDtype(),
    #         'icd_version': 'int8',
    #     }
    # )
    
    # # Hosp data for labitems descriptions
    # print('Loading data for labitems descriptions')
    # df_labitems_descr  = pd.read_csv(
    #     os.path.join(path_mimic_iv_hosp_dir, 'd_labitems.csv.gz'),
    #     dtype={
    #         'label': pd.StringDtype(),
    #         'fluid': pd.StringDtype(),
    #         'category': pd.StringDtype(),
    #         'loinc_code': pd.StringDtype()
    #     }
    # )

    # # Hosp data for lab hcpc events
    # df_hcpcsevents = pd.read_csv(
    #     os.path.join(path_mimic_iv_hosp_dir, 'hcpcsevents.csv.gz'),
    #     parse_dates=['chartdate'],
    #     dtype={
    #         'hcpcs_cd': pd.StringDtype(),
    #         'short_description': pd.StringDtype(),
    #         'seq_num': 'uint8'
    #     }
    # ).sort_values(['subject_id', 'chartdate', 'seq_num'])

    # # Hosp data for hcpcs description
    # df_d_hcpcs = pd.read_csv(
    #     os.path.join(path_mimic_iv_hosp_dir, 'd_hcpcs.csv.gz'),
    #     usecols=['code', 'short_description'],
    #     dtype={
    #         'code':pd.StringDtype(),
    #         'short_description': pd.StringDtype()
    #     }
    # )

    # # Hosp data for diagnosis-related group
    # df_drgcodes = pd.read_csv(
    #     os.path.join(path_mimic_iv_hosp_dir, 'drgcodes.csv.gz'),
    #     dtype={
    #         'drg_type': pd.StringDtype(),
    #         'description': pd.StringDtype(),
    #         'drg_code': 'int16'
    #     }
    # ).sort_values(
    #     ['subject_id', 'hadm_id', 'drg_severity'],
    #     ascending=[True, True, False]
    # )

    # # Hosp data for emar
    # df_emar = pd.read_csv(
    #     os.path.join(path_mimic_iv_hosp_dir, 'emar.csv.gz'),
    #     parse_dates=['charttime', 'scheduletime', 'storetime'],
    #     dtype={
    #         'hadm_id': 'Int64',
    #         'emar_id': pd.StringDtype(),
    #         'poe_id': pd.StringDtype(),
    #         'pharmacy_id': 'Int64',
    #         'medication': pd.StringDtype(), 
    #         'event_txt': pd.StringDtype()
    #     }
    # ).sort_values(['subject_id', 'charttime'])

    # # Hosp data for micro-biology events
    # df_microbiologyevents = pd.read_csv(
    #     os.path.join(path_mimic_iv_hosp_dir, 'microbiologyevents.csv.gz'),
    #     parse_dates=['chartdate', 'charttime', 'storedate', 'storetime'],
    #     dtype={
    #         'comments': pd.StringDtype(),
    #         'hadm_id': 'Int64',
    #         'spec_type_desc': pd.StringDtype(),
    #         'isolate_num': 'Int16',
    #         'quantity': pd.StringDtype(),
    #         'test_name': pd.StringDtype(),
    #         'ab_name': pd.StringDtype(),
    #         'ab_itemid': 'Int32',
    #         'dilution_text': pd.StringDtype(),
    #         'dilution_comparison': pd.StringDtype(),
    #         'org_name': pd.StringDtype(),
    #         'interpretation': pd.StringDtype(),
    #         'org_itemid': 'Int32'
    #     }
    # ).sort_values(['subject_id', 'charttime'])

    # # Hosp data for services
    # df_services = pd.read_csv(
    #     os.path.join(path_mimic_iv_hosp_dir, 'services.csv.gz'),
    #     parse_dates=['transfertime'],
    #     dtype={
    #         'prev_service': pd.StringDtype(),
    #         'curr_service': pd.StringDtype()
    #     }
    # ).sort_values(['subject_id', 'transfertime'])


# def update_hosp_data(hosp_data):
#     df_icd10_diag = pd.read_csv(
#         '../data/mimiciv_compressed/1.0/hosp/mimiciv_icd10_diagnoses.csv',
#         header=None,
#         names=['icd10']
#     )
    
#     hosp_data['diagnoses_icd_to_gsn']['icd10'] = df_icd10_diag.iloc[hosp_data['diagnoses_icd_to_gsn'].index].icd10.values
#     icd10_reduced = np.array([icd[:4] for icd in hosp_data['diagnoses_icd_to_gsn'].icd10.values])
#     hosp_data['diagnoses_icd_to_gsn']['icd10_reduced'] = icd10_reduced

#     df_icd10_proc = pd.read_csv(
#         '../data/mimiciv_compressed/1.0/hosp/mimiciv_icd10_procedures.csv',
#         header=None,
#         names=['icd10'])
#     hosp_data['procedures_icd']['icd10'] = df_icd10_proc.iloc[hosp_data['procedures_icd'].index].icd10.values

#     return hosp_data


# def get_na_to_ndc(hosp_data):
#     df_ndc_prod = pd.read_csv(
#         '../data/product.txt',
#         sep='\t',
#         usecols=[
#             'PRODUCTNDC',
#             'PROPRIETARYNAME',
#             'NONPROPRIETARYNAME',
#             'ACTIVE_NUMERATOR_STRENGTH',
#             'DOSAGEFORMNAME'
#         ],
#         dtype = {
#             'PRODUCTNDC': pd.StringDtype(),
#             'PROPRIETARYNAME': pd.StringDtype(),
#             'NONPROPRIETARYNAME': pd.StringDtype(),
#             'ACTIVE_NUMERATOR_STRENGTH': pd.StringDtype(),
#             'DOSAGEFORMNAME': pd.StringDtype()
#         }
#     )
#     df_ndc_prod['PRODUCTNDC'] = df_ndc_prod.PRODUCTNDC.str.replace('-', '', regex=False)

#     drugs_na = sorted(hosp_data['prescriptions'][hosp_data['prescriptions'].ndc == '0'].drug.unique())
#     na_to_ndc = np.array([[drug, 'UNK'] for drug in drugs_na], dtype=str)
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Venetoc')!=-1),1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Venetoc')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Sodium Chloride ')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Sodium Chl')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], '0.9% S')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Sodium Chl')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], '0.9 % S')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Sodium Chl')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], ' 0.9%')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Sodium Chl')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], '0.83% S')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Sodium Chl')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Esmolol')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Esmolol')].PRODUCTNDC.values[3]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], '1/2')!=-1), 1] = '02647802'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], '5% Dextrose')!=-1), 1] = '02647510'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Dextrose 5%')!=-1), 1] = '02647510'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], '10% Dextrose')!=-1), 1] = '63323824'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], '3,4-Diaminopyridine')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('amifampridine')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], '(NS)')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Sodium Chl')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'NS ')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Sodium Chl')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ABT')!=-1),1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Venetoc')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'AMP')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains(' amphetamine ')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'bira')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Abira')].PRODUCTNDC.values[1]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Acetaminophen')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Acetaminophen')].PRODUCTNDC.values[-2]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Alectinib')!=-1), 1] = '50242130'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Alisertib')!=-1), 1] = 'MLN8237'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Ambien')!=-1), 1] = df_ndc_prod[df_ndc_prod.PROPRIETARYNAME.str.contains('Ambien')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Anamorelin')!=-1), 1] = '672620006'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'zimilide')!=-1), 1] = 'NE10064'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'BD')!=-1), 1] = '82903201'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'BLY719')!=-1), 1] = '00780701'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'BYL719')!=-1), 1] = '00780701'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'oceprevir')!=-1), 1] = ' 00850314'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'rincidofovir')!=-1), 1] = '7962201265'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Bupivacaine 0.2')!=-1), 1] = '55150167'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Bupivacaine 0.1')!=-1), 1] = '69374970'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Bupivacaine 0.05')!=-1), 1] = '01439329'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Bupivacaine 0.025')!=-1), 1] = '01439328'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'CRLX101')!=-1), 1] = 'NLG207'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Caffeine')!=-1), 1] = '05172502'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'alcium')!=-1), 1] = '63323360'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Cangrelor')!=-1), 1] = '10122620'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Cem 102 300mg tab')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Fusidate')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'lopidogrel')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Clopidogrel')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ortro')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('ortro')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'otrosyn')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('ortro')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'rizotinib')!=-1), 1] = '73309090'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'osyntropin')!=-1), 1] = df_ndc_prod[df_ndc_prod.PROPRIETARYNAME.str.contains('Cosyntropin')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'D5')!=-1), 1] = '02647510'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'D7')!=-1), 1] = '02647510'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'D1')!=-1), 1] = '63323824'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'D2')!=-1), 1] = '63323824'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'D3')!=-1), 1] = '63323824'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'acomitinib')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('acomitinib')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Daliresp')!=-1), 1] = df_ndc_prod[df_ndc_prod.PROPRIETARYNAME.str.contains('aliresp')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Defibrotide')!=-1), 1] = '68727800'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Domperidone')!=-1), 1] = '17033326'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'onepezil')!=-1), 1] = '31722737'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'doxaban')!=-1), 1] = '65597201'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'mricasan')!=-1), 1] = '02960204'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ntremed')!=-1), 1] = 'ENMD2076'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'peleuton')!=-1), 1] = '04365400'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Erlotinib')!=-1), 1] = '00937663'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Etravirine')!=-1), 1] = '60219172'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Everolimus')!=-1), 1] = '00540470'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'usidic')!=-1), 1] = '734420001'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Fycompa')!=-1), 1] = '62856272'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'GILTERITINIB')!=-1), 1] = '04691425'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ilteri')!=-1), 1] = '04691425'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'liter')!=-1), 1] = '04691425'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'patitis')!=-1), 1] = '00064981'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'umu')!=-1), 1] = '00028501'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Hydrocortisone')!=-1), 1] = '01130541'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'uxolitin')!=-1), 1] = '50881005'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ABL001')!=-1), 1] = '00781091'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ALKS')!=-1), 1] = '7ZX1Q9SJ1F'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ASP1235')!=-1), 1] = '37I4D4Q7S2'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'nakinra')!=-1), 1] = '66658234'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Artesunate')!=-1), 1] = '73607011'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'BL-8040')!=-1), 1] = 'DA9G065962'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Bexanolone')!=-1), 1] = '72152547'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Cethrin')!=-1), 1] = 'B36V5S1RFJ'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'DAS181')!=-1), 1] = '227R1C272Q'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'enintuzuma')!=-1), 1] = 'H5324S1M7H'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ribulin')!=-1), 1] = '6285638901'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'GO-203')!=-1), 1] = '5YSY733NA3'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'H3B-880')!=-1), 1] = '90YLS47BRX'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'tacitinib')!=-1), 1] = '19J3781LPM'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'KRN-7000')!=-1), 1] = 'WX671898JF'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'etermovir')!=-1), 1] = '0006307501'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'aribavir')!=-1), 1] = '000630750,'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'avorixafor')!=-1), 1] = '0G9LGB5O2W'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Mepolizumab')!=-1), 1] = '01730881'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'staurin')!=-1), 1] = '0078069802'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'imodipine')!=-1), 1] = '23155512'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'OTL-38')!=-1), 1] = 'F7BD3Z4X8L'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Opiranserin')!=-1), 1] = 'AP031EC2NI'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'PEN-221')!=-1), 1] = 'L8993M383E'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ancrelipase')!=-1), 1] = '00321203'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'egvorhyaluronidase')!=-1), 1] = 'P01I4980ZS'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'INV-REGN1979')!=-1), 1] = '8R5CM46UIO'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Ruxolitnib')!=-1), 1] = '50881005'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'elonsertib')!=-1), 1] = 'NS3988A2TC'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'TISSEEL')!=-1), 1] = '03388402'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'erlipres')!=-1), 1] = '635860124'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Tivantinib')!=-1), 1] = 'PJ4H73IL17'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'alganciclovir')!=-1), 1] = '00040038'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Idelalisib')!=-1), 1] = '619581701'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'nsulin')!=-1), 1] = '00027510'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Iso-Osmotic Dextrose')!=-1), 1] = '03381301'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Iso-Osmotic Sodium Chloride')!=-1), 1] = '52533006712'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Isotonic Sodium Chloride')!=-1), 1] = '02642201'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Ket')!=-1), 1] = '01439508'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ketamine')!=-1), 1] = '01439508'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'tacrolimus')!=-1), 1] = '01680416'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'LaMICtal')!=-1), 1] = '01730754'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Lamictal')!=-1), 1] = '01730754'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Lactulose')!=-1), 1] = '45963438'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'edipasvir')!=-1), 1] = '6195818011'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Lipiodol')!=-1), 1] = '6768419011'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'xycodo')!=-1), 1] = '04060552'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'aalox')!=-1), 1] = '5486820930'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'orphine')!=-1), 1] = '40032540'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ylcysteine')!=-1), 1] = '05177504'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'NS')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Sodium Chl')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Niraparib')!=-1), 1] = '69656103'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Normal Saline Flush')!=-1), 1] = '00191188'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ovolog')!=-1), 1] = '01692001'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Nozin')!=-1), 1] = '02959025'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Ondansetron')!=-1), 1] = '04094755'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'rnithin')!=-1), 1] = '515935008'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Oseltamivir')!=-1), 1] = '00040800'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'PT-2977')!=-1), 1] = '00065331'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Papain')!=-1), 1] = '449110462'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Peramivir')!=-1), 1] = '72769181'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'erampanel')!=-1), 1] = '62856272'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'PrismaSol')!=-1), 1] = '24571101'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Profend')!=-1), 1] = '108193888'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Ramipril')!=-1), 1] = '00540106'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Relistor')!=-1), 1] = '65649150'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ibavirin')!=-1), 1] = '42494423'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Sodium Bicarbonate')!=-1), 1] = '71610438'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Sodium CI')!=-1), 1] = '1453788125'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Sodium Ci')!=-1), 1] = '1453788125'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Sofosbuvir')!=-1), 1] = '6195818011'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Sodium Chloride 3% (Hypertonic)')!=-1), 1] = '03380054'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Sodium Chloride 0.45% Flush')!=-1), 1] = '02647802'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Sodium Chloride 0.225 %')!=-1), 1] = '02641800'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Sterile')!=-1), 1] = '04094887'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Syringe (0.45% Sodium Chloride)')!=-1), 1] = '02647802'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Telaprevir')!=-1), 1] = '5116710001'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Tenecteplase')!=-1), 1] = '5024203706'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Thiami')!=-1), 1] = '06416228'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Tofacitinib')!=-1), 1] = '00690501'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Tolvaptan')!=-1), 1] = '31722869'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Trisodium')!=-1), 1] = '237316030'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Umbralisib')!=-1), 1] = '73150200'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Uridine triacetate')!=-1), 1] = '69468151'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'oxtalisib')!=-1), 1] = 'CVL1685GPH'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Yuvafem')!=-1), 1] = '42291962'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Pimasertib')!=-1), 1] = '6ON9RK82AL'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'Zanamivir')!=-1), 1] = '01730681'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'acyclovir')!=-1), 1] = '76420050'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'cortosyn')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('ortro')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'diph')!=-1), 1] = '6906709204'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'lamoTRIgine')!=-1), 1] = '2169522330'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'maa')!=-1), 1] = '5486820930'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'ganciclovir')!=-1), 1] = '6332331501'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'morp')!=-1), 1] = '00236011'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'roflumilast')!=-1), 1] = '03100088'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'sacubitril')!=-1), 1] = '00780659'
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'sodium ch')!=-1), 1] = df_ndc_prod[df_ndc_prod.NONPROPRIETARYNAME.str.contains('Sodium Chl')].PRODUCTNDC.values[0]
#     na_to_ndc[np.flatnonzero(np.core.defchararray.find(na_to_ndc[:,0], 'sodium citrate')!=-1), 1] = '1453788125'
    