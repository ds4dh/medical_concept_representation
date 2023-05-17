The raw data files are not available in this repository. To access it, go to https://physionet.org/content/mimiciv/2.2/. Only credentialed users who sign the DUA can access the files.

Once you have access to the files, you should upload them at the following locations of your local repository:
- data/datasets/mimic-iv-2.2/hosp/admissions.csv.gz
- data/datasets/mimic-iv-2.2/hosp/patients.csv.gz
- data/datasets/mimic-iv-2.2/hosp/transfers.csv.gz
- data/datasets/mimic-iv-2.2/hosp/diagnoses_icd.csv.gz
- data/datasets/mimic-iv-2.2/hosp/procedures_icd.csv.gz
- data/datasets/mimic-iv-2.2/hosp/prescriptions.csv.gz
- data/datasets/mimic-iv-2.2/hosp/labevents.csv.gz

You can then build the pre-processed datasets, using the following command:
```
python data/datasets/mimic-iv-2.2/process_mimic.py```
