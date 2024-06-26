# Automated phenotyping

## Project description

Repository for the manuscript entitled "Comparative neural word embeddings approaches for medical concept representation and patient trajectory prediction".
- This project aims to compare NLP models (word2vec, fastTex and GloVe), based on the quality of their representation of medical concepts.
- We use MIMIC-IV to train the models, from which we extract patient trajectories as sequences of (amongst others) ICD10 and ATC codes.
- We train the models with model-specific NLP tasks that use patient trajectory sequences as input.
- We evaluate the models by producing medical concept embeddings, clustering them, comparing them to existing biomedical terminologies, and using them for medical outcome and patient trajectory prediction tasks.

## Installation requirements

1. Clone the repository:
   ```bash
   git clone git@github.com:ds4dh/medical_concept_representation.git  # or https://github.com/ds4dh/medical_concept_representation.git
   cd medical_concept_representation
   ```
2. Install the required dependencies:

   ```bash
   ./create_env.sh
   conda activate medical_representation
   ```

   If you have version issues, you can build an environment with the packages listed in environment.yml

3. The project uses WandbLogger for experiment tracking. Ensure you have a Weights & Biases account set up for logging.

## Usage

You need to download the data yourself! Instructions for downloading and pre-processing the data are here: https://github.com/ds4dh/medical_concept_representation/tree/main/data

Once the pre-processed data is ready, train the models with:

```bash
python run_all_models.py  # long step, best in screen https://linuxize.com/post/how-to-use-linux-screen/
```

Once the models are trained, test the trained models with:

```bash
python run_all_models.py -t  # long step, best in screen https://linuxize.com/post/how-to-use-linux-screen/
```

Result figures 4, 5, and 7 will be available at your wandb log page.

For the other result figures, run:
```bash
python figures/figure_6.py
python figures/figure_8.py
python figures/figure_8_bis.py   # supplementary figures
python figures/figure_9.py
```

