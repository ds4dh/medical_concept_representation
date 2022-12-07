source ~/miniconda3/etc/profile.d/conda.sh  # to use conda from .sh script
conda create -y -n torch_autophe
conda activate torch_autophe
conda install -y pytorch torchdata pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -y pytorch-lightning scikit-learn pandas matplotlib nltk toml tqdm -c conda-forge -y
pip install gradient-descent-the-ultimate-optimizer
conda env export > environment.yml