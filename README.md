# MonoNet

Code release for the submission "MonoNet: towards interpretable models by learning monotonic features"

## Setup instructions

The preferred way to use this code is by creating a conda environment. E.g.

```bash
conda create --name mononet python=3.8
conda activate mononet
```

You can install this repository by running, from the repository root, (the editable mode flag `-e` is optional)

```bash
pip install -e .
```

## Summary of the scripts

### Reproduce the paper results/figures

We are in the processing of cleaning and generalizing the code. 
The code used to run produce the figures is in a "boilerplate" state, but hopefully
is encapsulated enough for reproducibility. 
We provide also the pretrained model we used for analysis at 
`experimental_code/tabular_data/nn_13_04_acc_91_saved.pickle`.
The data for the single cell experiments can be found in the folder
`experimental_code/tabular_data/`.

```bash
python experimental_code/tabular_data/main_MonoNet.py --data_root <path_to_FOLDER_with_data> --save_root <path_where_to_store_results> --in_file_NN <path_to_the_trained_model>
```

To this command, one or more of the following flags should be added:

```bash
  --compute_shap_values             To produce the shap values figure
  --analysis_with_violin_plots      To produce the statistical analysis (of the unconstrained block) figures
  --information_analysis            To run the information theoretical analysis
  --analysis_clustering             Run also the clustering-based analysis, within the info-theoretical analysis
```

For the vision examples, you can run the script ``experimental_code/vision_data/train_medmnist.py``. The available command line options are

```bash
train_medmnist.py [OPTIONS]

Options:
  --benchmark           If to perform benchmark over all datasets
  --data_flag TEXT      Dataset to benchmark, If the benchmark flag is not set. Available datasets:
                        ['pathmnist', 'octmnist', 'pneumoniamnist',
                        'dermamnist', 'breastmnist', 'bloodmnist',
                        'tissuemnist', 'organamnist', 'organcmnist',
                        'organsmnist']
  --results_path TEXT   Where to store/retrieve the results and the model
  --data_path TEXT      Where to store/retrieve the dataset  [required]
  --model_name TEXT     Model to train
  --n_epochs INTEGER    Number of training_epochs
  --lr FLOAT            Learning rate
  --batch_size INTEGER  Batch size
  --optimizer TEXT      Optimizer to use
  --help                Show this message and exit.
```

To extract the explanations from the monotonic CNN, you can refer to the script ``experimental_code/vision_data/interpret_monotonic_cnn.py``

```bash
Usage: interpret_monotonic_cnn.py [OPTIONS]

Options:
  --model_name TEXT    Model to interpret
  --data_path TEXT     Path where to store/load from the data
  --data_flag TEXT     Dataset to interpret
  --model_path TEXT    Path to the model checkpoint  [required]
  --data_idx INTEGER   Index of the datapoint to interpret
  --results_path TEXT  Where to store the interpretability results  [required]
  --help               Show this message and exit.
```


### Building/Training a MonoNet

You can find the building blocks of a MonoNet in `mononet/mononet.py`. 
These layers should be used as basic components of a MonoNet.
Examples of MonoNet and training scripts in different domains can be found at `mononet/monotonic_cnn.py` 
or `mononet/monotonic_mlp.py`.
For example, to train a MonoNet on the Single Cell data, you can run, from the repository root:

```bash
python bin/train_sc.py --data_path experimental_code/tabular_data --results_path <where_to_store_results>

```