# Experiments
## Requirements

The following setup was tested on a Ubuntu 18.04.5 LTS system with CUDA Version: 10.2:

```
conda create --name jmvamom python=3.8 && conda activate jmvamom
pip install -r requirements.txt
```

## Defining Experiment
In order to define parameters for the experiment, change the parameter dictionary in *experiment.py*.
By default, the values are set to the values used in the article.
## Starting Experiment
```
python experiment.py
```
## Results
Intermediate results will be saved as pickeled dictionaries (file extension .p), where keys are tracked quantities and values are lists of values.

The final result will be saved as pickeled pandas dataframe. To load it, use for example
```
import pickle
import pandas
df_result = pickle.load(open(path, 'rb'))
df_result.head() # show first few rows
```

The original results can be found in the pandas dataframe `original_result.p`.