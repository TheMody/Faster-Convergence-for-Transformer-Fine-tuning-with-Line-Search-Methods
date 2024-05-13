# Faster Convergence for Transformer Fine-tuning with Line Search Methods

The Repository to the Paper Faster Convergence for Transformer Fine-tuning with Line Search Methods

More recent implementation at:

https://github.com/TheMody/No-learning-rates-needed-Introducing-SALSA-Stable-Armijo-Line-Search-Adaptation
## Abstract

Recent works have shown that line search methods greatly increase performance of traditional stochastic gradient descent methods on a variety of datasets and architectures. In this work we succeed in extending line search methods to the novel and highly popular Transformer architecture and dataset domains in natural language processing. 
More specifically, we combine the Armijo line search with the Adam optimizer and extend it by subdividing the networks architecture into sensible units and perform the line search separately on these local units. 
Our optimization method outperforms the traditional Adam optimizer and achieves significant performance improvements for small data sets or small training budgets, while performing equal or better for other tested cases.
Our work is publicly available as a python package, which provides a hyperparameter-free pytorch optimizer that is compatible with arbitrary network architectures.

![Loss Curve](Plots/lossSST2small.png)

## Install

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3

for replication:
- `pip install transformers` for huggingface transformers <3 
- `pip install datasets` for huggingface datasets <3 
- `pip install tensorflow-datasets` for tensorflow datasets <3 
- `pip install wandb` for optional logging <3
- for easy replication use conda and environment.yml eg:
`$ conda env create -f environment.yml` and `$ conda activate sls3`



## Use in own projects

The custom optimizer is in \sls\adam_sls.py and \sls\sls_base.py 
Example Usage:

```
from sls.adam_sls import AdamSLS
optimizer = AdamSLS([model.parameters()])
```
For step size smoothing (not described in the original paper, but performs better) use::
```
optimizer = AdamSLS([model.parameters()], smooth = True, c = 0.5)
```
For splitting the learning rates in your network use:
```
optimizer = AdamSLS([parameterlistA, parameterlistB, ... etc], smooth = True, c = 0.5)
```

The typical pytorch forward pass needs to be changed from :
``` 
optimizer.zero_grad()
y_pred = model(batch_x)
loss = criterion(y_pred, batch_y)    
loss.backward()
optimizer.step()
scheduler.step() 
```
to:
``` 
closure = lambda : criterion(model(batch_x), batch_y)
optimizer.zero_grad()
loss = optimizer.step(closure = closure)
```

This code change is necessary since, the optimizers needs to perform additional forward passes and thus needs to have the forward pass encapsulated in a function.
see embedder.py in the fit() method for more details


## Replicating Results
For replicating the main Results of the Paper run:

```
$ python run_multiple.py
```


For replicating specific runs or trying out different hyperparameters use:

```
$ python main.py 
```

and change the config.json file appropriately



## Please cite:
Faster Convergence for Transformer Fine-tuning
with Line Search Methods 
from 
Philip Kenneweg,
Leonardo Galli,
Tristan Kenneweg,
Barbara Hammer
published in IJCNN 2023
