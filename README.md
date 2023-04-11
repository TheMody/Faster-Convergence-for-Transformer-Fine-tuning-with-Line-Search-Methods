# Faster Convergence for Transformer Fine-tuning with Line Search Methods

The Repository to the Paper Faster Convergence for Transformer Fine-tuning with Line Search Methods

![Loss Curve](Plots/lossSST2small.png)

## install

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `pip install transformers` for huggingface transformers <3 
- `pip install datasets` for huggingface datasets <3 
- `pip install tensorflow-datasets` for tensorflow datasets <3 
- `pip install wandb` for optional logging <3
- for easy replication just use conda and environment.yml


## Replicating Results
For replicating all Results of the Paper just run:

```
$ python run_multiple.py
```


For replicating specific runs or trying out different hyperparameters use:

```
$ python main.py 
```

and change the config.json file appropriately

## use in own projects

The custom optimizer is in \sls\adam_sls.py and \sls\sls_base.py 
Example Usage:

```
optimizer = AdamSLS( [model.parameters()] )
```
forward pass needs to be changed from :
``` 
optimizer.zero_grad()
y_pred = self(batch_x)
loss = self.criterion(y_pred, batch_y)    
loss.backward()
self.optimizer.step()
self.scheduler.step() 
```
to:
``` 
closure = lambda : self.criterion(self(batch_x), batch_y)
self.optimizer.zero_grad()
loss = self.optimizer.step(closure = closure)
```
