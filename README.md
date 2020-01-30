## Deep Workflow Scheduler

### Setup environment
In project directory

```
./vagrant up
```

OR

In project directory

```
docker build . -t img
docker run img
```

### Dependencies

python3.6:
* numpy 
* networkx 
* cython
* other pysimgrid dependencies
* torch

### Run experiments
```
python3.6 gcn_experiments.py
```
