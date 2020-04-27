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

python3.6 packages:
* numpy==1.18.2 
* networkx==2.4 
* cython==0.29.16
* other pysimgrid dependencies
* setuptools==20.7.0
* pysimgrid==1.0.0
* torch==1.4.0
* dgl==0.4.3.post2
* comet-ml==3.1.6

### Run experiments
```
python3.6 main.py
```
