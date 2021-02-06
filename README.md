# cifar-100
CIFAR100 rep

## Installation
    ! pip install tensorflow==2.3.0
    ! pip install -r requirements.txt
  
Or for docker, use the Dockerfile.gpu docker file
      
## Modules
EDA.ipynb  ->  exploratory data analysis notebook

etl.py  ->  dataset creation and manipulation module

models.py. ->  models and submodels module, all the architecures are here

train.py  -> training module, the 'train' routine accepts a model and a dataset pair to train. Default training is using tensorflows default .fit  
             method while custom training (predict_mask=True) is a custom training routine

predict.py  -> inference module, used for deplyment / testing the model

models_preformance.ipynb  ->  notebook for keeping track of different models performance


## Getting started
Its recomended to start with model_performance.ipynb

## Performance
Best model performance is 0.61 categorical accuracy and 0.87 top 5 categorical accuracy
