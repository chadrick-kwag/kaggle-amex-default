# amex default kaggle competition

competition url: https://www.kaggle.com/competitions/amex-default-prediction/overview

this repo tries to solve the problem using transformer encoder.

actual training dataset are processed versions from the original csv data given by the competition.

## how to train

1. setup an appropriate train yaml config file

2. run training

```
$ cd train
$ python trainer.py <config_file>
```

