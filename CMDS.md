Create the data
```shell
py scripts/kernel-synth.py -N 320 -J 5
```

```shell
py scripts/training/train.py --config scripts/training/configs/chronos-t5-tiny.yaml
py train.py --config scripts/training/configs/mlr.yaml
```
