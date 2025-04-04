Create the data
```shell
py scripts/kernel-synth.py -N 320 -J 5
py scripts/kernel-synth.py -N 32_000 -J 5
```

```shell
py scripts/training/train.py --config scripts/training/configs/reproduce.yaml
py train.py --config scripts/training/configs/mlr.yaml
```
