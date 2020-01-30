# Policy Gradient Methods - REINFORCE and REINFORCE with ACTOR-CRITIC
This repository contains an implementation of REINFORCE and ACTOR-CRITIC algorithms. The code is present in the `src/` directory.

## Requirements
- `tensorflow` < `2.0.0`
- `keras`
- `matplotlib`
- `gym`
- `numpy`

## Usage
In order to run the code, use the configuration file like `lunarlander.py`. Then run the following command to start training and evaluating the model.
```
python3 reinforce.py --env lunarlander.py  # REINFORCE
python3 a2c.py --env lunarlander_a2c.py  # A2C
```

Hope you find it helpful.
