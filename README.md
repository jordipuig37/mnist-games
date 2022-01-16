# MNIST Games

This is my implementation of the MultiStep MNIST Game from [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676) using Python and Torch.

## Results reporduction

In order to obtain the results showed in __ execute the following commands:

* Set up the environment:

```
python -m venv games-env
source games-env/bin/activate
pip install -r requirements.txt
```

* To replicate the experiments varying the __sigma__:
```
python main.py -conf conf.json -parameter sigma
```

* To replicate the experiments varying the __gamma__:
```
python main.py -conf conf.json -parameter gamma
```

## References:

[Original Paper](https://arxiv.org/abs/1605.06676)

[Heavily inspired by](https://github.com/minqi/learning-to-communicate-pytorch)