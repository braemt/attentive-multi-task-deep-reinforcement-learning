# Attentive Multi Task Deep Reinforcement Learning

The code contains an implementation and environments of Attentive Multitask Deep Reinforcement Learning (Bräm et al.). It uses the [A3C algorithm](https://arxiv.org/abs/1602.01783) based on the [universe-starter-agent](https://github.com/openai/universe-starter-agent).

# Dependencies

* Python 2.7 or 3.5
* [Golang](https://golang.org/doc/install)
* [six](https://pypi.python.org/pypi/six) (for py2/3 compatibility)
* [TensorFlow](https://www.tensorflow.org/) 0.12
* [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
* [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
* [gym](https://pypi.python.org/pypi/gym)
* libjpeg-turbo (`brew install libjpeg-turbo`)
* [universe](https://pypi.python.org/pypi/universe)
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)

# Getting Started

```
conda create --name universe-starter-agent python=3.5
source activate universe-starter-agent

brew install tmux htop cmake golang libjpeg-turbo      # On Linux use sudo apt-get install -y tmux htop cmake golang libjpeg-dev

pip install "gym[atari]"
pip install universe
pip install six
pip install tensorflow==1.5.0
conda install -y -c https://conda.binstar.org/menpo opencv3
conda install -y numpy
conda install -y scipy

pip install -e /path/to/environments/
```


Add the following to your `.bashrc` so that you'll have the correct environment when the `train.py` script spawns new bash shells
```source activate universe-starter-agent```

## Grid Worlds

`python train.py --env-id grid-worlds-v1,grid-worlds-v2 --log-dir /tmp/grid-worlds`

The command above will train an agent on the grid-worlds-v1 and grid-worlds-v2 tasks.

Once you start the training process, it will create a tmux session with a window for each of all processes started. You can connect to them by typing `tmux a` in the console.
Once in the tmux session, you can see all your windows with `ctrl-b w`.
To switch to window number 0, type: `ctrl-b 0`. Look up tmux documentation for more commands.

To access TensorBoard to see various monitoring metrics of the agent, open [http://localhost:12345/](http://localhost:12345/) in a browser.

You can stop the experiment with `tmux kill-session` command.
