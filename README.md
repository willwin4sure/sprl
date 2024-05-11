# sprl

`sprl` is a scalable self-play reinforcement learning framework,
built as a final project for the MIT graduate class
6.8200 Computational Sensorimotor Learning in Spring 2024.

The project aims to replicate the techniques of
[AlphaGo Zero](https://www.nature.com/articles/nature24270)
in order to solve two-player zero-sum abstract strategy games,
especially ones involving the placement of stones on a board,
such as Connect Four, Pentago, Othello (Reversi), Gomoku, Hex, and Go. 

The code can run on single machines or distribute across a compute
cluster such as [MIT SuperCloud](https://supercloud.mit.edu/).

## Code Organization

Python code is distributed across the `/src`, `/scripts`, and `/tests` directories.
Much of the code in `/src` is deprecated: it is an older and slower implementation
of the entire framework and much of it has been replaced with C++.

C++ code is available inside the `/cpp` directory, which is further subdivided.

In the current form of the code, the self-play data collection steps
are performed in C++. The game logic and Upper-Confidence Tree search
algorithm must be implemented in C++, as well as interface code to
save the data in a format parseable by `numpy`. Details such as
sub-tree reuse, virtual losses, data/inference symmetrization,
Dirichlet noise mixing, and parent Q-initialization are handled here.

Meanwhile, the training loop is implemented in Python. A standard
PyTorch training loop is implemented to improve the convolutional
neural network. The code is responsible for collating self-play data,
splitting it into training and validation sets, and tracing the networks
for inference in C++ using LibTorch.

## Requirements

Be sure to run `pip install -e .` in your environment to install the `src`
package, so the `import` statements all work. You will also need standard
packages such as `numpy` and `torch`.

To compile the C++ code, you will need a suitable LibTorch distribution.
You can learn how to set this up from the [docs](https://pytorch.org/cppdocs/installing.html).

### Linux

On Linux, simply `wget` and `unzip` the LibTorch ZIP archive, navigate
to the `/cpp` directory, and then run the commands:

```shell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
```

### Windows

On Windows, you will probably need to use a Visual Studio compiler on
Release mode (I couldn't get G++ to work). I would recommend installing
LibTorch into your root directory `C:/` and then adding the line
`list(APPEND CMAKE_PREFIX_PATH "C:\\libtorch\\share\\cmake\\Torch")`
into `CMakeLists.txt` in order for the compiler to locate it.

## Running the Code

Right now, there are two entrypoints into the code for the training loop.
You need to compile an executable such as `/cpp/build/OTHWorker.exe`
from `/cpp/OTHWorker.cpp`. These are run on the worker machines,
and the executable expects two command line arguments: its task id and the
total number of tasks.

Meanwhile, the controller is the Python script `/scripts/othello_controller.py`
The various constants can be changed for the run, but these **must** be sync-ed
with the constants in `cpp/constants.hpp` for correct behavior.
Further, the `RUN_NAME` constant **must** be sync-ed with the constant
string `runName` in `/cpp/OTHWorker.cpp`, otherwise the data will not be
transferred correctly.

There are two shell scripts `/scripts/othello_worker.sh` and
`/scripts/othello_controller.sh` that run these two components.
There is also a top-level `othello_main.sh` that automatically
submits the job to MIT SuperCloud in triples mode. Right now,
the code is designed to train the neural network on one GPU
controller machine and collect data across 384 worker CPU cores.
The machines operate in lock step, though it is written to be
fault-tolerant to any individual worker machine dying.
