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

## How to Develop and Build

For the Python code, you will need standard packages such as `numpy`
and `torch`. I would recommend creating a virtual environment
and installing them there. You can see the full list of requiements
in `requirements.txt`.

Be sure to also run `pip install -e .` to install the `src` package,
so all the `import` statements work properly.

For the C++ code, you will need a suitable compiler and CMake.
You will also need a LibTorch distribution, which you can learn
to setup from the [docs](https://pytorch.org/cppdocs/installing.html).

### Linux

On Linux, simply `wget` and `unzip` the LibTorch ZIP archive into
your favorite location on your location. To build,
navigate to the `/cpp` directory and then run the commands:

```shell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
```

This will build the files into the `/cpp/build` directory.

### Windows

On Windows, install LibTorch using the
[PyTorch website](https://pytorch.org/get-started/locally/)
into your favorite location on your computer.

You will probably need to build using a Visual Studio compiler
on Release mode (I couldn't get G++ to work).

If you are using the CMake extension in VSCode, I would recommend
adding the path to your LibTorch installation to the CMake
configuration settings. You can find this by clicking the
cog wheel icon in the CMake extension side bar.

Then, add an entry with item `CMAKE_PREFIX_PATH` and value
`/absolute/path/to/libtorch/share/cmake/Torch`, e.g.
`C:/libtorch/share/cmake/Torch`.

Then, to build, open a new workspace inside the `/cpp` folder
and click the Build button with the CMake extension in VSCode.

To actually run any resultant executable, you will also
need to copy every `.dll` file from `/libtorch/lib` into
the same directory as the executable to be dynamically linked
at runtime.

## Starting Training Runs

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

**Remember to recompile the C++ code if you change the constant values!**

There are two shell scripts `/scripts/othello_worker.sh` and
`/scripts/othello_controller.sh` that run these two components.
There is also a top-level `othello_main.sh` that automatically
submits the job to MIT SuperCloud in triples mode. Right now,
the code is designed to train the neural network on one GPU
controller machine and collect data across 384 worker CPU cores.
The machines operate in lock step, though it is written to be
fault-tolerant to any individual worker machine dying.

A current work in progress is distributed data parallel training
to train across multiple GPUs. This is not yet implemented.
