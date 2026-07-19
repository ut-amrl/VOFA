# VOFA
VOFA is developed on top of the [Booster Gym repository](https://github.com/BoosterRobotics/booster_gym) developed by [Booster Robotics](https://boosterobotics.com/).

## Installation

Follow these steps to set up your environment:

1. Create an environment with Python 3.8:

    ```sh
    $ conda create --name vofa python=3.8
    $ conda activate vofa
    ```

2. Install PyTorch with CUDA support:

    ```sh
    $ conda install numpy=1.21.6 pytorch=2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3. Install Isaac Gym

    Download Isaac Gym from [NVIDIA’s website](https://developer.nvidia.com/isaac-gym/download).

    Extract and install:

    ```sh
    $ tar -xzvf IsaacGym_Preview_4_Package.tar.gz
    $ cd isaacgym/python
    $ pip install -e .
    ```

    Configure the environment to handle shared libraries, otherwise cannot found shared library of `libpython3.8`:

    ```sh
    $ cd $CONDA_PREFIX
    $ mkdir -p ./etc/conda/activate.d
    $ vim ./etc/conda/activate.d/env_vars.sh  # Add the following line
    export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
    $ mkdir -p ./etc/conda/deactivate.d
    $ vim ./etc/conda/deactivate.d/env_vars.sh  # Add the following line
    export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
    unset OLD_LD_LIBRARY_PATH
    ```

4. Install Python dependencies:

    ```sh
    $ pip install -r requirements.txt
    ```

5. Download the pretrained VOFA policies from [Hugging Face](https://huggingface.co/zichao22/vofa):

    ```sh
    $ huggingface-cli download zichao22/vofa --repo-type model --local-dir vofa/logs/vofa
    ```

    This creates the following checkpoint layout:

    ```text
    vofa/logs/vofa/
    ├── student_policy/
    │   ├── config.yaml
    │   └── nn/
    │       └── model.pth
    └── teacher_policy/
        ├── config.yaml
        └── nn/
            └── model.pth
    ```

## Usage

### 1. Testing

To test a trained policy in Isaac Gym, run:

```sh
$ cd vofa
$ python run.py --alg <RUNNER_NAME> --test --config <PATH/TO/CONFIG_FILE> --checkpoint <PATH/TO/CHECKPOINT>
```

For example, to test the pretrained teacher policy:

```sh
$ python run.py --alg FalconPPO --test --config logs/vofa/teacher_policy/config.yaml --checkpoint logs/vofa/teacher_policy/nn/model.pth
```

To test the pretrained student policy:

```sh
$ python run.py --alg FalconCamEvaluate --test --config logs/vofa/student_policy/config.yaml --checkpoint logs/vofa/student_policy/nn/model.pth
```

Video recording is controlled by `viewer.record_video` in the config file. When enabled, videos are saved under the checkpoint directory in `../videos/<date-time>/video.mp4`.

### 2. Training
We provide a pre-trained checkpoint for the low-level whole body controller used in this work here for Booster T1 23Dof.

To train the teacher policy, run the following command:

```sh
$ cd vofa
$ python run.py --alg FalconPPO --config configs/train_teacher.yaml
```

To train the student policy, run the following command:
```sh
$ cd vofa
$ python run.py --alg FalconDAgger --config configs/train_student.yaml --expert_path <PATH/TO/TEACHER_CHECKPOINT>
```

Training logs and saved models will be stored in `logs/<date-time>/`.

#### Configurations

Training settings are loaded from `configs/**/<task>.yaml`. You can also override config values using command-line arguments:

- `--checkpoint`: Path of the model checkpoint to load (set to `-1` to use the most recent model).
- `--num_envs`: Number of environments to create.
- `--headless`: Run headless without creating a viewer window.
- `--sim_device`: Device for physics simulation (e.g., `cuda:0`, `cpu`). 
- `--rl_device`: Device for the RL algorithm (e.g., `cuda:0`, `cpu`). 
- `--seed`: Random seed.
- `--max_iterations`: Maximum number of training iterations.

To add a new task, create a config file in `envs/` and register the environment in `envs/__init__.py`.

### 3. Deployment

In progress...

## Citation

```bibtex
@misc{hu2026vofavisualobjectgoal,
      title={VOFA: Visual Object Goal Pushing with Force-Adaptive Control for Humanoids}, 
      author={Zichao Hu and Zifan Xu and Dongsik Chang and He Yin and Linh Tran and Roberto Martín-Martín and Peter Stone and Jingyu Qiao and Joydeep Biswas},
      year={2026},
      eprint={2605.01518},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.01518}, 
}
```
