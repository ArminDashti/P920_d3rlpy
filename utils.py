import argparse
import torch
import d3rlpy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from d3rlpy.datasets import get_minari
from d3rlpy.algos import SACConfig, CQLConfig, BEARConfig
from d3rlpy.metrics.evaluators import EnvironmentEvaluator
from d3rlpy.algos.qlearning.base import QLearningAlgoBase
from d3rlpy.torch_utility import TorchMiniBatch
from d3rlpy.algos.qlearning.torch import sac_impl
from d3rlpy.models.torch import get_parameter, build_squashed_gaussian_distribution
from d3rlpy.algos.qlearning.torch.ddpg_impl import DDPGBaseCriticLoss
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from d3rlpy.preprocessing import StandardRewardScaler, MinMaxRewardScaler
from d3rlpy.dataclass_utils import asdict_as_float
import sys
import inspect
device = "cuda" if torch.cuda.is_available() else "cpu"

