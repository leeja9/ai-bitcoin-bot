from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, set_exploration_mode
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_cells = 256 
lr = 3e-4
max_grad_norm = 1.0


frame_skip = 1
frames_per_batch = 1000 // frame_skip
total_frames = 50_000 // frame_skip


sub_batch_size = 8  
num_epochs = 10  
clip_epsilon = (
    0.2  
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4


def add_reward_columns(df: pd.DataFrame):
    """add reward columns to dataframe for incremental updates"""
    for col in ['lr', 'alr', 'var_sum']:
        df[col] = 0

def update_reward_columns(history: History) -> None:
    """Set this episode lr, alr, var_sum, sr, powc"""

    # Using weighted incremental algorithmic approach for average
    # https://math.stackexchange.com/questions/106700/incremental-averaging
    # general formula is: mean = ((n - 1) * last_mean + this_value) / n))

    # logarithmic return
    this_lr = 0
    # if position is 1 (100% BTC)
    if history['position', -1] == 1:
        this_lr = np.log(history['data_close', -1]) - np.log(history['data_close', -2])
    history.__setitem__(('data_lr', -1), this_lr) # update history with new lr


    # running average of logarithmic return
    n = len(history)
    last_alr = history['data_alr', -2]
    this_alr = ((n - 1) * last_alr + this_lr) / n
    history.__setitem__(('data_alr', -1), this_alr) # update history with new alr

    # running variance sum of logarithmic return
    # for each nth row, dividing this sum by n gives population variance
    last_alr = history['data_alr', -2]
    last_var_sum = history['data_var_sum', -2]
    this_var_sum = last_var_sum + abs((this_lr - last_alr) * (this_lr - this_alr))
    history.__setitem__(('data_var_sum', -1), this_var_sum)

def get_random_weights(arr_len):
    """get numpy array of random weights"""
    max_val = 100
    weight_vector = np.zeros(arr_len)
    for i in range(arr_len - 1):
        n = np.random.randint(0, max_val)
        max_val = max_val - n
        weight_vector[i] = n
    weight_vector /= 100
    weight_vector[-1] = 1 - sum(weight_vector[:-1])
    np.random.shuffle(weight_vector)
    return weight_vector


def reward_function(history: History) -> float:
    """reward function for gym-trading-env"""
    update_reward_columns(history)
    average_log_return = history['data_alr', -1]
    var_sum = history['data_var_sum', -1]
    variance = var_sum / len(history)
    std_dev = np.sqrt(variance)
    sharpe_ratio = average_log_return / 0.5
    this_lr = history['data_lr', -1]
    powc = 0
    # if this eposide position is 0 (100% USD) and last position was 1 (100% BTC)
    # this compute time can also be traded for memory by adding a tracking column if needed
    if (history['position', -1] == 0 and history['position', -2] == 1):
        idx = history[-2]['idx']
        
        # This is an infinite loop if idx == 0 and history['position', idx] != 0.
        while idx >= 0:
            if (history['position', idx] == 0):
                last_lr = history['data_lr', idx + 1]
                powc = this_lr - last_lr
    reward_vector = np.array([average_log_return, sharpe_ratio, powc])
    weight_vector = get_random_weights(len(reward_vector))
    reward = reward_vector @ weight_vector # dot product of random weights and reward values
    return reward

def dynamic_features(history: History) -> float:
    """Calculates dynamic features."""
    #dyn_features = [last_position, real_position]
    #return dyn_features
    
    pass



env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [0, 1], # -1 (=SHORT), 0(=SELL ALL), +1 (=BUY ALL)
        #trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        #borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
        #dynamic_feature_functions = [dynamic_features]
        reward_function = reward_function,
        portfolio_initial_value = 10000,
        #max_episode_duration = 1000,
    )

action_space = gym.spaces.Discrete(2)

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tahn(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tahn(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tahn(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tahn(),
    nn.LazyLinear(2, device=device),
    NormalParamExtractor(),
)


policy_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["action"])


policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["action"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.action_spec.space.minimum,
        "max": env.action_spec.space.maximum,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)



data_collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)


replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)



advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor=policy_module,
    critic=value_module,
    advantage_key="advantage",
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)


class BitcoinTrainingAgent:
    """Agent using PPO."""
    def __init__(
        self,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
    ) -> None:
        """Initialize hyperparameters"""
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs):
        """Given an observation, choose an action"""
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        next_obs_tuple = tuple(next_obs)
        obs_tuple = tuple(obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs_tuple])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs_tuple][action]
        )

        self.q_values[obs_tuple][action] = (
            self.q_values[obs_tuple][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)
   
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)