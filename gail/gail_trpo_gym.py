import argparse
import gym
import os
import sys
import pickle
import time
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym import wrappers
from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch.autograd import Variable
from torch import nn
from core.ppo import ppo_step
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent
from datetime import datetime as dt


Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')  # 環境名
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')  # エキスパート軌道パス
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')  # render設定
parser.add_argument('--log-std', type=float, default=0, metavar='G',
                    help='log std for the policy (default: 0)')  # 方策のlog_std?(0.0)
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.99)')  # 割引変数γ(0.99)
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=3000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=20, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
args = parser.parse_args()


def env_factory(thread_id):
    env = gym.make(args.env_name)
    env.seed(args.seed + thread_id)  # 環境の乱数シードの設定
    return env


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

env_dummy = env_factory(0)
state_dim = env_dummy.observation_space.shape[0]
is_disc_action = len(env_dummy.action_space.shape) == 0
action_dim = (1 if is_disc_action else env_dummy.action_space.shape[0])
ActionTensor = LongTensor if is_disc_action else DoubleTensor

"""define actor, critic and discrimiator"""
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, env_dummy.action_space.n)
else:
    policy_net = Policy(state_dim, env_dummy.action_space.shape[0])
value_net = Value(state_dim)
discrim_net = Discriminator(state_dim + action_dim)
discrim_criterion = nn.BCELoss()
if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()
    discrim_net = discrim_net.cuda()
    discrim_criterion = discrim_criterion.cuda()

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 5  # PPOのエポック数
optim_batch_size = 4096  # PPOのバッチサイズ

# load trajectory
expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))


def expert_reward(state, action):
    state_action = Tensor(np.hstack([state, action]))
    return -math.log(discrim_net(Variable(state_action, volatile=True)).data.numpy()[0])


"""create agent"""
agent = Agent(env_factory, policy_net, custom_reward=expert_reward,
              running_state=running_state, render=args.render, num_threads=args.num_threads)


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state))
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward))
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
    if use_gpu:
        states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    values = value_net(Variable(states, volatile=True)).data
    fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)

    #lr_mult = max(1.0 - float(i_iter) / args.max_iter_num, 0)
    max_kl = 0.01
    damping = 0.1

    #update discriminator
    for _ in range(3):
    #for _ in range(1):
        expert_state_actions = Tensor(expert_traj)
        if use_gpu:
            expert_state_actions = expert_state_actions.cuda()
        g_o = discrim_net(Variable(torch.cat([states, actions], 1)))
        e_o = discrim_net(Variable(expert_state_actions))
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, Variable(ones((states.shape[0], 1)))) + \
                       discrim_criterion(e_o, Variable(zeros((expert_traj.shape[0], 1))))
        discrim_loss.backward()
        optimizer_discrim.step()

    #perform mini-batch PPO update
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm)
        if use_gpu:
            perm = perm.cuda()
        states, actions, returns, advantages, fixed_log_probs = \
            states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm]

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]
            trpo_step(policy_net, value_net, states_b, actions_b, returns_b, advantages_b, max_kl, damping, args.l2_reg)
            #ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
            #        advantages_b, fixed_log_probs_b, lr_mult, args.learning_rate, args.clip_epsilon, args.l2_reg)

def main_loop():
    tstr = dt.now().strftime('%m-%d-%H-%M')
    assets_path = "./assets/GAIL_TRPO/" + args.env_name + "/" + tstr
    os.mkdir(assets_path)
    log_path = "./log/GAIL_TRPO/" + args.env_name + "/" + tstr
    os.mkdir(log_path)

    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        if use_gpu:
            discrim_net.cpu()
        batch, log = agent.collect_samples(args.min_batch_size)  # 環境との相互作用 (step)
        if use_gpu:
            discrim_net.cuda()

        t0 = time.time()
        update_params(batch, i_iter)  # 学習部（パラメータ更新）
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            """報酬等のデータの書き出し"""
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1 - t0, log['avg_c_reward'], log['avg_reward']))

            with open(os.path.join('./log/GAIL_TRPO/{}/{}/GAIL_TRPO_{}.csv'.format(args.env_name, tstr, args.env_name)), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i_iter, log['avg_reward']])

        if args.save_model_interval > 0 and (i_iter + 1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu(), discrim_net.cpu()
            """方策，価値，D(or状態)の保存"""
            pickle.dump((policy_net, value_net, running_state), open(os.path.join(assets_dir(), 'GAIL_TRPO/{}/{}/GAIL_TRPO_STATE{}_[{}]_.p'.format(args.env_name, tstr, args.env_name, i_iter)), 'wb'))
            pickle.dump((policy_net, value_net, discrim_net), open(os.path.join(assets_dir(),
                                                                                  'GAIL_TRPO/{}/{}/GAIL_TRPO_DIS{}_[{}]_.p'.format(
                                                                                      args.env_name, tstr,
                                                                                      args.env_name,
                                                                                      i_iter)), 'wb'))

            if use_gpu:
                policy_net.cuda(), value_net.cuda(), discrim_net.cuda()


main_loop()
