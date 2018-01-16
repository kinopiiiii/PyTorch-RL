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
from core.common import estimate_advantages
from core.agent import Agent
from core.agent_term import AgentTerm
from datetime import datetime as dt
from dm_control import suite

DEBUG = False
term_num = 0

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
parser.add_argument('--tau', type=float, default=0.97, metavar='G',  # 学習率?
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',  # PPOのL2
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',  # ADAMの学習率
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',  # クリッピング値
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=3000, metavar='N',  # イテレーション回数
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',  # ログを取る間隔
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=10, metavar='N',  # モデルを保存する間隔
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
    policy_net = DiscretePolicy(state_dim, env_dummy.action_space.n)  # 離散方策関数
else:
    policy_net = Policy(state_dim, env_dummy.action_space.shape[0])  # 連続方策関数
value_net = Value(state_dim)
discrim_net = Discriminator(state_dim + action_dim)  # Dネットワーク
discrim_criterion = nn.BCELoss()  # 相互エントロピー損失関数

if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()
    discrim_net = discrim_net.cuda()
    discrim_criterion = discrim_criterion.cuda()

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

# load trajectory
expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))


def expert_reward(state, action):
    """Discriminator定義(Custom RewardにDからの返し値を設定)"""
    state_action = Tensor(np.hstack([state, action]))
    return -math.log(discrim_net(Variable(state_action, volatile=True)).data.numpy()[0])


"""create agent"""
agent = AgentTerm(env_factory, policy_net, custom_reward=expert_reward,
              running_state=running_state, render=args.render, num_threads=args.num_threads)

"""batchが学習対象の\pi_theta"""

def update_params(batch, i_iter):
    states_term_ = []
    actions_term_ = []
    rewards_term_ = []
    masks_term_ = []
    states_ = []
    actions_ = []
    rewards_ = []
    masks_ = []
    TERMINAL = False
    FULL_TERMINAL = False
    global term_num
    term_num = 0

    for batch_i in range(99,len(batch.state),100):
        #print(batch_i)
        if batch.term[batch_i] == 1:
            for batch_j in range(100):
                TERMINAL = True
                states_term_.append(batch.state[batch_i-99+batch_j])
                actions_term_.append(batch.action[batch_i-99+batch_j])
                rewards_term_.append(batch.reward[batch_i-99+batch_j])
                masks_term_.append(batch.mask[batch_i-99+batch_j])
                term_num = term_num + 1
                #print(term_num)
                if term_num == 2400:
                    FULL_TERMINAL = True

        else:
            for batch_k in range(100):
                states_.append(batch.state[batch_i-99+batch_k])
                actions_.append(batch.action[batch_i-99+batch_k])
                rewards_.append(batch.reward[batch_i-99+batch_k])
                masks_.append(batch.mask[batch_i-99+batch_k])
    """batch(学習対象)の抜き出し"""
    if not FULL_TERMINAL:
        states = torch.from_numpy(np.stack(states_))
        actions = torch.from_numpy(np.stack(actions_))
        rewards = torch.from_numpy(np.stack(rewards_))
        masks = torch.from_numpy(np.stack(masks_).astype(np.float64))
        if use_gpu:
            states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
        values = value_net(Variable(states, volatile=True)).data  # 状態価値の算出?
        fixed_log_probs = policy_net.get_log_prob(Variable(states, volatile=True), Variable(actions)).data  # πθoldの算出（PPOのrの下）

    if TERMINAL:
        states_term = torch.from_numpy(np.stack(states_term_))
        actions_term = torch.from_numpy(np.stack(actions_term_))
        rewards_term = torch.from_numpy(np.stack(rewards_term_))
        masks_term = torch.from_numpy(np.stack(masks_term_).astype(np.float64))
        if use_gpu:
            states_term, actions_term, rewards_term, masks_term = states_term.cuda(), actions_term.cuda(), rewards_term.cuda(), masks_term.cuda()

    """get advantage estimation from the trajectories"""
    if not FULL_TERMINAL:
        advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)  # アドバンテージ関数の推定

    lr_mult = max(1.0 - float(i_iter) / args.max_iter_num, 0)  # わからん

    """Discriminator更新"""
    for _ in range(3):
        """
        エキスパート配列 50000*30 状態23 行動7
        学習軌道配列   2400*30 状態23 行動7
        """

        expert_state_actions = Tensor(expert_traj)  # エキスパートの配列をDoubleTensorに変換
        if use_gpu:
            expert_state_actions = expert_state_actions.cuda()

        if TERMINAL:  # タスク達成済み軌道がある場合，expert_state_actionsに加える
            torch_state_action = torch.cat([states_term, actions_term], 1)
            expert_state_actions = torch.cat([expert_state_actions, torch_state_action], 0)

        if not FULL_TERMINAL:
            g_o = discrim_net(Variable(torch.cat([states, actions], 1)))  # D(G) Gの状態，行動データを連結して，Dに入力，結果受け取る
        e_o = discrim_net(Variable(expert_state_actions))  # D(E) エキスパートの状態，行動データを連結して，Dに入力，結果受け取る
        optimizer_discrim.zero_grad()

        """全部1の配列との相互エントロピー（D（G）を1に近づけるための損失関数）+全部0の配列との相互エントロピー（D（E）を0に近づけるための損失関数）"""
        if TERMINAL:  # タスク達成軌道がある場合
            if FULL_TERMINAL:  # すべてがタスク達成軌道の場合
                discrim_loss = discrim_criterion(e_o, Variable(zeros((expert_traj.shape[0]+states_term.shape[0], 1))))
            else:
                discrim_loss = discrim_criterion(g_o, Variable(ones((states.shape[0], 1)))) + discrim_criterion(e_o, Variable(zeros((expert_traj.shape[0] + states_term.shape[0], 1))))
        else:
            discrim_loss = discrim_criterion(g_o, Variable(ones((states.shape[0], 1)))) + discrim_criterion(e_o, Variable(zeros((expert_traj.shape[0], 1))))
        discrim_loss.backward()  # Dの逆伝播
        optimizer_discrim.step()  # DのADAM更新

    if not FULL_TERMINAL:  # 学習する対象があるならPPO更新
        """Generator(PPO)の更新 ミニバッチ学習"""
        # optimization epoch number and batch size for PPO
        optim_epochs = 5  # PPOのエポック数
        optim_batch_size = 4096  # PPOのバッチサイズ
        #optim_epochs = 10
        #optim_batch_size = 64

        optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))  # 最適イテレーション数: バッチ数（2400?）/最適バッチサイズ4096 の切り上げ（1）
        for _ in range(optim_epochs):  # PPOのエポック数学習
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm)
            if use_gpu:
                perm = perm.cuda()
            states, actions, returns, advantages, fixed_log_probs = states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm]
            for i in range(optim_iter_num):  # バッチサイズで分割して，optim_iter_num分 PPOの学習
                ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]
                ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b, lr_mult, args.learning_rate, args.clip_epsilon, args.l2_reg)


def main_loop():
    global term_num
    LOGGING = False
    print("LOGGING = ", LOGGING)
    if LOGGING:
        tstr = dt.now().strftime('%m-%d-%H-%M')
        assets_path = "./assets/GAIL_TERM/" + args.env_name + "/" + tstr
        os.mkdir(assets_path)
        log_path = "./log/GAIL_TERM/" + args.env_name + "/" + tstr
        os.mkdir(log_path)

    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        if use_gpu:
            discrim_net.cpu()
        batch, log = agent.collect_samples(args.min_batch_size)  # 環境との相互作用 (step) logに情報が入る batchに軌道が入る?
        if use_gpu:
            discrim_net.cuda()

        #print("batch",batch)
        #print("batch_term",batch_term)

        t0 = time.time()
        update_params(batch, i_iter)  # 学習部（パラメータ更新）
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            """報酬等のデータの書き出し"""
            print('{} {} samp:{:.4f}  update:{:.4f}  ex_R_avg:{:.2f}  Score:{:.2f}  pScore:{:.4f}  Rate:{:.2f}'.format(i_iter, args.env_name, log['sample_time'], t1 - t0, log['avg_c_reward'], log['avg_reward'],log['avg_path_reward'],term_num/24))

            if LOGGING:
                with open(os.path.join('./log/GAIL_TERM/{}/{}/GAIL_TERM_{}.csv'.format(args.env_name, tstr, args.env_name)), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i_iter, log['avg_c_reward'], log['avg_reward'],log['avg_path_reward'],term_num/24])

        if args.save_model_interval > 0 and (i_iter + 1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu(), discrim_net.cpu()
            if LOGGING:
                """方策，価値，D(or状態)の保存"""
                pickle.dump((policy_net, value_net, running_state), open(os.path.join(assets_dir(), 'GAIL_TERM/{}/{}/GAIL_TERM_STATE{}_[{}]_.p'.format(args.env_name, tstr, args.env_name, i_iter)), 'wb'))
                pickle.dump((policy_net, value_net, discrim_net), open(os.path.join(assets_dir(), 'GAIL_TERM/{}/{}/GAIL_TERM_DIS{}_[{}]_.p'.format(args.env_name, tstr, args.env_name, i_iter)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda(), discrim_net.cuda()


main_loop()
