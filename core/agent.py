import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
from torch.autograd import Variable
import math
import time


def collect_samples(pid, queue, env, policy, custom_reward, mean_action,
                    tensor, render, running_state, update_rs, min_batch_size):
    torch.randn(pid, )
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    total_path_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    num_term = 0

    while num_steps < min_batch_size:
        state = env.reset()
        if running_state is not None:
            state = running_state(state, update=update_rs)
        reward_episode = 0

        for t in range(10000):
            state_var = Variable(tensor(state).unsqueeze(0), volatile=True)
            if mean_action:
                action = policy(state_var)[0].data[0].numpy()
            else:
                action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
#            next_state, reward, done, _ = env.step(action)
            next_state, reward, done, d = env.step(action)
            if t == 99:
                path_reward = d.get('path_reward')
                terminal_condition = d.get('terminal_condition')

            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state, update=update_rs)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1  # mask:終了判定 doneと逆

            memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
            if done:
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)
        total_path_reward += path_reward
        num_term += terminal_condition


    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    log['total_path_reward'] = total_path_reward
    log['avg_path_reward'] = total_path_reward / num_episodes
    log['num_term'] = num_term
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['total_path_reward'] = sum([x['total_path_reward'] for x in log_list])
    log['avg_path_reward'] = log['total_path_reward'] / log['num_episodes']
    log['num_term'] = sum([x['num_term'] for x in log_list])
    log['success_rate'] = log['num_term'] / log['num_episodes'] * 100
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env_factory, policy, custom_reward=None, mean_action=False, render=False,
                 tensor_type=torch.DoubleTensor, running_state=None, num_threads=1):
        self.env_factory = env_factory
        self.policy = policy
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.tensor = tensor_type
        self.num_threads = num_threads
        self.env_list = []
        for i in range(num_threads):
            self.env_list.append(self.env_factory(i))

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        if use_gpu:
            self.policy.cpu()
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))  # (最小バッチ（サンプリングタイムステップ）サイズ（）/スレッド（4）)
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env_list[i + 1], self.policy, self.custom_reward, self.mean_action,
                           self.tensor, False, self.running_state, False, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env_list[0], self.policy, self.custom_reward, self.mean_action,
                                      self.tensor, self.render, self.running_state, True, thread_batch_size)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        if use_gpu:
            self.policy.cuda()
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
