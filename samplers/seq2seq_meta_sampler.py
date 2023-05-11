from samplers.base import Sampler
from samplers.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from utils import utils, logger
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools

class Seq2SeqMetaSampler(Sampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_policy_search.envs.base.MetaEnv) : environment object
        policy (meta_policy_search.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(self,
                env,
                policy,
                rollouts_per_meta_task,
                meta_batch_size,
                max_path_length,
                envs_per_task=None,
                parallel=False
                ):
        super(Seq2SeqMetaSampler, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')

        self.envs_per_task = rollouts_per_meta_task if envs_per_task is None else envs_per_task
        self.meta_batch_size = meta_batch_size
        self.total_samples = meta_batch_size * rollouts_per_meta_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0

        # setup vectorized environment
        if self.parallel:
            self.vec_env = MetaParallelEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MetaIterativeEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)

    def update_tasks(self):
        """
        Samples a new goal for each meta task
        """
        tasks = self.env.sample_tasks(self.meta_batch_size)
        # 断言：声明 len(tasks) 必须要和 meta_batch_size 相等，如果发生异常，那么就表示为假
        assert len(tasks) == self.meta_batch_size
        # 然后为每一个环境设置任务
        self.vec_env.set_tasks(tasks)
        # 返回的是设置的任务
        return tasks

    def obtain_samples(self, log=False, log_prefix=''):
        """
        为每个 task 收集 batch_size 的 轨迹
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        # 初始化过程
        # path 是一个通过 OrderedDict 定义的有序字典，大小为 meta_batch_size,然后将字典中的每一个元素设置为空
        paths = OrderedDict()
        for i in range(self.meta_batch_size):
            paths[i] = []

        # 记录 已采样的样本数量 为 0
        n_samples = 0
        # 初始化运行中的轨迹字典。
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        # 创建一个进度条对象，用于追踪采样的进度，总共需要采集 total_samplers个数据点。
        pbar = ProgBar(self.total_samples)
        # 初始化 策略执行时间和环境执行时间
        policy_time, env_time = 0, 0

        policy = self.policy

        # initial reset of envs
        obses = self.vec_env.reset()

        while n_samples < self.total_samples:
            # execute policy
            t = time.time()
            # obs_per_task = np.split(np.asarray(obses), self.meta_batch_size)
            obs_per_task = np.array(obses)

            actions, logits, values = policy.get_actions(obs_per_task)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            # actions = np.concatenate(actions)

            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)

            # print("rewards shape is: ", np.array(rewards).shape)
            # print("finish time shape is: ", np.array(env_infos).shape)


            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            new_samples = 0
            for idx, observation, action, logit, reward, value, done, task_finish_times in zip(itertools.count(), obses, actions, logits,
                                                                                    rewards, values, dones, env_infos):
                # append new samples to running paths

                # handling
                for single_ob, single_ac, single_logit, single_reward, single_value, single_task_finish_time \
                        in zip(observation, action, logit, reward, value, task_finish_times):
                    running_paths[idx]["observations"]= single_ob
                    running_paths[idx]["actions"] = single_ac
                    running_paths[idx]["logits"] = single_logit
                    running_paths[idx]["rewards"] = single_reward
                    running_paths[idx]["finish_time"] = single_task_finish_time
                    running_paths[idx]["values"] = single_value

                    paths[idx // self.envs_per_task].append(dict(
                        observations=np.squeeze(np.asarray(running_paths[idx]["observations"])),
                        actions=np.squeeze(np.asarray(running_paths[idx]["actions"])),
                        logits = np.squeeze(np.asarray(running_paths[idx]["logits"])),
                        rewards=np.squeeze(np.asarray(running_paths[idx]["rewards"])),
                        finish_time = np.squeeze(np.asarray(running_paths[idx]["finish_time"])),
                        values  = np.squeeze(np.asarray(running_paths[idx]["values"]))
                    ))

                # if running path is done, add it to paths and empty the running path
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)
        return paths

def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], logits=[], rewards=[])




