import tensorflow as tf
import numpy as np
import time
from utils import logger

class Trainer(object):
    def __init__(self,algo,
                env,
                sampler, #seq2seq_meta_sampler
                sample_processor,
                policy,
                n_itr,
                greedy_finish_time,
                start_itr=0,
                inner_batch_size = 500,
                save_interval = 100):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr  # 指定要迭代的次数
        self.start_itr = start_itr  #指定从第几次开始迭代
        self.inner_batch_size = inner_batch_size    #指定每一批次训练数据的大小，默认为 500
        self.greedy_finish_time = greedy_finish_time    # 指定贪婪结束的时间
        self.save_interval = save_interval  # 指定模型保存的间隔，默认为100次迭代

    def train(self):
        """
        实现元学习算法的训练过程
        Implement the MRLCO training process for task offloading problem
        为任务卸载问题实施 MRLCO 训练过程
        """
        # 将当前时间定为开始时间
        start_time = time.time()
        avg_ret = []    #迭代期间的平均奖励
        avg_loss = []   # 迭代期间的平均损失
        avg_latencies = []  #迭代期间的平均延迟
        # 循环训练次数
        for itr in range(self.start_itr, self.n_itr):
            #迭代初始时间设定为当前时间
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")
            # 为每一个元任务进行采样，
            # 根据 sampler 里面的代码，好像在这一步就已经 进行了agent 训练。
            # 为每一个元任务更新子任务
            task_ids = self.sampler.update_tasks()
            #  path 是一个列表，列表里面存放的元素是字典，采样的东西放在 path 中
            paths = self.sampler.obtain_samples(log=False, log_prefix='')
            # 打印采集的 path 数量
            print("sampled path length is: ", len(paths[0]))

            # 对于每个任务，先调用 贪婪算法执行一下，然后返回执行时间，记录在列表中。
            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            # 再将贪婪算法执行的平均延迟 记录在日志中
            logger.logkv('Average greedy latency,', np.mean(greedy_run_time))

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            # 对采样的 path 进行处理
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')
            print("------------------- Inner Policy Update --------------------")
            """ ------------------- Inner Policy Update --------------------"""
            # 以 inner_batch_size 大小的批次调用 PPO 算法来更新代理的策略和价值网络
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size )

            #print("task losses: ", losses)
            print("average task losses: ", np.mean(policy_losses))
            avg_loss.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))


            print("------------------ Resample from updated sub-task policy ------------")
            """ ------------------ Resample from updated sub-task policy ------------"""
            print("Evaluate the one-step update for sub-task policy")
            # 使用现有策略 进行重新采样，得到一个新的路径 new_paths
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            # 然后对新的path 进行处理
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')

            print("------------------ Outer Policy Update ---------------------")
            """ ------------------ Outer Policy Update ---------------------"""
            logger.log("Optimizing policy...")
            # 外循环更新 元策略。
            self.algo.UpdateMetaPolicy()

            """ ------------------- Logging Stuff --------------------------"""
            # 计算整个元任务的 平均奖励
            ret = np.array([])
            for i in range(5):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret)

            # 计算整个元任务的平均延迟
            latency = np.array([])
            for i in range(5):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)    # 记录当前训练次数
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', avg_latency)

            logger.dumpkvs()    # 将日志输出到标准输出中
            avg_ret.append(avg_reward)  # 将平均奖励添加到 avg_ret 列表中
            # 将策略模型保存在指定文件中
            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_"+str(itr)+".ckpt")
        # 在训练结束时，将最终策略模型保存在指定文件夹中
        self.policy.core_policy.save_variables(save_path="./meta_model_inner_step1/meta_model_final.ckpt")

        return avg_ret, avg_loss, avg_latencies


if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir="./meta_offloading20_log-inner_step1/", format_strs=['stdout', 'log', 'csv'])

    # 每次元更新采样的任务数
    META_BATCH_SIZE = 10

    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)
    # 设置卸载环境
    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=100,
                                graph_number=100,
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_3/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_5/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_6/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_7/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_9/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_10/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_11/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_13/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_14/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_15/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_17/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_18/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_19/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_21/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_22/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_23/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_25/random.20.",
                                ],
                                time_major=False)

    action, greedy_finish_time = env.greedy_solution()
    print("avg greedy solution: ", np.mean(greedy_finish_time))
    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))
    print()

    baseline = ValueFunctionBaseline()

    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, encoder_units=128, decoder_units=128,
                                    vocab_size=2)

    sampler = Seq2SeqMetaSampler(
        env=env,    # 指定环境对象，表示采样器在该环境下进行采样
        policy=meta_policy, # 制定了元策略对象，表示采样器将使用该元策略来生成样本数据
        rollouts_per_meta_task=1,  # 指定每个元任务的轨迹数量。每个元任务将执行一次轨迹采样 Sample trajectories set
        meta_batch_size=META_BATCH_SIZE, # 指定 meta_batch 的大小，即一次采样中同时处理的元任务数量
        max_path_length=20000,
        parallel=False,
    )

    sample_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=0.99,
                                                   gae_lambda=0.95,
                                                   normalize_adv=True, # 指定对优势估计进行归一化处理
                                                   positive_adv=False) #
    algo = MRLCO(policy=meta_policy,
                         meta_sampler=sampler,
                         meta_sampler_process=sample_processor,
                         inner_lr=5e-4,
                         outer_lr=5e-4,
                         meta_batch_size=META_BATCH_SIZE,
                         num_inner_grad_steps=1,
                         clip_value = 0.3)

    trainer = Trainer(algo = algo,
                        env=env,
                        sampler=sampler,
                        sample_processor=sample_processor,
                        policy=meta_policy,
                        n_itr=2000,
                        greedy_finish_time= greedy_finish_time,
                        start_itr=0,
                        inner_batch_size=1000)
    # session 会话控制，一些session 可能拥有一些资源，当不需要该session 时，就需要对其资源进行释放。
    # 使用一下方法可以 创建上下文 context 时执行，当上下文退出时自动释放
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        avg_ret, avg_loss, avg_latencies = trainer.train()


