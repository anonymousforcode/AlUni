from collections import deque, OrderedDict
import numpy as np
import copy

from iaiu.utils.eval_util import create_stats_ordered_dict
from iaiu.collectors.base_collector import StepCollector
import iaiu.torch_modules.utils as ptu
from iaiu.utils.logger import logger
import time
from tqdm import tqdm

class SimpleCollector(StepCollector):
    def __init__(
        self,
        env,
        agent,
        pool=None,
        with_path_buff=False,
        action_repeat=1,
        memory_size=5,
        agent_step_kwargs={}
    ):
        self._env = env
        self._agent = agent
        self._pool = pool
        self._n_env = env.n_env
        self.with_path_buff = with_path_buff 
        self._init_path_buff()

        self._size = memory_size
        self._return_memory = deque(maxlen=memory_size)
        self._length_memory = deque(maxlen=memory_size)
        self._ret = self._len = []
        self._start_new_path()

        self.env_step_time = 0
        self.agent_step_time = 0
        self.total_step = 0
        if action_repeat is not None:
            self.action_repeat = action_repeat
        else:
            self.action_repeat = env.action_repeat
        self.agent_step_kwargs = agent_step_kwargs

    def _init_path_buff(self):
        if self.with_path_buff:
            self.path = {
                "actions": [],
                "agent_infos": [],
                "observations": [],
                "rewards": [],
                "terminals": [],
                "env_infos": []
            }

    def collect_new_steps(
        self,
        num_steps,
        max_path_length=1000,
        action_when_terminal="reset",
        use_tqdm=False, # !!!gaila
        **update_kwargs
    ):
        agent_step_kwargs = copy.deepcopy(self.agent_step_kwargs)
        agent_step_kwargs.update(update_kwargs)
        assert action_when_terminal in ["reset", "stop"]
        assert num_steps % self.action_repeat == 0
        assert max_path_length % self.action_repeat == 0
        _max_path_length = max_path_length // self.action_repeat
        step_count = 0
        if action_when_terminal == "stop" and self._t.all():
            self._start_new_path()
        
        stop = False
        if use_tqdm:
            pbar = tqdm(total=num_steps)

        while not stop and step_count < num_steps:
            num_collect, stop = self._collect_one_step(
                _max_path_length, 
                action_when_terminal,
                **agent_step_kwargs
            )
            num_collect = num_collect * self.action_repeat
            if use_tqdm:
                pbar.update(num_collect)
            step_count += num_collect

        if use_tqdm:
            pbar.close()
        self.total_step += num_steps
        return step_count

    def _collect_one_step(
            self,
            _max_path_length,
            action_when_terminal,
            **agent_step_kwargs
        ):
        t = time.time()
        a, agent_info = self._agent.step(self._cur_o, **agent_step_kwargs)
        self.agent_step_time += time.time() - t
        t = time.time()
        if hasattr(self._env, "step_np"):
            next_o, r, d, env_info = self._env.step_np(a)
        else: 
            next_o, r, d, env_info = self._env.step(a)
        self.env_step_time += time.time() - t
        live = 1-self._t
        num_collect = live.sum()
        r = live * r
        self._t = np.logical_or(self._t, d)
        if self._pool is not None:
            self._pool.add_samples(
                {
                    'observations': self._cur_o,
                    'next_observations': next_o,
                    'actions': a,
                    'rewards': r,
                    'terminals': self._t,
                    'agent_infos': agent_info,
                    'env_infos': env_info
                }
            )
        self._cur_o = next_o
        if self.with_path_buff:
            self.path["actions"].append(a)
            self.path["agent_infos"].append(agent_info)
            self.path["observations"].append(self._cur_o)
            self.path["rewards"].append(r)
            self.path["terminals"].append(self._t)
            self.path["env_infos"].append(env_info)
        
        # statistic
        self._ret = self._ret + r
        self._len = self._len + live*self.action_repeat
        self._step_id += 1
        if self._t.all() or self._step_id >= _max_path_length:
            self._agent.end_a_path(next_o)
            if self._pool is not None:
                self._pool.end_a_path(next_o)
            # save statistics
            for item in self._ret:
                self._return_memory.append(item[0])
            for item in self._len:
                self._length_memory.append(item[0])

            if action_when_terminal == "stop":
                stop = True
            elif action_when_terminal == "reset":
                stop = False
                self._start_new_path()
            else:
                raise NotImplementedError
        else:
            stop = False

        return num_collect, stop

    def _start_new_path(self):
        self._t = np.zeros((self._n_env,1))
        self._cur_o = self._env.reset() # (1, 9, 84, 84)
        self._agent.start_new_path(self._cur_o)
        if self._pool is not None:
            self._pool.start_new_path(self._cur_o)

        self._step_id = 0
        self._ret = np.zeros((self._n_env,1))
        self._len = np.zeros((self._n_env,1), dtype=int)

        if self.with_path_buff:
            self.path["observations"].append(self._cur_o)
            self.path["terminals"].append(self._t)
    
    def start_epoch(self, epoch=None, clear_memory=True):
        if clear_memory:
            self._init_path_buff()
        self.env_step_time = 0
        self.agent_step_time = 0

    def end_epoch(self, epoch=None):
        pass

    # TODO with_path for more informed diag
    def get_diagnostics(self):
        diagnostics =  OrderedDict([
            ('last_return', self._return_memory[-1]),
            ('length', np.mean(self._length_memory)),
            ('total_step', self.total_step),
            ('agent_step_time', self.agent_step_time),
            ('env_step_time', self.env_step_time)
        ])
        diagnostics.update(
            create_stats_ordered_dict(
                'Return', 
                self._return_memory
            )
        )
        diagnostics.update(
            create_stats_ordered_dict(
                'Path Length', 
                self._length_memory
            )
        )
        return diagnostics


if __name__ == "__main__":
    from iaiu.environments.dmc_env import DMControlEnv
    from iaiu.pools.trajectory_pool import TrajectoryPool
    from iaiu.agents.base_agent import Agent
    env = DMControlEnv("cheetah","run")
    # env.observation_space.shape, env.action_space.shape:
    # (9, 64, 64), (6, )
    pi = Agent(env)
    pool = TrajectoryPool(env, 300, 6)
    env.reset()
    cl = SimpleCollector(env, pi, pool)
    cl.collect_new_steps(4000,100)
    print(cl._return_memory, cl._length_memory)
    # deque([5.6175959677494065, 4.397530891333625, 12.36968306868883, 0.6087472093729139], maxlen=5)
    # deque([100.0, 100.0, 100.0, 100.0], maxlen=5)
    print(pool._size) # 283
    np.save('./TrajPool_1.npy', pool.dataset)

