import time
import gym
import numpy as np
import torch
import copy
import os
import re
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from rl_games.algos_torch import torch_ext

from rl_games.common import env_configurations
from rl_games.algos_torch import model_builder
from rl_games.common import vecenv
from rl_games.common.a2c_common import print_statistics


class BasePlayer(object):

    def __init__(self, params):
        self.config = config = params['config']
        self.load_networks(params)
        self.env_name = self.config['env_name']
        self.player_config = self.config.get('player', {})
        self.env_config = self.config.get('env_config', {})
        self.env_config = self.player_config.get('env_config', self.env_config)
        self.env_info = self.config.get('env_info')
        self.clip_actions = config.get('clip_actions', True)
        self.seed = self.env_config.pop('seed', None)
        if self.env_info is None:
            use_vecenv = self.player_config.get('use_vecenv', False)
            if use_vecenv:
                print('[BasePlayer] Creating vecenv: ', self.env_name)
                self.env = vecenv.create_vec_env(
                    self.env_name, self.config['num_actors'], **self.env_config)
                self.env_info = self.env.get_env_info()
            else:
                print('[BasePlayer] Creating regular env: ', self.env_name)
                self.env = self.create_env()
                self.env_info = env_configurations.get_env_info(self.env)
        else:
            self.env = config.get('vec_env')

        self.num_agents = self.env_info.get('agents', 1)
        self.value_size = self.env_info.get('value_size', 1)
        self.action_space = self.env_info['action_space']

        self.observation_space = self.env_info['observation_space']
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.has_central_value = self.config.get(
            'central_value_config') is not None
        self.device_name = self.config.get('device_name', 'cuda')
        self.render_env = self.player_config.get('render', False)
        self.games_num = self.player_config.get('games_num', 2000)
        self.is_deterministic = self.player_config.get('deterministic', False)
        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)

        self.max_steps = 108000 // 4
        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)
        self.num_actors = config.get('num_actors', 1)
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)
        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode
        self.step = None

        self.device = torch.device(self.device_name)

        self.train_dir = config.get('train_dir', 'runs')
        self.experiment_dir = os.path.join(self.train_dir, config['name'])

        self.eval_nn_dir = os.path.join(self.experiment_dir, 'eval_nn' if not self.is_deterministic else 'eval_nn_det')
        self.eval_obs_dir = os.path.join(self.experiment_dir, 'eval_obs' if not self.is_deterministic else 'eval_obs_det')
        os.makedirs(self.eval_nn_dir, exist_ok=True)
        os.makedirs(self.eval_obs_dir, exist_ok=True)

        self.evaluation = self.player_config.get("evaluation", False) # run player as evaluation player to evaluate new checkpoints
        self.num_action_bins = config.get("num_action_bins", 100)
        self.save_observations = self.config.get("save_observations", False)

        self.writer = SummaryWriter(self.experiment_dir)

        # tensorboard hparams & metrics
        self._hparams = {
            'games_num': self.games_num,
            'n_games_life': self.n_game_life,
            'max_steps': self.max_steps,
        }

        self._metrics = {
            "rewards/step": 0,
        }


    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if 'obs' in obs:
                obs = obs['obs']
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert (obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(
                weights['running_mean_std'])

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_deterministic=False):
        raise NotImplementedError('step')

    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size(
            )[2]), dtype=torch.float32).to(self.device) for s in rnn_states]

    def _run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                        all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards / done_count
                        cur_steps_done = cur_steps / done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1} w: {game_res:.1}')
                        else:
                            print(f'reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

    def _run_eval(self):
        total_time_start = time.time()
        step_time = 0.0
        n_games_per_agent = self.games_num
        render = self.render_env

        sum_rewards = 0
        sum_steps = 0
        games_played = 0

        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        has_masks = self.env.has_action_mask() if has_masks_func else False

        op_agent = getattr(self.env, "create_agent", None)
        if self.is_rnn:
            self.init_rnn()

        obses = self.env_reset(self.env)
        batch_size = self.get_batch_size(obses, 1)

        cr = torch.zeros(batch_size, dtype=torch.float32)
        steps = torch.zeros(batch_size, dtype=torch.float32)
        games_finished_per_agent = torch.zeros(batch_size, dtype=torch.int32)
        actions = []    # store evaluation actions
        observations = []   # store observations

        while torch.any(games_finished_per_agent.lt(n_games_per_agent)):
            # play and collect data until every agent finished n_games_per_agent times
            eval_not_done = games_finished_per_agent.lt(n_games_per_agent)

            if has_masks:
                masks = self.env.get_action_mask()
                action = self.get_masked_action(obses, masks, self.is_deterministic)
            else:
                action = self.get_action(obses, self.is_deterministic)
            legal_actions = torch.zeros_like(action)
            legal_actions[eval_not_done] = action[eval_not_done]

            step_start = time.time()
            obses, r, done, info = self.env_step(self.env, legal_actions)
            step_end = time.time()
            d_step = step_end - step_start
            step_time += d_step

            cr += r
            steps += 1

            # collect actions
            actions.append(action[eval_not_done])

            # collect observations
            if self.save_observations:
                observations.append(obses[eval_not_done])

            all_done_indices = done.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            new_dones = done.to(torch.bool) & eval_not_done
            all_new_done_indices = new_dones.nonzero(as_tuple=False)
            new_done_indices = all_new_done_indices[::self.num_agents]
            games_finished_per_agent += new_dones.int()

            done_count = len(done_indices)
            games_played += len(new_done_indices)

            if done_count > 0:
                if self.is_rnn:
                    for s in self.states:
                        s[:, all_done_indices, :] *= 0.0

                cur_rewards = cr[new_done_indices].sum().item()
                sum_rewards += cur_rewards

                cur_steps = steps[new_done_indices].sum().item()
                sum_steps += cur_steps

                cr = cr * (1.0 - done.float())
                steps = steps * (1.0 - done.float())

                if torch.all(games_finished_per_agent.ge(n_games_per_agent)):
                    break

        mean_rewards = sum_rewards / games_played
        mean_lengths = sum_steps / games_played

        print('av reward:', mean_rewards, 'av steps:', mean_lengths)

        actions = torch.vstack(actions).to(self.device)

        if self.save_observations:
            observations = torch.vstack(observations).to(self.device)

        total_time_end = time.time()
        total_time = total_time_end - total_time_start

        return mean_rewards, mean_lengths, games_played, total_time, step_time, actions, observations

    def run(self):
        if self.evaluation:
            # evaluation player checks all checkpoints and logs to tensorboard
            exp, ssi, sei = hparams(self._hparams, metric_dict=self._metrics)
            self.writer.file_writer.add_summary(exp)
            self.writer.file_writer.add_summary(ssi)
            self.writer.file_writer.add_summary(sei)

            total_time = 0

            checkpoints = [os.path.join(self.eval_nn_dir, c) for c in os.listdir(self.eval_nn_dir)]
            checkpoints.sort(key=lambda x: os.path.getmtime(x))
            for fn in checkpoints:
                # restore checkpoint
                self.restore(fn)
                if not self.step:
                    # check filename and get frame
                    match = re.search(r".*/frame_(\d+).*.pth", fn)
                    if not match:
                        continue
                    self.step = int(match.group(1))

                # run evaluation
                mean_rewards, mean_lengths, games_played, epoch_total_time, step_time, actions, observations = self._run_eval()

                curr_frames = (mean_lengths * games_played)
                total_time += epoch_total_time
                fps_step = curr_frames / step_time
                fps_total = curr_frames / epoch_total_time

                epoch_num = self.step / self.num_frames_per_epoch

                print_statistics(self.print_stats, curr_frames, step_time, total_time, epoch_total_time,
                                 epoch_num, self.max_epochs, self.step, self.max_frames)

                # log action histogram
                a_min = float(self.action_space.low.min())
                a_max = float(self.action_space.high.max())
                bin_edges = torch.linspace(a_min, a_max, self.num_action_bins + 1)
                for  i in range(actions.shape[1]):
                    self.writer.add_histogram(f"actions/dim{i}", actions[:, i], self.step, bins=bin_edges)

                # save observations
                if self.save_observations:
                    # create filename
                    fn = os.path.join(self.eval_obs_dir, self.config['name'] + '_ep_' + str(epoch_num) + ".pth")
                    torch_ext.safe_filesystem_op(torch.save, observations, fn)

                self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.step)
                self.writer.add_scalar('performance/step_fps', fps_step, self.step)
                self.writer.add_scalar('performance/step_time', step_time, self.step)
                self.writer.add_scalar('performance/step_time', step_time, self.step)

                self.writer.add_scalar('info/epochs', epoch_num, self.step)

                self.writer.add_scalar('rewards/step', mean_rewards, self.step)
                self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                self.writer.add_scalar('episode_lengths/step', mean_lengths, self.step)
                self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

        else:
            self._run()

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if 'obs' in obses:
                obses = obses['obs']
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if 'observation' in obses:
                first_key = 'observation'
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size
