# following code is manually copied from stable-baseline3, as it depends on an older version of gym, but I don't want to solve this version conflict

import gym
from torchvision import transforms as T
import numpy as np
from gym import spaces
from collections import deque
from typing import Optional, Any, cast, List, Union
import torch
import numpy as np
from gym.spaces import Box
from gym import ObservationWrapper
import os
import gym
from typing import Callable, Optional

from gym import logger
import json
import os
import os.path
import pkgutil
import subprocess
import tempfile
from io import StringIO

import distutils.spawn
import distutils.version
import numpy as np

from gym import error, logger

try:
    import cv2  # pytype:disable=import-error

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None


class WarpFrame(gym.ObservationWrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: the environment
    :param width:
    :param height:
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.

    :param env: the environment
    """

    def __init__(self, env: gym.Env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward: float) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(reward)


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2, ) + env.observation_space.shape,
                                    dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class SkipFrames(gym.Wrapper):
    """Skip timesteps: repeat action, accumulate reward, take last obs."""

    def __init__(self, env, skip: int = 4):
        super(SkipFrames, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            info["steps"] = i + 1
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: the environment to wrap
    :param noop_max: the maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class LazyFrames:
    r"""Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to
    compress the observations.

    .. note::

        This object should only be converted to numpy array just before forward pass.

    Args:
        lz4_compress (bool): use lz4 to compress the frames internally

    """
    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames), ) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(
                self._frames[int_or_slice])  # single frame
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]],
            axis=0)

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame),
                                 dtype=self.dtype).reshape(self.frame_shape)
        return frame


class FrameStack(ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    .. note::

        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.

    .. note::

        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.

    Example::

        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)

    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally

    """

    def __init__(self, env, num_stack, lz4_compress=False):
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...],
                        num_stack,
                        axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...],
                         num_stack,
                         axis=0)
        self.observation_space = Box(low=low,
                                     high=high,
                                     dtype=self.observation_space.dtype)

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames),
                                                    self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = None  # Unused
        [self.frames.append(obs) for _ in range(self.num_stack)]

        if kwargs.get("return_info", False):
            return self.observation(), info
        else:
            return self.observation()


def resolve_lazy_frames(lazy_frames: LazyFrames) -> torch.Tensor:

    rlt = torch.stack(
        cast(
            List[torch.Tensor],
            [lazy_frames[i] for i in range(len(lazy_frames))],
        ))

    assert rlt.size(0) == len(lazy_frames)
    return rlt


class ToTensorEnv(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        (h, w, c) = env.observation_space.shape

        self.observation_space = Box(low=0,
                                     high=1,
                                     dtype=np.float32,
                                     shape=(c, h, w))

        self.transform = T.ToTensor()

    def observation(self, observation):
        observation = self.transform(observation)
        assert observation.shape == self.observation_space.shape
        return observation


class PreprocessObservation(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = Box(
            low=0,
            high=1,
            shape=(84, 84),
            dtype=np.float32,
        )

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((84, 84)),
            T.Grayscale(),
            T.ToTensor(),
            T.Lambda(lambda x: x.squeeze(0))
        ])

    def observation(self, observation):
        observation = self.transform(observation)
        assert observation.shape == self.observation_space.shape
        return observation


def touch(path):
    open(path, "a").close()


class VideoRecorder:
    """VideoRecorder renders a nice movie of a rollout, frame by frame. It
    comes with an `enabled` option so you can still use the same code
    on episodes where you don't want to record video.

    Note:
        You are responsible for calling `close` on a created
        VideoRecorder, or else you may leak an encoder process.

    Args:
        env (Env): Environment to take video of.
        path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
        base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.
        metadata (Optional[dict]): Contents to save to the metadata file.
        enabled (bool): Whether to actually record video, or just no-op (for convenience)
    """

    def __init__(self,
                 env,
                 path=None,
                 metadata=None,
                 enabled=True,
                 base_path=None):
        modes = ['human', 'rgb_array']

        self._async = env.metadata.get("semantics.async")
        self.enabled = enabled
        self._closed = False

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        self.ansi_mode = False
        if "rgb_array" not in modes:
            if "ansi" in modes:
                self.ansi_mode = True
            else:
                logger.info(
                    f'Disabling video recorder because {env} neither supports video mode "rgb_array" nor "ansi".'
                )
                # Whoops, turns out we shouldn't be enabled after all
                self.enabled = False
                return

        if path is not None and base_path is not None:
            raise error.Error(
                "You can pass at most one of `path` or `base_path`.")

        self.last_frame = None
        self.env = env

        required_ext = ".json" if self.ansi_mode else ".mp4"
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + required_ext
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(suffix=required_ext,
                                                 delete=False) as f:
                    path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            hint = (
                " HINT: The environment is text-only, therefore we're recording its text output in a structured JSON format."
                if self.ansi_mode else "")
            raise error.Error(
                f"Invalid path given: {self.path} -- must have file extension {required_ext}.{hint}"
            )
        # Touch the file in any case, so we know it's present. (This
        # corrects for platform platform differences. Using ffmpeg on
        # OS X, the file is precreated, but not on Linux.
        touch(path)

        self.frames_per_sec = env.metadata.get("render_fps", 30)
        self.output_frames_per_sec = env.metadata.get("render_fps",
                                                      self.frames_per_sec)

        # backward-compatibility mode:
        self.backward_compatible_frames_per_sec = env.metadata.get(
            "video.frames_per_second", 30)
        self.backward_compatible_output_frames_per_sec = env.metadata.get(
            "video.output_frames_per_second", self.frames_per_sec)
        if self.frames_per_sec != self.backward_compatible_frames_per_sec:
            logger.deprecation(
                '`env.metadata["video.frames_per_second"] is marked as deprecated and will be replaced with `env.metadata["render_fps"]` '
                "see https://github.com/openai/gym/pull/2654 for more details")
            self.frames_per_sec = self.backward_compatible_frames_per_sec
        if self.output_frames_per_sec != self.backward_compatible_output_frames_per_sec:
            logger.deprecation(
                '`env.metadata["video.output_frames_per_second"] is marked as deprecated and will be replaced with `env.metadata["render_fps"]` '
                "see https://github.com/openai/gym/pull/2654 for more details")
            self.output_frames_per_sec = self.backward_compatible_output_frames_per_sec

        self.encoder = None  # lazily start the process
        self.broken = False

        # Dump metadata
        self.metadata = metadata or {}
        self.metadata["content_type"] = ("video/vnd.openai.ansivid"
                                         if self.ansi_mode else "video/mp4")
        self.metadata_path = f"{path_base}.meta.json"
        self.write_metadata()

        logger.info("Starting new video recorder writing to %s", self.path)
        self.empty = True

    @property
    def functional(self):
        return self.enabled and not self.broken

    def capture_frame(self):
        """Render the given `env` and add the resulting frame to the video."""
        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return
        logger.debug("Capturing video frame: path=%s", self.path)

        render_mode = "ansi" if self.ansi_mode else "rgb_array"
        frame = self.env.render(mode=render_mode)

        if frame is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warn(
                    "Env returned None on render(). Disabling further rendering for video recorder by marking as disabled: path=%s metadata_path=%s",
                    self.path,
                    self.metadata_path,
                )
                self.broken = True
        else:
            self.last_frame = frame
            if self.ansi_mode:
                self._encode_ansi_frame(frame)
            else:
                self._encode_image_frame(frame)

    def close(self):
        """Flush all data to disk and close any open frame encoders."""
        if not self.enabled or self._closed:
            return

        if self.encoder:
            logger.debug("Closing video encoder: path=%s", self.path)
            self.encoder.close()
            self.encoder = None
        else:
            # No frames captured. Set metadata, and remove the empty output file.
            os.remove(self.path)

            if self.metadata is None:
                self.metadata = {}
            self.metadata["empty"] = True

        # If broken, get rid of the output file, otherwise we'd leak it.
        if self.broken:
            logger.info(
                "Cleaning up paths for broken video recorder: path=%s metadata_path=%s",
                self.path,
                self.metadata_path,
            )

            # Might have crashed before even starting the output file, don't try to remove in that case.
            if os.path.exists(self.path):
                os.remove(self.path)

            if self.metadata is None:
                self.metadata = {}
            self.metadata["broken"] = True

        self.write_metadata()

        # Stop tracking this for autoclose
        self._closed = True

    def write_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def __del__(self):
        # Make sure we've closed up shop when garbage collecting
        self.close()

    def _encode_ansi_frame(self, frame):
        if not self.encoder:
            self.encoder = TextEncoder(self.path, self.frames_per_sec)
            self.metadata["encoder_version"] = self.encoder.version_info
        self.encoder.capture_frame(frame)
        self.empty = False

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = ImageEncoder(self.path, frame.shape,
                                        self.frames_per_sec,
                                        self.output_frames_per_sec)
            self.metadata["encoder_version"] = self.encoder.version_info

        try:
            self.encoder.capture_frame(frame)
        except error.InvalidFrame as e:
            logger.warn(
                "Tried to pass invalid video frame, marking as broken: %s", e)
            self.broken = True
        else:
            self.empty = False


class TextEncoder:
    """Store a moving picture made out of ANSI frames. Format adapted from
    https://github.com/asciinema/asciinema/blob/master/doc/asciicast-v1.md"""

    def __init__(self, output_path, frames_per_sec):
        self.output_path = output_path
        self.frames_per_sec = frames_per_sec
        self.frames = []

    def capture_frame(self, frame):
        string = None
        if isinstance(frame, str):
            string = frame
        elif isinstance(frame, StringIO):
            string = frame.getvalue()
        else:
            raise error.InvalidFrame(
                f"Wrong type {type(frame)} for {frame}: text frame must be a string or StringIO"
            )

        frame_bytes = string.encode("utf-8")

        if frame_bytes[-1:] != b"\n":
            raise error.InvalidFrame(
                f'Frame must end with a newline: """{string}"""')

        if b"\r" in frame_bytes:
            raise error.InvalidFrame(
                f'Frame contains carriage returns (only newlines are allowed: """{string}"""'
            )

        self.frames.append(frame_bytes)

    def close(self):
        # frame_duration = float(1) / self.frames_per_sec
        frame_duration = 0.5

        # Turn frames into events: clear screen beforehand
        # https://rosettacode.org/wiki/Terminal_control/Clear_the_screen#Python
        # https://rosettacode.org/wiki/Terminal_control/Cursor_positioning#Python
        clear_code = b"%c[2J\033[1;1H" % (27)
        # Decode the bytes as UTF-8 since JSON may only contain UTF-8
        events = [(
            frame_duration,
            (clear_code + frame.replace(b"\n", b"\r\n")).decode("utf-8"),
        ) for frame in self.frames]

        # Calculate frame size from the largest frames.
        # Add some padding since we'll get cut off otherwise.
        height = max(frame.count(b"\n") for frame in self.frames) + 1
        width = (max(
            max(len(line) for line in frame.split(b"\n"))
            for frame in self.frames) + 2)

        data = {
            "version": 1,
            "width": width,
            "height": height,
            "duration": len(self.frames) * frame_duration,
            "command": "-",
            "title": "gym VideoRecorder episode",
            "env": {},  # could add some env metadata here
            "stdout": events,
        }

        with open(self.output_path, "w") as f:
            json.dump(data, f)

    @property
    def version_info(self):
        return {"backend": "TextEncoder", "version": 1}


class ImageEncoder:

    def __init__(self, output_path, frame_shape, frames_per_sec,
                 output_frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise error.InvalidFrame(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e., RGB values for a w-by-h image, with an optional alpha channel."
                .format(frame_shape))
        self.wh = (w, h)
        self.includes_alpha = pixfmt == 4
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec
        self.output_frames_per_sec = output_frames_per_sec

        if distutils.spawn.find_executable("avconv") is not None:
            self.backend = "avconv"
        elif distutils.spawn.find_executable("ffmpeg") is not None:
            self.backend = "ffmpeg"
        elif pkgutil.find_loader("imageio_ffmpeg"):
            import imageio_ffmpeg

            self.backend = imageio_ffmpeg.get_ffmpeg_exe()
        else:
            raise error.DependencyNotInstalled(
                """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`. Alternatively, please install imageio-ffmpeg with `pip install imageio-ffmpeg`"""
            )

        self.start()

    @property
    def version_info(self):
        return {
            "backend":
            self.backend,
            "version":
            str(
                subprocess.check_output([self.backend, "-version"],
                                        stderr=subprocess.STDOUT)),
            "cmdline":
            self.cmdline,
        }

    def start(self):
        self.cmdline = (
            self.backend,
            "-nostats",
            "-loglevel",
            "error",  # suppress warnings
            "-y",
            # input
            "-f",
            "rawvideo",
            "-s:v",
            "{}x{}".format(*self.wh),
            "-pix_fmt",
            ("rgb32" if self.includes_alpha else "rgb24"),
            "-framerate",
            "%d" % self.frames_per_sec,
            "-i",
            "-",  # this used to be /dev/stdin, which is not Windows-friendly
            # output
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "%d" % self.output_frames_per_sec,
            self.output_path,
        )

        logger.debug('Starting %s with "%s"', self.backend,
                     " ".join(self.cmdline))
        if hasattr(os, "setsid"):  # setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline,
                                         stdin=subprocess.PIPE,
                                         preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise error.InvalidFrame(
                f"Wrong type {type(frame)} for {frame} (must be np.ndarray or np.generic)"
            )
        if frame.shape != self.frame_shape:
            raise error.InvalidFrame(
                f"Your frame has shape {frame.shape}, but the VideoRecorder is configured for shape {self.frame_shape}."
            )
        if frame.dtype != np.uint8:
            raise error.InvalidFrame(
                f"Your frame has data type {frame.dtype}, but we require uint8 (i.e. RGB values from 0-255)."
            )

        try:
            if distutils.version.LooseVersion(
                    np.__version__) >= distutils.version.LooseVersion("1.9.0"):
                self.proc.stdin.write(frame.tobytes())
            else:
                self.proc.stdin.write(frame.tostring())
        except Exception as e:
            stdout, stderr = self.proc.communicate()
            logger.error("VideoRecorder encoder failed: %s", stderr)

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            logger.error(f"VideoRecorder encoder exited with status {ret}")


def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id**(1.0 / 3)))**3 == episode_id
    else:
        return episode_id % 1000 == 0


class RecordVideo(gym.Wrapper):

    def __init__(
        self,
        env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
    ):
        super().__init__(env)

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None
                            for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = 0

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        if not self.recording and self._video_enabled():
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={
                "step_id": self.step_id,
                "episode_id": self.episode_id
            },
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)

        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return observations, rewards, dones, infos

    def close_video_recorder(self) -> None:
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def close(self):
        self.close_video_recorder()

    def __del__(self):
        self.close_video_recorder()


class RescaleAction(gym.ActionWrapper):
    r"""Rescales the continuous action space of the environment to a range [min_action, max_action].

    Example::

        >>> RescaleAction(env, min_action, max_action).action_space == Box(min_action, max_action)
        True

    """

    def __init__(self, env, scale):
        assert scale > 0
        min_action = -scale
        max_action = scale
        assert isinstance(
            env.action_space, spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(min_action,
                             max_action).all(), (min_action, max_action)

        super().__init__(env)
        self.scale = scale
        self.min_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) +
            min_action)
        self.max_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) +
            max_action)
        self.action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        assert np.all(np.greater_equal(action, self.min_action)), (
            action,
            self.min_action,
        )
        assert np.all(np.less_equal(action,
                                    self.max_action)), (action,
                                                        self.max_action)
        action = self.scale * action
        action = np.clip(action, -self.scale, self.scale)
        return action
