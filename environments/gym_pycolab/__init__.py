# THIS FILE IS NEW OR MODIFIED COMPARED TO https://github.com/deepmind/pycolab
# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gym.envs.registration import register

tags = {'pycolab': True, 'wrapper_config.TimeLimit.max_episode_steps': 200}

for i in range(1, 21):
    register(
        id='grid-worlds-v{}'.format(i),
        entry_point='gym_pycolab.envs:PycolabGridWorldsLevel{}Env'.format(i),
        tags=tags
    )

for i in range(101, 104):
    register(
        id='grid-worlds-v{}'.format(i),
        entry_point='gym_pycolab.envs:PycolabGridWorldsLevel{}Env'.format(i),
        tags=tags
    )

register(
    id='connect-n-v1',
    entry_point='gym_pycolab.envs:PycolabConnect4Env',
    tags=tags
)

register(
    id='connect-n-v2',
    entry_point='gym_pycolab.envs:PycolabConnect4RotatedEnv',
    tags=tags
)

register(
    id='connect-n-v3',
    entry_point='gym_pycolab.envs:PycolabConnect5Env',
    tags=tags
)

register(
    id='connect-n-v4',
    entry_point='gym_pycolab.envs:PycolabConnect5RotatedEnv',
    tags=tags
)