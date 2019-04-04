# THIS FILE IS NEW OR MODIFIED COMPARED TO https://github.com/deepmind/pycolab
"""Find the reward."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import random
import sys

from gym_pycolab import ascii_art
from gym_pycolab import human_ui
from gym_pycolab import rendering
from gym_pycolab import things as plab_things
from gym_pycolab.prefab_parts import sprites as prefab_sprites

REPAINT_MAPPING = {'P': chr(127),
                   'T': chr(102),
                   'E': chr(89),
                   '+': chr(76),
                   '#': chr(51),
                   'D': chr(50),
                   'A': chr(38),
                   '-': chr(25),
                   ' ': chr(0)
                   }

INVERSE_REPAINT_MAPPING = {v: k for k, v in REPAINT_MAPPING.items()}



C_BACKGROUND = REPAINT_MAPPING[' ']
C_PLAYER = REPAINT_MAPPING['P']
C_TERMINAL_REWARD = REPAINT_MAPPING['T']
C_POSITIVE_REWARD = REPAINT_MAPPING['+']
C_NEGATIVE_REWARD = REPAINT_MAPPING['-']
C_WALL = REPAINT_MAPPING['#']
C_DOOR = REPAINT_MAPPING['D']
C_ACTIVATION = REPAINT_MAPPING['A']

FG_COLOURS = {k: (int(ord(v) * 999.0 / 127.0), int(ord(v) * 999.0 / 127.0), int(ord(v) * 999.0 / 127.0)) for k, v in
              REPAINT_MAPPING.items()}


def num_empty(level, shape):
    my_game_art = level
    empty = 0
    row = 0
    for i in range(shape[0]):
        row += 1
        col = 0
        for j in range(shape[1]):
            col += 1
            if my_game_art[i][j] == INVERSE_REPAINT_MAPPING[C_BACKGROUND]:
                empty += 1
        my_game_art[i] = list(my_game_art[i])
    return empty


def make_game(raw_level=None, terminal_reward=0.5, bonus_reward=0.5, per_step_cost=0.0, positive_rewards=0,
              negative_rewards=0, swap_rewards=False, confined_to_board=False, off_board_cost=0.5,
              wall_is_terminal=False, wall_cost=0.0, swap_actions=False, doors=False, round_world=False):
    """Builds and returns the reward finding game."""
    if raw_level is None:
        raw_level = ['        ',
                     '        ',
                     '        ',
                     '        ',
                     '        ',
                     '        ',
                     '        ',
                     '        ']
    level_art = []
    for _i in range(len(raw_level)):
        level_art.append([])
        for _j in range(len(raw_level[_i])):
            level_art[_i].append(REPAINT_MAPPING[raw_level[_i][_j]])

    shape = [len(raw_level), len(raw_level[0])]
    empty = num_empty(raw_level, shape)

    my_game_art = list(level_art)
    shape = [len(my_game_art), len(my_game_art[0])]
    random_placed_characters = [C_PLAYER]

    my_game_art, position_info = _add_characters(my_game_art, random_placed_characters, empty, C_BACKGROUND)

    my_game_art[7][6] = C_TERMINAL_REWARD
    if positive_rewards > 0:
        my_game_art[1][1] = C_POSITIVE_REWARD
    if positive_rewards > 1:
        my_game_art[4][2] = C_POSITIVE_REWARD
    if positive_rewards > 2:
        my_game_art[6][7] = C_POSITIVE_REWARD

    if negative_rewards > 0:
        my_game_art[2][5] = C_NEGATIVE_REWARD
    if negative_rewards > 1:
        my_game_art[4][6] = C_NEGATIVE_REWARD
    if negative_rewards > 2:
        my_game_art[6][1] = C_NEGATIVE_REWARD

    if doors:
        my_game_art[0][3] = C_ACTIVATION
        my_game_art[1][7] = C_ACTIVATION
        my_game_art[7][4] = C_ACTIVATION
        my_game_art[2][2] = C_DOOR
        my_game_art[2][3] = C_DOOR
        my_game_art[5][4] = C_DOOR
        my_game_art[5][5] = C_DOOR

    if round_world:
        my_game_art[2][2] = C_WALL
        my_game_art[2][3] = C_WALL
        my_game_art[5][4] = C_WALL
        my_game_art[5][5] = C_WALL

    for i in range(shape[0]):
        my_game_art[i] = "".join(my_game_art[i])

    game = ascii_art.ascii_art_to_game(my_game_art, what_lies_beneath=C_BACKGROUND,
                                       sprites={},
                                       drapes={C_TERMINAL_REWARD: TerminalRewardDrape,
                                               C_POSITIVE_REWARD: PositiveRewardDrape,
                                               C_NEGATIVE_REWARD: NegativeRewardDrape,
                                               C_DOOR: DoorDrape},
                                       update_schedule=[[],
                                                        [C_TERMINAL_REWARD, C_NEGATIVE_REWARD, C_POSITIVE_REWARD,
                                                         C_DOOR]],
                                       z_order=[C_NEGATIVE_REWARD, C_POSITIVE_REWARD, C_TERMINAL_REWARD, C_DOOR])
    game.add_sprite(C_PLAYER, position_info[C_PLAYER][0], PlayerSprite, shape=shape, terminal_reward=terminal_reward,
                    bonus_reward=bonus_reward, per_step_cost=per_step_cost, swap_rewards=swap_rewards,
                    confined_to_board=confined_to_board, off_board_cost=off_board_cost,
                    wall_is_terminal=wall_is_terminal, wall_cost=wall_cost, swap_actions=swap_actions, doors=doors,
                    round_world=round_world)
    return game

# may manipulate game_art directly
def _add_characters(game_art, characters, empty, what_lies_beneath):
    positions = sorted(random.sample(range(empty), len(characters)))
    ids = random.sample(range(len(characters)), len(characters))

    shape = [len(game_art), len(game_art[0])]
    position_info = {}
    for c in characters:
        position_info[c] = []
    pos = 0
    id = 0
    row = 0
    for i in range(shape[0]):
        row += 1
        col = 0
        for j in range(shape[1]):
            col += 1
            if game_art[i][j] == what_lies_beneath:
                if positions[id] == pos:
                    game_art[i] = list(game_art[i])  # prevent modifying global art
                    if characters[ids[id]] is not C_PLAYER:
                        game_art[i][j] = characters[ids[id]]
                    position_info[characters[ids[id]]].append((i, j))
                    id += 1
                    if id == len(ids):
                        return game_art, position_info
                pos += 1
    return game_art, position_info


class PlayerSprite(prefab_sprites.MazeWalker):
    """A `Sprite` for our player.
    """

    def __init__(self, corner, position, character, shape=None, terminal_reward=0.5, bonus_reward=0.5,
                 per_step_cost=0.0, swap_rewards=False, confined_to_board=False, off_board_cost=0.5,
                 wall_is_terminal=False, wall_cost=0.0, swap_actions=False, doors=False, round_world=False):
        """Inform superclass that we can't walk through walls."""
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable=[C_WALL, C_DOOR], confined_to_board=confined_to_board)
        self.shape = shape
        self.terminal_reward = terminal_reward
        self.bonus_reward = bonus_reward
        self.per_step_cost = per_step_cost
        self.swap_rewards = swap_rewards
        self.confined_to_board = confined_to_board
        self.off_board_cost = off_board_cost
        self.wall_is_terminal = wall_is_terminal
        self.wall_cost = wall_cost
        self.swap_actions = swap_actions
        self.doors = doors
        self.round_world = round_world

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Apply motion commands.
        intended_position = [self.position[0], self.position[1]]

        if self.swap_actions:
            if actions == 0:
                actions = 1
            elif actions == 1:
                actions = 0
            elif actions == 2:
                actions = 3
            elif actions == 3:
                actions = 2

        original_position = self.position

        if actions == 0:  # walk upward?
            self._north(board, the_plot)
            intended_position[0] -= 1
        elif actions == 1:  # walk downward?
            self._south(board, the_plot)
            intended_position[0] += 1
        elif actions == 2:  # walk leftward?
            self._west(board, the_plot)
            intended_position[1] -= 1
        elif actions == 3:  # walk rightward?
            self._east(board, the_plot)
            intended_position[1] += 1
        elif actions == 4:  # quit?
            the_plot.terminate_episode()
            return
        else:
            return

        if self.round_world:
            intended_position[0] = intended_position[0] % self.shape[0]
            intended_position[1] = intended_position[1] % self.shape[1]

        if intended_position[0] < 0 or intended_position[0] >= self.shape[0] or intended_position[1] < 0 or \
                intended_position[1] >= self.shape[1]:
            the_plot.add_reward(- self.off_board_cost)
            if not self.confined_to_board:
                the_plot.terminate_episode()
                return

        elif board[intended_position[0]][intended_position[1]] == ord(C_WALL) or board[intended_position[0]][
            intended_position[1]] == ord(C_DOOR):
            the_plot.add_reward(- self.wall_cost)
            if self.wall_is_terminal:
                the_plot.terminate_episode()
                return
            if self.round_world:
                self._teleport(original_position)

        elif self.round_world:
            self._teleport(intended_position)

        if layers[C_TERMINAL_REWARD][self.position]:
            the_plot.add_reward(self.terminal_reward)
            the_plot.terminate_episode()
            return

        if self.swap_rewards:
            factor = - 1.0
        else:
            factor = 1.0

        if layers[C_NEGATIVE_REWARD][self.position]:
            the_plot.add_reward(- self.bonus_reward * factor)

        if layers[C_POSITIVE_REWARD][self.position]:
            the_plot.add_reward(self.bonus_reward * factor)

        the_plot.add_reward(- self.per_step_cost)


class TerminalRewardDrape(plab_things.Drape):
    """Drape for all positive rewards.
    """

    def __init__(self, curtain, character):
        super(TerminalRewardDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        return


class PositiveRewardDrape(plab_things.Drape):
    """Drape for all positive rewards.
    """

    def __init__(self, curtain, character):
        super(PositiveRewardDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        self.curtain[things[C_PLAYER].position] = False


class NegativeRewardDrape(plab_things.Drape):
    """Drape for all negative rewards.
    """

    def __init__(self, curtain, character):
        super(NegativeRewardDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        self.curtain[things[C_PLAYER].position] = False


class DoorDrape(plab_things.Drape):
    """Drape for all doors.
    """

    def __init__(self, curtain, character):
        super(DoorDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # todo hardcoded
        if things[C_PLAYER].position == (0, 3) or things[C_PLAYER].position == (1, 7) or things[C_PLAYER].position == (
                7, 4):
            self.curtain[(2, 2)] = False
            self.curtain[(2, 3)] = False
            self.curtain[(5, 4)] = False
            self.curtain[(5, 5)] = False


def main(argv=()):
    WALL_LEVEL = ['    #   ',
                  '    #   ',
                  '##  #   ',
                  '        ',
                  '        ',
                  '   #  ##',
                  '   #    ',
                  '   #    ']
    game = make_game(raw_level=WALL_LEVEL, terminal_reward=1.0, doors=True)

    repainter = rendering.ObservationCharacterRepainter(INVERSE_REPAINT_MAPPING)
    ui = human_ui.CursesUi(
        keys_to_actions={
            # Basic movement.
            curses.KEY_UP: 0,
            curses.KEY_DOWN: 1,
            curses.KEY_LEFT: 2,
            curses.KEY_RIGHT: 3,
            'q': 4,
            -1: 5

        },
        repainter=repainter,
        delay=2000)

    ui.play(game)


if __name__ == '__main__':
    main(sys.argv)
