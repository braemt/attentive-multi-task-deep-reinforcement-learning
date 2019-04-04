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

REPAINT_MAPPING = {'X': chr(30),
                   'O': chr(65),
                   'M': chr(110),
                   ' ': chr(111)
                   }

INVERSE_REPAINT_MAPPING = {v: k for k, v in REPAINT_MAPPING.items()}

C_BACKGROUND = REPAINT_MAPPING[' ']
C_MANAGER = REPAINT_MAPPING['M']
C_PLAYER = REPAINT_MAPPING['X']
C_ENEMY = REPAINT_MAPPING['O']

FG_COLOURS = {k: (int(ord(v) * 999.0 / 127.0), int(ord(v) * 999.0 / 127.0), int(ord(v) * 999.0 / 127.0)) for k, v in
              REPAINT_MAPPING.items()}


def make_game(shape=[6, 7], connect_n=4, won_reward=1.0, lost_reward=-1.0):
    """Builds and returns the reward finding game."""
    raw_level = []
    for _ in range(shape[0]):
        raw_level.append(' ' * shape[1])

    level_art = []
    for _i in range(len(raw_level)):
        level_art.append([])
        for _j in range(len(raw_level[_i])):
            level_art[_i].append(REPAINT_MAPPING[raw_level[_i][_j]])

    shape = [len(raw_level), len(raw_level[0])]
    my_game_art = list(level_art)

    for i in range(shape[0]):
        my_game_art[i] = "".join(my_game_art[i])

    game = ascii_art.ascii_art_to_game(my_game_art, what_lies_beneath=C_BACKGROUND,
                                       sprites={},
                                       drapes={})

    game.add_sprite(C_MANAGER, (0, 0), GameManagerSprite, shape=shape, connect_n=connect_n, won_reward=won_reward,
                    lost_reward=lost_reward)
    game.add_drape(C_PLAYER, PlayerDrape)
    game.add_drape(C_ENEMY, EnemyDrape)
    return game


class GameManagerSprite(prefab_sprites.MazeWalker):
    """A `Sprite` for our player.
    """

    def __init__(self, corner, position, character, shape=[6, 7], connect_n=4, won_reward=1.0, lost_reward=-1.0):
        """Inform superclass that we can't walk through walls."""
        super(GameManagerSprite, self).__init__(
            corner, position, character, impassable='')
        self.shape = shape
        self.connect_n = connect_n
        self.won_reward = won_reward
        self.lost_reward = lost_reward
        self._teleport((-1, -1))
        self.moves = 0

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Apply motion commands.
        if actions == self.shape[1]:
            the_plot.terminate_episode()
            return
        if int(self.shape[1] + 1 if actions is None else actions) > self.shape[1]:
            return

        if self.correct_action(board, actions):
            pos = self.get_postion_for_actions(board, actions)
            the_plot['player'] = (pos, actions)
            board[pos][actions] = ord(C_PLAYER)
            self.moves += 1
            if self.game_over(board, the_plot['player']):
                the_plot.add_reward(self.won_reward)
                the_plot.terminate_episode()
                return
            elif self.tie():
                the_plot.terminate_episode()
                return

            enemy_actions = self.enemy_action(board)
            random.shuffle(enemy_actions)
            for a in enemy_actions:
                if self.correct_action(board, a):
                    pos = self.get_postion_for_actions(board, a)
                    the_plot['enemy'] = (pos, a)
                    board[pos][a] = ord(C_ENEMY)
                    self.moves += 1
                    if self.game_over(board, the_plot['enemy']):
                        the_plot.add_reward(self.lost_reward)
                        the_plot.terminate_episode()
                        return
                    elif self.tie():
                        the_plot.terminate_episode()
                        return
                    break
        else:
            the_plot.add_reward(self.lost_reward)
            the_plot.terminate_episode()
            return

    def enemy_action(self, board):
        return list(range(self.shape[1]))

    def tie(self):
        return self.moves >= self.shape[0] * self.shape[1]

    def game_over(self, board, position):
        seq = [1 + self.get_seq_length(board, position, -1, 0) + self.get_seq_length(board, position, 1, 0)]
        seq.append(1 + self.get_seq_length(board, position, 0, -1) + self.get_seq_length(board, position, 0, 1))
        seq.append(1 + self.get_seq_length(board, position, -1, 1) + self.get_seq_length(board, position, 1, -1))
        seq.append(1 + self.get_seq_length(board, position, -1, -1) + self.get_seq_length(board, position, 1, 1))

        if max(seq) >= self.connect_n:
            return True
        return False

    def get_seq_length(self, board, position, dx, dy):
        c = board[position[0]][position[1]]
        seq_length = 0
        px = position[0] + dx
        py = position[1] + dy
        while 0 <= px and px < self.shape[0] and 0 <= py and py < self.shape[1] and board[px][py] == c:
            px += dx
            py += dy
            seq_length += 1
        return seq_length

    def correct_action(self, board, actions):
        return board[0][actions] == ord(C_BACKGROUND)

    def get_postion_for_actions(self, board, actions):
        for i in range(len(board)):
            if board[len(board) - i - 1][actions] == ord(C_BACKGROUND):
                return len(board) - i - 1
        return None


class PlayerDrape(plab_things.Drape):
    """Drape for all positive rewards.
    """

    def __init__(self, curtain, character):
        super(PlayerDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if the_plot.get('player', False):
            self.curtain[the_plot['player']] = True


class EnemyDrape(plab_things.Drape):
    """Drape for all positive rewards.
    """

    def __init__(self, curtain, character):
        super(EnemyDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if the_plot.get('enemy', False):
            self.curtain[the_plot['enemy']] = True


def main(argv=()):
    game = make_game()

    repainter = rendering.ObservationCharacterRepainter(INVERSE_REPAINT_MAPPING)
    ui = human_ui.CursesUi(
        keys_to_actions={
            # Basic movement.
            '1': 0,
            '2': 1,
            '3': 2,
            '4': 3,
            '5': 4,
            '6': 5,
            '7': 6,
            'q': 7,
            -1: 8

        },
        repainter=repainter,
        delay=10)

    ui.play(game)


if __name__ == '__main__':
    main(sys.argv)
