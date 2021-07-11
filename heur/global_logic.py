from enum import IntEnum, auto

import numpy as np

import utils
from glyph import Hunger, G
from level import Level
from strategy import Strategy


class Milestones(IntEnum):
    FIND_GNOMISH_MINES = auto()
    FIND_SOKOBAN = auto()
    GO_DOWN = auto() # TODO


class GlobalLogic:
    def __init__(self, agent):
        self.agent = agent
        self.milestone = Milestones(1)
        self.step_completion_log = {}  # Milestone -> (step, turn)

    # TODO: rething the situation with wizard's tower
    def level_dfs(self, v, path, target, vis):
        if v in vis:
            return

        if v == target:
            return path

        vis.add(v)
        try:
            for k, t in self.agent.get_level(*v).get_stairs(down=True, up=True, portal=True).items():
                if t is None:
                    continue
                glyph = self.agent.get_level(*v).objects[k]
                dir = '>' if glyph in G.STAIR_DOWN else '<' if glyph in G.STAIR_UP else ''
                assert dir, glyph # TODO: portals
                r = self.level_dfs(t[0], path + [(k, t, dir)], target, vis)
                if r is not None:
                    return r
        finally:
            vis.remove(v)

    def go_to_level(self, dungeon_number, level_number):
        path = self.level_dfs(self.agent.current_level().key(), [], (dungeon_number, level_number), set())
        assert path is not None

        with self.agent.env.debug_log(f'going to {(dungeon_number, level_number)} level'):
            for (y, x), _, dir in path:
                self.agent.explore1(None).until(self.agent, lambda: self.agent.bfs()[y, x] != -1).run()
                self.agent.go_to(y, x, debug_tiles_args=dict(color=(0, 0, 255), is_path=True))
                self.agent.move(dir)

    def get_unexplored_stairs(self, **kwargs):
        stairs = self.agent.current_level().get_stairs(**kwargs)
        return [k for k, v in stairs.items() if v is None]

    def gnomish_mines_entry(self):
        stairs = self.agent.get_level(Level.GNOMISH_MINES, 1).get_stairs(up=True)
        assert len(stairs) == 1
        return list(stairs.values())[0][0]

    @utils.debug_log('exploring stairs')
    @Strategy.wrap
    def check_unexplored_stairs_strategy(self, **kwargs):
        stairs = self.get_unexplored_stairs(**kwargs)
        if len(stairs) == 0:
            yield False
        yield True

        self.agent.explore1(None).until(self.agent,
            lambda: any(map(lambda pos: self.agent.bfs()[pos] != -1, self.get_unexplored_stairs(**kwargs)))).run()

        dis = self.agent.bfs()
        stairs = [pos for pos in self.get_unexplored_stairs(**kwargs) if dis[pos] != -1]
        assert len(stairs) > 0

        ty, tx = stairs[0]
        glyph = self.agent.current_level().objects[ty, tx]
        dir = '>' if glyph in G.STAIR_DOWN else '<' if glyph in G.STAIR_UP else ''
        assert dir # TODO: portals

        self.agent.go_to(ty, tx, debug_tiles_args=dict(color=(0, 0, 255), is_path=True))
        self.agent.move(dir)

    @Strategy.wrap
    def current_strategy(self):
        # TODO: to refactor

        if self.milestone == Milestones.FIND_GNOMISH_MINES:
            level = self.agent.current_level()
            if level.dungeon_number == Level.GNOMISH_MINES:
                self.milestone = Milestones(self.milestone + 1)
                yield False

            if level.dungeon_number == Level.DUNGEONS_OF_DOOM and level.level_number == 1:
                if self.check_unexplored_stairs_strategy(down=True).check_condition():
                    yield True
                    self.check_unexplored_stairs_strategy(down=True).run()
                    return
                if len(self.agent.current_level().get_stairs(down=True)) > 0:
                    yield True
                    self.go_to_level(Level.DUNGEONS_OF_DOOM, 2)
                    return
                yield False

            if level.dungeon_number == Level.DUNGEONS_OF_DOOM and level.level_number > 4:
                if self.check_unexplored_stairs_strategy(up=True).check_condition():
                    yield True
                    self.check_unexplored_stairs_strategy(up=True).run()
                    return
                if len(self.agent.current_level().get_stairs(up=True)) > 0:
                    yield True
                    self.go_to_level(Level.DUNGEONS_OF_DOOM, 4)
                    return
                yield False

            if level.dungeon_number == Level.DUNGEONS_OF_DOOM and 2 <= level.level_number <= 4:
                if self.check_unexplored_stairs_strategy(up=True, down=True).check_condition():
                    yield True
                    self.check_unexplored_stairs_strategy(up=True, down=True).run()
                    return

                if len(self.agent.current_level().get_stairs(down=True)) == 0 \
                        or len(self.agent.current_level().get_stairs(up=True)) == 0:
                    yield False


                for level_with_unexplored_staircase in range(2, 5):
                    if len(self.agent.get_level(Level.DUNGEONS_OF_DOOM,
                                                level_with_unexplored_staircase).get_stairs(down=True)) >= 2:
                        break
                else:
                    level_with_unexplored_staircase = None

                if level_with_unexplored_staircase is not None:
                    yield True
                    self.go_to_level(Level.DUNGEONS_OF_DOOM,
                            level.level_number - np.sign(level_with_unexplored_staircase - level.level_number))
                    return
                else:
                    if self.agent.rng.random() > 1 / 500:
                        yield False
                    yield True
                    self.go_to_level(Level.DUNGEONS_OF_DOOM,
                            self.agent.rng.choice([i for i in range(2, 5) if abs(i - level.level_number) <= 1]))
                    return

                assert 0

        elif self.milestone == Milestones.FIND_SOKOBAN:
            level = self.agent.current_level()
            if level.dungeon_number == Level.SOKOBAN:
                self.milestone = Milestones(self.milestone + 1)
                yield False

            if level.dungeon_number == Level.GNOMISH_MINES:
                if self.check_unexplored_stairs_strategy(up=True).check_condition():
                    yield True
                    self.check_unexplored_stairs_strategy(up=True).run()
                    return
                if len(self.agent.current_level().get_stairs(up=True)) > 0:
                    yield True
                    self.go_to_level(*self.gnomish_mines_entry())
                    return
                yield False

            if level.dungeon_number == Level.DUNGEONS_OF_DOOM and level.level_number < 6:
                if self.check_unexplored_stairs_strategy(down=True).check_condition():
                    yield True
                    self.check_unexplored_stairs_strategy(down=True).run()
                    return
                if len(self.agent.current_level().get_stairs(down=True)) > \
                        int(self.gnomish_mines_entry() == level.key()):
                    yield True
                    self.go_to_level(Level.DUNGEONS_OF_DOOM, level.level_number + 1)
                    return
                yield False

            if level.dungeon_number == Level.DUNGEONS_OF_DOOM and level.level_number > 10:
                if self.check_unexplored_stairs_strategy(up=True).check_condition():
                    yield True
                    self.check_unexplored_stairs_strategy(up=True).run()
                    return
                if len(self.agent.current_level().get_stairs(up=True)) > 0:
                    yield True
                    self.go_to_level(Level.DUNGEONS_OF_DOOM, 10)
                    return
                yield False

            if level.dungeon_number == Level.DUNGEONS_OF_DOOM and 6 <= level.level_number <= 10:
                if self.check_unexplored_stairs_strategy(up=True, down=True).check_condition():
                    yield True
                    self.check_unexplored_stairs_strategy(up=True, down=True).run()
                    return

                if len(self.agent.current_level().get_stairs(down=True)) == 0 \
                        or len(self.agent.current_level().get_stairs(up=True)) == 0:
                    yield False

                for level_with_unexplored_staircase in range(6, 11):
                    if len(self.agent.get_level(Level.DUNGEONS_OF_DOOM,
                                                level_with_unexplored_staircase).get_stairs(up=True)) >= 2:
                        break
                else:
                    level_with_unexplored_staircase = None

                if level_with_unexplored_staircase is not None:
                    yield True
                    self.go_to_level(Level.DUNGEONS_OF_DOOM,
                            level.level_number - np.sign(level_with_unexplored_staircase - level.level_number))
                    return
                else:
                    if self.agent.rng.random() > 1 / 500:
                        yield False
                    yield True
                    self.go_to_level(Level.DUNGEONS_OF_DOOM,
                            self.agent.rng.choice([i for i in range(6, 11) if abs(i - level.level_number) <= 1]))
                    return

            # assert 0, level.key()
            yield False

        elif self.milestone == Milestones.GO_DOWN:
            # TODO
            if self.check_unexplored_stairs_strategy(down=True).check_condition():
                yield True
                self.check_unexplored_stairs_strategy(down=True).run()
                return

            yield False

        assert 0, self.milestone


    def global_strategy(self):
        return \
            (self.agent.explore1(0).before(self.agent.explore1(None).preempt(self.agent, [
                self.current_strategy().condition(lambda:
                    self.agent.blstats.score > 850 + 100 * len(self.agent.levels) and \
                    self.agent.blstats.hitpoints >= 0.9 * self.agent.blstats.max_hitpoints)
            ], continue_after_preemption=False))).repeat().preempt(self.agent, [
                self.agent.eat1().condition(lambda: self.agent.blstats.time % 3 == 0 and \
                                                    self.agent.blstats.hunger_state >= Hunger.NOT_HUNGRY),
                self.agent.eat_from_inventory(),
            ]).preempt(self.agent, [
                self.agent.fight1(),
            ]).preempt(self.agent, [
                self.agent.emergency_strategy(),
            ])
