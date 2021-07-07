import re

import nle.nethack as nh
from nle.nethack import actions as A

from glyph import WEA


class Item:
    UNKNOWN = 0
    CURSED = 1
    UNCURSED = 2
    BLESSED = 3

    def __init__(self, glyphs, count=1, status=UNKNOWN, modifier=None, equipped=False, at_ready=False, text=None):
        # glyphs is a list of possible glyphs for this item
        assert isinstance(glyphs, list)
        assert all(map(nh.glyph_is_object, glyphs))
        assert len(glyphs)

        self.glyphs = glyphs
        self.count = count
        self.status = status
        self.modifier = modifier
        self.equipped = equipped
        self.at_ready = at_ready
        self.text = text

        self.category = ord(nh.objclass(nh.glyph_to_obj(self.glyphs[0])).oc_class)

        assert all(map(lambda glyph: ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class) == self.category, self.glyphs))

    def base_cost(self):
        assert len(self.glyphs) == 1, 'TODO: what in this case?'
        return nh.objclass(nh.glyph_to_obj(self.glyphs[0])).oc_cost

    ######## WEAPON

    def is_weapon(self):
        assert self.category != nh.WEAPON_CLASS or len(self.glyphs) == 1, self.glyphs
        return self.category == nh.WEAPON_CLASS

    def get_dps(self, big_monster):
        assert self.is_weapon()
        return WEA.get_dps(self.glyphs[0], big_monster) + (self.modifier if self.modifier is not None else 0)

    def is_launcher(self):
        if not self.is_weapon():
            return False

        return nh.objdescr.from_idx(nh.glyph_to_obj(self.glyphs[0])).oc_name in \
               ['bow', 'elven bow', 'orcish bow', 'yumi', 'crossbow', 'sling']

    def is_fired_projectile(self, launcher=None):
        if not self.is_weapon():
            return False

        arrows = ['arrow', 'elven arrow', 'orcish arrow', 'silver arrow', 'ya']

        if launcher is None:
            return nh.objdescr.from_idx(nh.glyph_to_obj(self.glyphs[0])).oc_name in \
                   arrows + ['crossbow bolt']
        else:
            launcher_name = nh.objdescr.from_idx(nh.glyph_to_obj(launcher.glyphs[0])).oc_name
            if launcher_name == 'crossbow':
                return nh.objdescr.from_idx(nh.glyph_to_obj(self.glyphs[0])).oc_name == 'crossbow bolt'
            elif launcher_name == 'sling':
                # TODO: sling ammo
                return False
            else:  # any bow
                assert launcher_name in ['bow', 'elven bow', 'orcish bow', 'yumi']
                return nh.objdescr.from_idx(nh.glyph_to_obj(self.glyphs[0])).oc_name in arrows

    def is_thrown_projectile(self):
        if not self.is_weapon():
            return False

        # TODO: boomerang
        return nh.objdescr.from_idx(nh.glyph_to_obj(self.glyphs[0])).oc_name in \
               ['orcish dagger', 'dagger silver', 'athame dagger', 'elven dagger',
                'worm tooth', 'knife', 'stiletto', 'scalpel', 'crysknife',
                'dart', 'shuriken', ]

    def __str__(self):
        return (f'{self.count}_'
                f'{self.status if self.status is not None else ""}_'
                f'{self.modifier if self.modifier is not None else ""}_'
                f'{",".join([nh.objdescr.from_idx(nh.glyph_to_obj(glyph)).oc_name for glyph in self.glyphs])}'
                )


class ItemManager:
    def __init__(self, agent):
        self.agent = agent

    def get_item_from_text(self, text, category=None, glyph=None):
        # TODO: there are some problems with 'inv_glyphs' and the environment gives incorrect inventory glyphs.
        #       I'm ignoring them for now
        glyph = None

        assert category is not None or glyph is not None
        assert glyph is None or nh.glyph_is_normal_object(glyph)

        if category is None:
            category = ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)
        else:
            assert glyph is None or category == ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)

        assert category not in [nh.BALL_CLASS, nh.ROCK_CLASS, nh.RANDOM_CLASS]

        matches = re.findall(
            '^(a|an|\d+)( (cursed|uncursed|blessed))?( (very |thoroughly )?(rustproof|poisoned|corroded|rusty|burnt|rotted))*( ([+-]\d+))? ([a-zA-z0-9- ]+)( \(([0-9]+:[0-9]+)\))?( \(([a-zA-Z0-9; ]+)\))?$',
            text)
        assert len(matches) <= 1, text
        assert len(matches), text

        count, _, status, effects, _, _, _, modifier, name, _, uses, _, info = matches[0]
        # TODO: effects, uses

        if info in {'weapon in paw', 'weapon in hand', 'weapon in paws', 'weapon in hands', 'being worn',
                    'being worn; slippery', 'wielded'}:
            equipped = True
            at_ready = False
        elif info in {'at the ready', 'in quiver', 'in quiver pouch'}:
            equipped = False
            at_ready = True
        elif info in {'', 'alternate weapon; not wielded'}:
            equipped = False
            at_ready = False
        else:
            assert 0, info

        count = int({'a': 1, 'an': 1}.get(count, count))
        status = {'': Item.UNKNOWN, 'cursed': Item.CURSED, 'uncursed': Item.UNCURSED, 'blessed': Item.BLESSED}[status]
        modifier = None if not modifier else {'+': 1, '-': -1}[modifier[0]] * int(modifier[1:])

        if category == nh.WEAPON_CLASS:
            name_augmentation = lambda x: [x, f'{x}s']
        elif category == nh.ARMOR_CLASS:
            name_augmentation = lambda x: [x, f'pair of {x}']
        elif category in [nh.AMULET_CLASS, nh.FOOD_CLASS, nh.GEM_CLASS, nh.POTION_CLASS, nh.RING_CLASS,
                          nh.SCROLL_CLASS, nh.SPBOOK_CLASS, nh.TOOL_CLASS, nh.WAND_CLASS, nh.COIN_CLASS]:
            return Item(
                [i + nh.GLYPH_OBJ_OFF for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) == category and
                 nh.objdescr.from_idx(i).oc_name is not None],
                count, status, modifier, equipped, at_ready, text)
        else:
            assert 0, category

        ret = []

        if name == 'wakizashi':
            name = 'short sword'
        elif name == 'ninja-to':
            name = 'broadsword'
        elif name == 'nunchaku':
            name == 'flail'
        elif name == 'shito':
            name == 'knife'
        elif name == 'naginata':
            name == 'glaive'

        for i in range(nh.NUM_OBJECTS):
            if ord(nh.objclass(i).oc_class) != category:
                continue

            obj_descr = nh.objdescr.from_idx(i).oc_descr
            obj_name = nh.objdescr.from_idx(i).oc_name
            if (obj_name and name in name_augmentation(obj_name)) or (
                    obj_descr and name in name_augmentation(obj_descr)):
                ret.append(i)

        assert len(ret) == 1, (ret, name, text)
        assert glyph is None or ret[0] == nh.glyph_to_obj(glyph), \
            ((ret[0], nh.objdescr.from_idx(ret[0])),
             (nh.glyph_to_obj(glyph), nh.objdescr.from_idx(nh.glyph_to_obj(glyph))))
        return Item([r + nh.GLYPH_OBJ_OFF for r in ret], count, status, modifier, equipped, at_ready, text)


class Inventory:
    def __init__(self, agent):
        self.agent = agent
        self.item_manager = ItemManager(self)
        self.items = []
        self.letters = []

        self._previous_inv_strs = None
        self.items_below_me = []
        self.letters_below_me = []

    def on_panic(self):
        self.items_below_me = []
        self.letters_below_me = []

    def update(self):
        self.items_below_me = []
        self.letters_below_me = []

        if self._previous_inv_strs is not None and (self.agent.last_observation['inv_strs'] == self._previous_inv_strs).all():
            return

        self.items = []
        self.letters = []
        for item_name, category, glyph, letter in zip(
                self.agent.last_observation['inv_strs'],
                self.agent.last_observation['inv_oclasses'],
                self.agent.last_observation['inv_glyphs'],
                self.agent.last_observation['inv_letters']):
            item_name = bytes(item_name).decode().strip('\0')
            letter = chr(letter)
            if not item_name:
                continue
            item = self.item_manager.get_item_from_text(item_name, category=category, glyph=glyph)

            self.items.append(item)
            self.letters.append(letter)

        self._previous_inv_strs = self.agent.last_observation['inv_strs']

    def get_letter(self, item):
        assert item in self.items
        return self.letters[self.items.index(item)]

    ####### ACTIONS

    def wield(self, item):
        if item is None: # fists
            letter = '-'
        else:
            letter = self.get_letter(item)

        with self.agent.atom_operation():
            self.agent.step(A.Command.WIELD)
            assert 'What do you want to wield?' in self.agent.message
            self.agent.enter_text(letter)
            assert re.search('(^[a-zA-z] - |welds itself to)', self.agent.message), self.agent.message

        return True

    def get_items_below_me(self):
        with self.agent.atom_operation():
            self.agent.step(A.Command.PICKUP)
        assert bool(self.agent.popup) ^ bool(self.agent.message), (self.agent.popup, self.agent.message)

        if self.agent.message:
            raise NotImplementedError()
        else:
            assert self.agent.popup[0] == 'Pick up what?'
            lines = self.agent.popup[1:]
            name_to_category = {
                'Amulets': nh.AMULET_CLASS,
                'Armors': nh.ARMOR_CLASS,
                'Comestibles': nh.FOOD_CLASS,
                'Coins': nh.COIN_CLASS,
                'Gems/Stones': nh.GEM_CLASS,
                'Potions': nh.POTION_CLASS,
                'Rings': nh.RING_CLASS,
                'Scrolls': nh.SCROLL_CLASS,
                'Spellbooks': nh.SPBOOK_CLASS,
                'Tools': nh.TOOL_CLASS,
                'Weapons': nh.WEAPON_CLASS,
                'Wands': nh.WAND_CLASS,
            }
            category = None
            items = []
            letters = []
            for line in lines:
                if line in name_to_category:
                    category = name_to_category[line]
                    continue
                letter, line = line[0], line[4:]
                letters.append(letter)
                items.append(self.item_manager.get_item_from_text(line, category))

        self.items_below_me = items
        self.letters_below_me = letters
        return items

    def pickup(self, items):
        if isinstance(items, Item):
            items = [items]

        assert all(map(lambda item: item in self.items_below_me, items))

        with self.atom_operation():
            self.step(A.Command.PICKUP)
            self.step(A.Command.ESC)
        return True
