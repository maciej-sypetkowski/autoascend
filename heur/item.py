import functools
import re

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

from glyph import WEA
import objects


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

    def get_dps(self, large_monster):
        assert self.is_weapon()

        weapon = objects.weapon_from_glyph(self.glyphs[0])
        dmg = WEA.expected_damage(weapon.damage_large if large_monster else weapon.damage_small)

        # TODO: take into account things from:
        # https://github.com/facebookresearch/nle/blob/master/src/weapon.c : hitval
        # https://github.com/facebookresearch/nle/blob/master/src/weapon.c : dmgval
        # https://github.com/facebookresearch/nle/blob/master/src/uhitm.c : find_roll_to_hit

        if self.modifier is not None and self.modifier > 0:
            dmg += self.modifier

        to_hit = 1
        to_hit += 6 # compensation, TODO: abon, etc
        to_hit += weapon.hitbon
        if self.modifier is not None:
            to_hit += self.modifier

        return np.array([to_hit > i for i in range(1, 21)]).astype(np.float32).mean().item() * dmg

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
        # TODO: aklys, Mjollnir
        return nh.objdescr.from_idx(nh.glyph_to_obj(self.glyphs[0])).oc_name in \
               ['orcish dagger', 'dagger silver', 'athame dagger', 'elven dagger',
                'worm tooth', 'knife', 'stiletto', 'scalpel', 'crysknife',
                'dart', 'shuriken']

    def __str__(self):
        if self.text is not None:
            return self.text
        return (f'{self.count}_'
                f'{self.status if self.status is not None else ""}_'
                f'{self.modifier if self.modifier is not None else ""}_'
                f'{",".join([nh.objdescr.from_idx(nh.glyph_to_obj(glyph)).oc_name for glyph in self.glyphs])}'
                )

    def __repr__(self):
        return str(self)


class ItemManager:
    def __init__(self, agent):
        self.agent = agent

    @functools.lru_cache(1024 * 256)
    def get_item_from_text(self, text, category=None, glyph=None):
        # TODO: there are some problems with 'inv_glyphs' and the environment gives incorrect inventory glyphs.
        #       I'm ignoring them for now
        glyph = None

        #assert category is not None or glyph is not None
        assert glyph is None or nh.glyph_is_normal_object(glyph)

        if category is None and glyph is not None:
            category = ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)
        assert glyph is None or category is None or category == ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)

        assert category not in [nh.BALL_CLASS, nh.RANDOM_CLASS]

        matches = re.findall(
            r'^(a|an|\d+)'
            r'( empty)?'
            r'( (cursed|uncursed|blessed))?'
            r'( (very |thoroughly )?(rustproof|poisoned|corroded|rusty|burnt|rotted|partly eaten|partly used))*'
            r'( ([+-]\d+))? '
            r'([a-zA-z0-9- ]+)'
            r'( \(([0-9]+:[0-9]+|no charge)\))?'
            r'( \(([a-zA-Z0-9; ]+)\))?'
            r'( \(for sale, (\d+) ?[a-zA-Z- ]+\))?'
            r'$',
            text)
        assert len(matches) <= 1, text
        assert len(matches), text

        count, _, effects1, status, effects2, _, _, _, modifier, name, _, uses, _, info, _, shop_price = matches[0]
        # TODO: effects, uses

        if info in {'weapon in paw', 'weapon in hand', 'weapon in paws', 'weapon in hands', 'being worn',
                    'being worn; slippery', 'wielded'}:
            equipped = True
            at_ready = False
        elif info in {'at the ready', 'in quiver', 'in quiver pouch', 'lit'}:
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

        if name == 'wakizashi':
            name = 'short sword'
        elif name == 'ninja-to':
            name = 'broadsword'
        elif name == 'nunchaku':
            name = 'flail'
        elif name == 'shito':
            name = 'knife'
        elif name == 'naginata':
            name = 'glaive'
        elif name == 'gunyoki':
            name = 'food ration'
        elif name == 'osaku':
            name = 'lock pick'
        elif name == 'tanko':
            name = 'plate mail'
        elif name == 'pair of yugake':
            name = 'pair of leather gloves'
        elif name == 'kabuto':
            name = 'helmet'
        elif name in ['potion of holy water', 'potions of holy water']:
            name = 'potion of water'
            status = Item.BLESSED
        elif name in ['potion of unholy water', 'potions of unholy water']:
            name = 'potion of water'
            status = Item.CURSED
        elif name in ['flint stone', 'flint stones']:
            name = 'flint'
        elif name in ['unlabeled scroll', 'unlabeled scrolls', 'blank paper']:
            name = 'scroll of blank paper'
        elif name == 'eucalyptus leaves':
            name = 'eucalyptus leaf'
        elif name == 'pair of lenses':
            name = 'lenses'
        elif name.startswith('small glob'):
            name = name[len('small '):]


        # TODO: pass to Item class instance
        if name.startswith('tin of ') or name.startswith('tins of '):
            name = 'tin'
        elif name.endswith(' corpse') or name.endswith(' corpses') or ' corpse named ' in name:
            name = 'corpse'
        elif name.startswith('statue of ') or name.startswith('statues of '):
            name = 'statue'
        elif name.startswith('figurine of ') or name.startswith('figurines of '):
            name = 'figurine'
        elif name.startswith('paperback book named ') or name.startswith('paperback books named ') or name in ['novel', 'paperback']:
            name = 'spellbook of novel'

        if ' named ' in name:
            # TODO: many of these are artifacts
            name = name[:name.index(' named ')]


        pressumed_category = None

        # object identified (look on names)
        ret_from_names = set()
        prefixes = [
            ('scroll of ', nh.SCROLL_CLASS),
            ('scrolls of ', nh.SCROLL_CLASS),
            ('spellbook of ', nh.SPBOOK_CLASS),
            ('spellbooks of ', nh.SPBOOK_CLASS),
            ('ring of ', nh.RING_CLASS),
            ('rings of ', nh.RING_CLASS),
            ('wand of ', nh.WAND_CLASS),
            ('wands of ', nh.WAND_CLASS),
            ('amulet of ', nh.AMULET_CLASS),
            ('amulets of ', nh.AMULET_CLASS),
            ('potion of ', nh.POTION_CLASS),
            ('potions of ', nh.POTION_CLASS),
            ('', nh.GEM_CLASS),
            ('', nh.ARMOR_CLASS),
            ('pair of ', nh.ARMOR_CLASS),
            ('', nh.WEAPON_CLASS),
            ('', nh.TOOL_CLASS),
            ('', nh.FOOD_CLASS),
            ('', nh.COIN_CLASS),
            ('', nh.ROCK_CLASS),
        ]
        suffixes = [
            ('s', nh.GEM_CLASS),
            ('s', nh.WEAPON_CLASS),
            ('s', nh.TOOL_CLASS),
            ('s', nh.FOOD_CLASS),
            ('s', nh.COIN_CLASS),
        ]
        for i in range(nh.NUM_OBJECTS):
            for pref, c in prefixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_name = nh.objdescr.from_idx(i).oc_name
                    if obj_name and name == pref + obj_name:
                        ret_from_names.add(i)

            for suf, c in suffixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_name = nh.objdescr.from_idx(i).oc_name
                    if obj_name and (name == obj_name + suf or \
                                     (c == nh.FOOD_CLASS and \
                                      name == obj_name.split()[0] + suf + ' ' + ' '.join(obj_name.split()[1:]))):
                        ret_from_names.add(i)


        # object unidentified (look on descriptions)
        ret_from_descriptions = set()
        prefixes = [
            ('scroll labeled ', nh.SCROLL_CLASS),
            ('scrolls labeled ', nh.SCROLL_CLASS),
            ('', nh.ARMOR_CLASS),
            ('pair of ', nh.ARMOR_CLASS),
            ('', nh.WEAPON_CLASS),
            ('', nh.TOOL_CLASS),
            ('', nh.FOOD_CLASS),
        ]
        suffixes = [
            (' amulet', nh.AMULET_CLASS),
            (' amulets', nh.AMULET_CLASS),
            (' gem', nh.GEM_CLASS),
            (' gems', nh.GEM_CLASS),
            (' stone', nh.GEM_CLASS),
            (' stones', nh.GEM_CLASS),
            (' potion', nh.POTION_CLASS),
            (' potions', nh.POTION_CLASS),
            (' spellbook', nh.SPBOOK_CLASS),
            (' spellbooks', nh.SPBOOK_CLASS),
            (' ring', nh.RING_CLASS),
            (' rings', nh.RING_CLASS),
            (' wand', nh.WAND_CLASS),
            (' wands', nh.WAND_CLASS),
            ('s', nh.ARMOR_CLASS),
            ('s', nh.WEAPON_CLASS),
            ('s', nh.TOOL_CLASS),
            ('s', nh.FOOD_CLASS),
        ]

        matches = set()
        for i in range(nh.NUM_OBJECTS):
            for pref, c in prefixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_descr = nh.objdescr.from_idx(i).oc_descr
                    if obj_descr and name == pref + obj_descr:
                        matches.add(i)

            for suf, c in suffixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_descr = nh.objdescr.from_idx(i).oc_descr
                    if obj_descr and name == obj_descr + suf:
                        matches.add(i)

        matches = list(matches)
        assert len(matches) == 0 or len({ord(nh.objclass(i).oc_class) for i in matches}), (text, name)

        if len(matches) > 0:
            pressumed_category = ord(nh.objclass(matches[0]).oc_class)

            if pressumed_category in [nh.AMULET_CLASS, nh.GEM_CLASS, nh.POTION_CLASS, nh.RING_CLASS, nh.SCROLL_CLASS, \
                                      nh.SPBOOK_CLASS, nh.WAND_CLASS]:
                # TODO: smart solving
                for i in range(nh.NUM_OBJECTS):
                    if ord(nh.objclass(i).oc_class) == pressumed_category:
                        ret_from_descriptions.add(i)
                assert len(ret_from_descriptions) > 0, text

            else:
                assert pressumed_category in [nh.ARMOR_CLASS, nh.WEAPON_CLASS, nh.FOOD_CLASS, nh.TOOL_CLASS], text
                ret_from_descriptions = ret_from_descriptions.union(matches)


        assert (len(ret_from_names) > 0) ^ (len(ret_from_descriptions) > 0), \
               (name, ret_from_names, ret_from_descriptions)
        if ret_from_names:
            assert len(ret_from_names) == 1, text
            ret = list(ret_from_names)
        else:
            ret = list(ret_from_descriptions)
            assert len(matches) > 0, (text, name)

        assert category is None or category == ord(nh.objclass(ret[0]).oc_class), (text, category, ord(nh.objclass(ret[0]).oc_class))
        return Item([r + nh.GLYPH_OBJ_OFF for r in ret], count, status, modifier, equipped, at_ready, text)


class Inventory:
    def __init__(self, agent):
        self.agent = agent
        self.item_manager = ItemManager(self)
        self.items = []
        self.letters = []

        self._previous_inv_strs = None
        self._previous_blstats = None
        self._stop_updating = False
        self.items_below_me = None
        self.letters_below_me = None

    def on_panic(self):
        self.items_below_me = None
        self.letters_below_me = None
        self._previous_blstats = None

    def update(self):
        if self._stop_updating:
            return

        if self._previous_blstats is None or \
                (self._previous_blstats.y, self._previous_blstats.x, \
                 self._previous_blstats.level_number, self._previous_blstats.dungeon_number) != \
                (self.agent.blstats.y, self.agent.blstats.x, \
                 self.agent.blstats.level_number, self.agent.blstats.dungeon_number):

            assume_appropriate_message = self._previous_blstats is not None

            self._previous_blstats = self.agent.blstats
            self.items_below_me = None
            self.letters_below_me = None

            try:
                self._stop_updating = True
                self.get_items_below_me(assume_appropriate_message=assume_appropriate_message)
            finally:
                self._stop_updating = False

        assert self.items_below_me is not None and self.letters_below_me is not None

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

        if item is not None and item.equipped:
            return True

        with self.agent.atom_operation():
            self.agent.step(A.Command.WIELD)
            if "Don't be ridiculous" in self.agent.message:
                return False
            assert 'What do you want to wield' in self.agent.message, self.agent.message
            self.agent.enter_text(letter)
            if 'You cannot wield a two-handed sword while wearing a shield.' in self.agent.message:
                # TODO: handle it better
                return False
            assert re.search(r'(You secure the tether\.  )?(^[a-zA-z] - |welds? itself to|You are already wielding that|'
                             r'You are already empty handed)', \
                             self.agent.message), self.agent.message

        return True

    def get_items_below_me(self, assume_appropriate_message=False):
        with self.agent.panic_if_position_changes():
            with self.agent.atom_operation():
                if not assume_appropriate_message:
                    self.agent.step(A.Command.LOOK)

                if 'Things that are here:' not in self.agent.popup:
                    if 'You see no objects here.' in self.agent.message:
                        items = []
                        letters = []
                    elif 'You see here ' in self.agent.message:
                        item_str = self.agent.message[self.agent.message.index('You see here ') + len('You see here '):]
                        item_str = item_str[:item_str.index('.')]
                        items = [self.item_manager.get_item_from_text(item_str)]
                        letters = [None]
                    else:
                        items = []
                        letters = []
                else:
                    self.agent.step(A.Command.PICKUP)
                    if 'You cannot reach the bottom of the pit.' in self.agent.message or \
                            'There is nothing here to pick up.' in self.agent.message:
                        items = []
                        letters = []
                    else:
                        assert 'Pick up what?' in self.agent.popup, (self.agent.popup, self.agent.message)
                        lines = self.agent.popup[self.agent.popup.index('Pick up what?') + 1:]
                        name_to_category = {
                            'Amulets': nh.AMULET_CLASS,
                            'Armor': nh.ARMOR_CLASS,
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
                            'Boulders/Statues': nh.ROCK_CLASS,
                        }
                        category = None
                        items = []
                        letters = []
                        for line in lines:
                            if line in name_to_category:
                                category = name_to_category[line]
                                continue
                            assert line[1:4] == ' - ', line
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
