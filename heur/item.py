import contextlib
import functools
import re
from collections import namedtuple
from functools import partial

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

import objects as O
import utils
from character import Character
from exceptions import AgentPanic
from glyph import WEA, G, C, MON
from strategy import Strategy


class Item:
    # beatitude
    UNKNOWN = 0
    CURSED = 1
    UNCURSED = 2
    BLESSED = 3

    # shop status
    NOT_SHOP = 0
    FOR_SALE = 1
    UNPAID = 2

    def __init__(self, objs, glyphs, count=1, status=UNKNOWN, modifier=None, equipped=False, at_ready=False,
                 monster_id=None, shop_status=NOT_SHOP, price=0, dmg_bonus=None, to_hit_bonus=None,
                 naming='', comment='', text=None):
        assert isinstance(objs, list) and len(objs) >= 1
        assert isinstance(glyphs, list) and len(glyphs) >= 1 and all((nh.glyph_is_object(g) for g in glyphs))
        assert isinstance(count, int)

        self.objs = objs
        self.glyphs = glyphs
        self.count = count
        self.status = status
        self.modifier = modifier
        self.equipped = equipped
        self.at_ready = at_ready
        self.monster_id = monster_id
        self.shop_status = shop_status
        self.price = price
        self.dmg_bonus = dmg_bonus
        self.to_hit_bonus = to_hit_bonus
        self.naming = naming
        self.comment = comment
        self.text = text

        self.content = None  # for checked containers it will be set after the constructor
        self.container_id = None  # for containers and possible containers it will be set after the constructor

        self.category = O.get_category(self.objs[0])
        assert all((ord(nh.objclass(nh.glyph_to_obj(g)).oc_class) == self.category for g in self.glyphs))

    def display_glyphs(self):
        if self.is_corpse():
            assert self.monster_id is not None
            return [nh.GLYPH_BODY_OFF + self.monster_id]
        if self.is_statue():
            assert self.monster_id is not None
            return [nh.GLYPH_STATUE_OFF + self.monster_id]
        return self.glyphs

    def is_unambiguous(self):
        return len(self.objs) == 1

    def can_be_dropped_from_inventory(self):
        return not (
            (isinstance(self.objs[0], (O.Weapon, O.WepTool)) and self.status == Item.CURSED and self.equipped) or
            (isinstance(self.objs[0], O.Armor) and self.equipped) or
            (self.is_unambiguous() and self.object == O.from_name('loadstone') and self.status == Item.CURSED)
        )

    def weight(self):
        return self.count * self.unit_weight()

    def unit_weight(self):
        if self.is_corpse():
            return MON.permonst(self.monster_id).cwt

        if self.is_possible_container():
            return 100000

        if self.objs[0] in [
            O.from_name("glob of gray ooze"),
            O.from_name("glob of brown pudding"),
            O.from_name("glob of green slime"),
            O.from_name("glob of black pudding"),
        ]:
            assert self.is_unambiguous()
            return 10000  # weight is unknown

        weight = max((obj.wt for obj in self.objs))

        if self.is_container():
            weight += self.content.weight()  # TODO: bag of holding

        return weight

    @property
    def object(self):
        assert self.is_unambiguous()
        return self.objs[0]

    ######## WEAPON

    def is_weapon(self):
        return self.category == nh.WEAPON_CLASS

    def get_weapon_bonus(self, large_monster):
        assert self.is_weapon()

        hits = []
        dmgs = []
        for weapon in self.objs:
            dmg = WEA.expected_damage(weapon.damage_large if large_monster else weapon.damage_small)
            to_hit = 1 + weapon.hitbon
            if self.modifier is not None:
                dmg += max(0, self.modifier)
                to_hit += self.modifier

            dmg += 0 if self.dmg_bonus is None else self.dmg_bonus
            to_hit += 0 if self.to_hit_bonus is None else self.to_hit_bonus

            dmgs.append(dmg)
            hits.append(to_hit)

        # assume the worse
        return min(hits), min(dmgs)

    def is_launcher(self):
        if not self.is_weapon() or not self.is_unambiguous():
            return False

        return self.object.name in ['bow', 'elven bow', 'orcish bow', 'yumi', 'crossbow', 'sling']

    def is_fired_projectile(self, launcher=None):
        if not self.is_weapon() or not self.is_unambiguous():
            return False

        arrows = ['arrow', 'elven arrow', 'orcish arrow', 'silver arrow', 'ya']

        if launcher is None:
            return self.object.name in (arrows + ['crossbow bolt']) # TODO: sling ammo
        else:
            launcher_name = launcher.object.name
            if launcher_name == 'crossbow':
                return self.object.name == 'crossbow bolt'
            elif launcher_name == 'sling':
                # TODO: sling ammo
                return False
            else:  # any bow
                assert launcher_name in ['bow', 'elven bow', 'orcish bow', 'yumi'], launcher_name
                return self.object.name in arrows

    def is_thrown_projectile(self):
        if not self.is_weapon() or not self.is_unambiguous():
            return False

        # TODO: boomerang
        # TODO: aklys, Mjollnir
        return self.object.name in \
               ['dagger', 'orcish dagger', 'dagger silver', 'athame dagger', 'elven dagger',
                'worm tooth', 'knife', 'stiletto', 'scalpel', 'crysknife',
                'dart', 'shuriken']

    def __str__(self):
        if self.text is not None:
            return self.text
        return (f'{self.count}_'
                f'{self.status if self.status is not None else ""}_'
                f'{self.modifier if self.modifier is not None else ""}_'
                f'{",".join(list(map(lambda x: x.name, self.objs)))}'
                )

    def __repr__(self):
        return str(self)

    ######## ARMOR

    def is_armor(self):
        return self.category == nh.ARMOR_CLASS

    def get_ac(self):
        assert self.is_armor()
        return self.object.ac - (self.modifier if self.modifier is not None else 0)

    ######## FOOD

    def is_food(self):
        if isinstance(self.objs[0], O.Food):
            assert self.is_unambiguous()
            return True

    def nutrition_per_weight(self):
        # TODO: corpses/tins
        assert self.is_food()
        return self.object.nutrition / max(self.unit_weight(), 1)

    def is_corpse(self):
        if self.objs[0] == O.from_name('corpse'):
            assert self.is_unambiguous()
            return True
        return False

    ######## STATUE

    def is_statue(self):
        if self.objs[0] == O.from_name('statue'):
            assert self.is_unambiguous()
            return True
        return False

    ######## CONTAINER

    def is_container(self):
        # bag of tricks is not considered to be a container.
        # If the identifier doesn't exist yet, it's not consider a container
        return self.content is not None

    def is_possible_container(self):
        if self.is_container():
            return False

        if self.is_unambiguous() and self.object.name == 'bag of tricks':
            return False
        return any((isinstance(obj, O.Container) for obj in self.objs))

    def content(self):
        assert self.is_container()
        return self.content


class ContainerContent:
    def __init__(self):
        self.reset()

    def reset(self):
        self.items = []
        self.locked = False

    def __iter__(self):
        return iter(self.items)

    def weight(self):
        if self.locked:
            return 100000
        return sum((item.weight() for item in self.items))


class ItemManager:
    def __init__(self, agent):
        self.agent = agent
        self.object_to_glyph = {}
        self.glyph_to_object = {}
        self._last_object_glyph_mapping_update_step = None
        self._glyph_to_price_range = {}
        self._is_not_bag_of_tricks = set()

        # the container content should be edited instead of creating new one if exists.
        # Item.content keeps reference to it
        self.container_contents = {}  # container_id -> ContainerContent
        self._last_container_identifier = 0

    def on_panic(self):
        self.update_object_glyph_mapping()

    def update(self):
        if self._last_object_glyph_mapping_update_step is None or \
                self._last_object_glyph_mapping_update_step + 200 < self.agent.step_count:
            self.update_object_glyph_mapping()

    def update_object_glyph_mapping(self):
        with self.agent.atom_operation():
            self.agent.step(A.Command.KNOWN)
            for line in self.agent.popup:
                if line.startswith('*'):
                    assert line[1] == ' ' and line[-1] == ')' and line.count('(') == 1 and line.count(')') == 1, line
                    name = line[1 : line.find('(')].strip()
                    desc = line[line.find('(') + 1 : -1].strip()

                    n_objs, n_glyphs = ItemManager.parse_name(name)
                    d_glyphs = O.desc_to_glyphs(desc, O.get_category(n_objs[0]))
                    assert d_glyphs
                    if len(n_objs) == 1 and len(d_glyphs) == 1:
                        obj, glyph = n_objs[0], d_glyphs[0]

                        assert glyph not in self.glyph_to_object or self.glyph_to_object[glyph] == obj
                        self.glyph_to_object[glyph] = obj
                        assert obj not in self.object_to_glyph or self.object_to_glyph[obj] == glyph
                        self.object_to_glyph[obj] = glyph

            self._last_object_glyph_mapping_update_step = self.agent.step_count

    def _get_new_container_identifier(self):
        ret = self._last_container_identifier
        self._last_container_identifier += 1
        return str(ret)

    def _buy_price_identification(self):
        if self.agent.blstats.charisma <= 5:
            charisma_multiplier = 2
        elif self.agent.blstats.charisma <= 7:
            charisma_multiplier = 1.5
        elif self.agent.blstats.charisma <= 10:
            charisma_multiplier = 1 + 1 / 3
        elif self.agent.blstats.charisma <= 15:
            charisma_multiplier = 1
        elif self.agent.blstats.charisma <= 17:
            charisma_multiplier = 3 / 4
        elif self.agent.blstats.charisma <= 18:
            charisma_multiplier = 2 / 3
        else:
            charisma_multiplier = 1 / 2

        if (self.agent.character.role == Character.TOURIST and self.agent.blstats.experience_level < 15) or \
                (self.agent.inventory.items.shirt is not None and self.agent.inventory.items.suit is None and
                 self.agent.inventory.items.cloak is None) or \
                (self.agent.inventory.items.helm is not None and self.agent.inventory.items.helm.is_unambiguous() and
                 self.agent.inventory.items.helm.object == O.from_name('dunce cap')):
            dupe_multiplier = (4 / 3, 4 / 3)
        elif self.agent.inventory.items.helm is not None and \
                any(((obj == O.from_name('dunce cap')) for obj in self.agent.inventory.items.helm.objs)):
            dupe_multiplier = (4 / 3, 1)
        else:
            dupe_multiplier = (1, 1)

        for item in self.agent.inventory.items_below_me:
            if item.shop_status == Item.FOR_SALE and not item.is_unambiguous() and len(item.glyphs) == 1 and \
                    isinstance(item.objs[0], (O.Armor, O.Ring, O.Wand, O.Scroll, O.Potion, O.Tool)):
                # TODO: not an artifact
                # TODO: Base price of partly eaten food, uncursed water, and (x:-1) wands is 0.
                assert item.price % item.count == 0
                low = int(item.price / item.count / charisma_multiplier / (4 / 3) / dupe_multiplier[0])
                if isinstance(item.objs[0], (O.Weapon, O.Armor)):
                    low = 0  # +10 zorkmoids for each point of enchantment

                if low <= 5:
                    low = 0
                high = int(item.price / item.count / charisma_multiplier / dupe_multiplier[1]) + 1
                if item.glyphs[0] in self._glyph_to_price_range:
                    l, h = self._glyph_to_price_range[item.glyphs[0]]
                    low = max(low, l)
                    high = min(high, h)
                assert low <= high, (low, high)
                self._glyph_to_price_range[item.glyphs[0]] = (low, high)

                # update mapping for that object
                self.possible_objects_from_glyph(item.glyphs[0])

    def price_identification(self):
        if self.agent.character.prop.hallu:
            return
        if not self.agent.current_level().shop_interior[self.agent.blstats.y, self.agent.blstats.x]:
            return

        self._buy_price_identification()

    def update_possible_objects(self, item):
        possibilities_from_glyphs = set.union(*(set(self.possible_objects_from_glyph(glyph)) for glyph in item.glyphs))
        item.objs = [o for o in item.objs if o in possibilities_from_glyphs]
        assert len(item.objs)

    def get_item_from_text(self, text, category=None, glyph=None, *, position):
        # position acts as a container identifier if the container is not called. If the item is in inventory set it to None
        # TODO: when blind, it may not work as expected, e.g. "a shield", "a gem", "a potion", etc

        if self.agent.character.prop.hallu:
            glyph = None

        objs, glyphs, count, status, modifier, *args = \
                self.parse_text(text, category, glyph)
        category = O.get_category(objs[0])

        possibilities_from_glyphs = set.union(*(set(self.possible_objects_from_glyph(glyph)) for glyph in glyphs))
        objs = [o for o in objs if o in possibilities_from_glyphs]
        assert len(objs), ([O.objects[g - nh.GLYPH_OBJ_OFF].desc for g in glyphs], [o.name for o in possibilities_from_glyphs])

        if status == Item.UNKNOWN and (
                self.agent.character.role == Character.PRIEST or
                (modifier is not None and category not in [nh.ARMOR_CLASS, nh.RING_CLASS])):
            # TODO; oc_charged, amulets of yendor
            status = Item.UNCURSED

        old_objs = None
        old_glyphs = None
        while old_objs != objs or old_glyphs != glyphs:
            old_objs = objs
            old_glyphs = glyphs

            objs = [o for o in objs if o not in self.object_to_glyph or self.object_to_glyph[o] in glyphs]
            glyphs = [g for g in glyphs if g not in self.glyph_to_object or self.glyph_to_object[g] in objs]
            if len(objs) == 1 and len(glyphs) == 1:
                assert glyphs[0] not in self.glyph_to_object or self.glyph_to_object[glyphs[0]] == objs[0]
                self.glyph_to_object[glyphs[0]] = objs[0]
                assert objs[0] not in self.object_to_glyph or self.object_to_glyph[objs[0]] == glyphs[0]
                self.object_to_glyph[objs[0]] = glyphs[0]
            elif len(objs) == 1 and objs[0] in self.object_to_glyph:
                glyphs = [self.object_to_glyph[objs[0]]]
            elif len(glyphs) == 1 and glyphs[0] in self.glyph_to_object:
                objs = [self.glyph_to_object[glyphs[0]]]

        item = Item(objs, glyphs, count, status, modifier, *args, text)

        if item.is_possible_container() or item.is_container():
            if item.comment:
                identifier = item.comment
            else:
                if position is not None:
                    identifier = position
                else:
                    identifier = self._get_new_container_identifier()
            item.container_id = identifier
            if identifier in self.container_contents:
                item.content = self.container_contents[identifier]
                if len(item.glyphs) == 1 and item.glyphs[0] not in self._is_not_bag_of_tricks:
                    self._is_not_bag_of_tricks.add(item.glyphs[0])
                    self.update_possible_objects(item)

        return item

    def possible_objects_from_glyph(self, glyph):
        assert nh.glyph_is_object(glyph)
        if glyph in self.glyph_to_object:
            return [self.glyph_to_object[glyph]]
        objs = [obj for obj in O.possibilities_from_glyph(glyph)
                if obj not in self.object_to_glyph and
                   (glyph not in self._glyph_to_price_range or not hasattr(obj, 'cost') or
                    self._glyph_to_price_range[glyph][0] <= obj.cost <= self._glyph_to_price_range[glyph][1])
        ]
        objs = [obj for obj in objs if glyph not in self._is_not_bag_of_tricks or obj.name != 'bag of tricks']
        if len(objs) == 1:
            self.glyph_to_object[glyph] = objs[0]
            assert objs[0] not in self.object_to_glyph
            self.object_to_glyph[objs[0]] = glyph

            # update objects with have the same possible glyph
            for g in O.possible_glyphs_from_object(objs[0]):
                self.possible_objects_from_glyph(g)
        assert len(objs), (O.objects[glyph - nh.GLYPH_OBJ_OFF].desc, self._glyph_to_price_range[glyph])
        return objs

    @staticmethod
    @utils.copy_result
    @functools.lru_cache(1024 * 1024)
    def parse_text(text, category=None, glyph=None):
        assert glyph is None or nh.glyph_is_normal_object(glyph), glyph

        if category is None and glyph is not None:
            category = ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)
        assert glyph is None or category is None or category == ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)

        assert category not in [nh.BALL_CLASS, nh.RANDOM_CLASS]

        matches = re.findall(
            r'^(a|an|the|\d+)'
            r'( empty)?'
            r'( (cursed|uncursed|blessed))?'
            r'( (very |thoroughly )?(rustproof|poisoned|corroded|rusty|burnt|rotted|partly eaten|partly used|diluted|unlocked|locked))*'
            r'( ([+-]\d+))? '
            r"([a-zA-z0-9-!'# ]+)"
            r'( \(([0-9]+:[0-9]+|no charge)\))?'
            r'( \(([a-zA-Z0-9; ]+(, flickering)?[a-zA-Z0-9; ]+)\))?'
            r'( \((for sale|unpaid), (\d+ aum, )?((\d+)[a-zA-Z- ]+|no charge)\))?'
            r'$',
            text)
        assert len(matches) <= 1, text
        assert len(matches), text

        (
            count,
            _,
            effects1,
            status,
            effects2, _, _, _,
            modifier,
            name,
            _, uses, _,
            info, _, _,
            shop_status, _, _, shop_price
        ) = matches[0]
        # TODO: effects, uses

        if info in {'being worn', 'being worn; slippery', 'wielded'} or info.startswith('weapon in ') or \
                info.startswith('tethered weapon in '):
            equipped = True
            at_ready = False
        elif info in {'at the ready', 'in quiver', 'in quiver pouch', 'lit'}:
            equipped = False
            at_ready = True
        elif info in {'', 'alternate weapon; not wielded', 'alternate weapon; notwielded'}:
            equipped = False
            at_ready = False
        else:
            assert 0, info

        if not shop_price:
            shop_price = 0
        else:
            shop_price = int(shop_price)

        count = int({'a': 1, 'an': 1, 'the': 1}.get(count, count))
        status = {'': Item.UNKNOWN, 'cursed': Item.CURSED, 'uncursed': Item.UNCURSED, 'blessed': Item.BLESSED}[status]
        modifier = None if not modifier else {'+': 1, '-': -1}[modifier[0]] * int(modifier[1:])
        monster_id = None

        comment = ''
        naming = ''
        if ' named ' in name:
            # TODO: many of these are artifacts
            naming = name[name.index(' named ') + len(' named '):]
            name = name[:name.index(' named ')]
            if '#' in naming:
                # all given names by the bot, starts with #
                pos = naming.index('#')
                comment = naming[pos + 1:]
                naming = naming[:pos]
            else:
                comment = ''

        if shop_status == '':
            shop_status = Item.NOT_SHOP
        elif shop_status == 'for sale':
            shop_status = Item.FOR_SALE
        elif shop_status == 'unpaid':
            shop_status = Item.UNPAID
        else:
            assert 0, shop_status

        if name in ['potion of holy water', 'potions of holy water']:
            name = 'potion of water'
            status = Item.BLESSED
        elif name in ['potion of unholy water', 'potions of unholy water']:
            name = 'potion of water'
            status = Item.CURSED
        elif name in ['gold piece', 'gold pieces']:
            status = Item.UNCURSED

        # TODO: pass to Item class instance
        if name.startswith('tin of ') or name.startswith('tins of '):
            mon_name = name[len('tin of '):].strip()
            if mon_name.endswith(' meat'):
                mon_name = mon_name[:-len(' meat')]
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            if mon_name == 'spinach':
                monster_id = None
            else:
                monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'tin'
        elif name.endswith(' corpse') or name.endswith(' corpses'):
            mon_name = name[:name.index('corpse')].strip()
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'corpse'
        elif name.startswith('statue of ') or name.startswith('statues of ') or \
                name.startswith('historic statue of ') or name.startswith('historic statues of '):
            if name.startswith('historic'):
                mon_name = name[len('historic statue of '):].strip()
            else:
                mon_name = name[len('statue of '):].strip()
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'statue'
        elif name.startswith('figurine of ') or name.startswith('figurines of '):
            mon_name = name[len('figurine of '):].strip()
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'figurine'
        elif name in ['novel', 'paperback', 'paperback book']:
            name = 'spellbook of novel'
        elif name.endswith(' egg') or name.endswith(' eggs'):
            monster_id = nh.glyph_to_mon(MON.from_name(name[:-len(' egg')].strip()))
            name = 'egg'

        if ' containing ' in name:
            # TODO: use number of items for verification
            name = name[:name.index(' containing ')]

        dmg_bonus, to_hit_bonus = None, None

        if naming:
            if naming in ['Hachi', 'Idefix', 'Slasher', 'Sirius']:  # pet names
                pass
            elif name == 'corpse':
                pass
            elif name == 'spellbook of novel':
                pass
            else:
                name = naming

        if name == 'Excalibur':
            name = 'long sword'
            dmg_bonus = 5.5  # 1d10
            to_hit_bonus = 3  # 1d5
        elif name == 'Mjollnir':
            name = 'war hammer'
            dmg_bonus = 12.5  # 1d24
            to_hit_bonus = 3  # 1d5
        elif name == 'Cleaver':
            name = 'battle-axe'
            dmg_bonus = 3.5  # 1d6
            to_hit_bonus = 2  # 1d3
        elif name == 'Sting':
            name = 'elven dagger'
            dmg_bonus = 2.5  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Grimtooth':
            name = 'orcish dagger'
            dmg_bonus = 3.5  # +1d6
            to_hit_bonus = 1.5  # 1d2
        elif name in ['Sunsword', 'Frost Brand', 'Fire Brand', 'Demonbane', 'Giantslayer']:
            name = 'long sword'
            dmg_bonus = 5.5  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Vorpal Blade':
            name = 'long sword'
            dmg_bonus = 1
            to_hit_bonus = 3  # 1d5
        elif name == 'Orcrist':
            name = 'elven broadsword'
            dmg_bonus = 6  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Magicbane':
            name = 'athame'
            dmg_bonus = 7.25  # TODO
            to_hit_bonus = 2  # 1d3
        elif name == 'Grayswandir':
            name = 'silver saber'
            dmg_bonus = 15  # TODO: x2 + 1d20
            to_hit_bonus = 3  # 1d5
        elif name == 'Stormbringer':
            name = 'runesword'
            dmg_bonus = 6  # 1d2+1d8
            to_hit_bonus = 3  # 1d5
        elif name == 'Snickersnee':
            name = 'katana'
            dmg_bonus = 4.5  # 1d8
            to_hit_bonus = 1  # +1
        elif name == 'Trollsbane':
            name = 'morning star'
            dmg_bonus = 5.25  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Ogresmasher':
            name = 'war hammer'
            dmg_bonus = 3  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Dragonbane':
            name = 'broadsword'
            dmg_bonus = 4.75  # TODO: x2
            to_hit_bonus = 3  # 1d5

        objs, ret_glyphs = ItemManager.parse_name(name)
        assert category is None or category == O.get_category(objs[0]), (text, category, O.get_category(objs[0]))

        if glyph is not None:
            assert glyph in ret_glyphs
            pos = O.possibilities_from_glyph(glyph)
            if objs[0].name not in ['elven broadsword', 'runed broadsword']:
                assert all(map(lambda o: o in pos, objs)), (objs, pos)
            ret_glyphs = [glyph]
            objs = sorted(set(objs).intersection(O.possibilities_from_glyph(glyph)))

        return (
            objs, ret_glyphs, count, status, modifier, equipped, at_ready, monster_id, shop_status, shop_price,
            dmg_bonus, to_hit_bonus, naming, comment,
        )


    @staticmethod
    @utils.copy_result
    @functools.lru_cache(1024 * 256)
    def parse_name(name):
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
        elif name in ['pair of yugake', 'yugake']:
            name = 'pair of leather gloves'
        elif name == 'kabuto':
            name = 'helmet'
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
        elif name == 'knives':
            name = 'knife'

        # object identified (look on names)
        obj_ids = set()
        prefixes = [
            ('scroll of ', nh.SCROLL_CLASS),
            ('scrolls of ', nh.SCROLL_CLASS),
            ('spellbook of ', nh.SPBOOK_CLASS),
            ('spellbooks of ', nh.SPBOOK_CLASS),
            ('ring of ', nh.RING_CLASS),
            ('rings of ', nh.RING_CLASS),
            ('wand of ', nh.WAND_CLASS),
            ('wands of ', nh.WAND_CLASS),
            ('', nh.AMULET_CLASS),
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
                        obj_ids.add(i)

            for suf, c in suffixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_name = nh.objdescr.from_idx(i).oc_name
                    if obj_name and (name == obj_name + suf or \
                                     (c == nh.FOOD_CLASS and \
                                      name == obj_name.split()[0] + suf + ' ' + ' '.join(obj_name.split()[1:]))):
                        obj_ids.add(i)


        # object unidentified (look on descriptions)
        appearance_ids = set()
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

        for i in range(nh.NUM_OBJECTS):
            for pref, c in prefixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_descr = nh.objdescr.from_idx(i).oc_descr
                    if obj_descr and name == pref + obj_descr:
                        appearance_ids.add(i)

            for suf, c in suffixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_descr = nh.objdescr.from_idx(i).oc_descr
                    if obj_descr and name == obj_descr + suf:
                        appearance_ids.add(i)

        appearance_ids = list(appearance_ids)
        assert len(appearance_ids) == 0 or len({ord(nh.objclass(i).oc_class) for i in appearance_ids}), name

        assert (len(obj_ids) > 0) ^ (len(appearance_ids) > 0), (name, obj_ids, appearance_ids)

        if obj_ids:
            assert len(obj_ids) == 1, name
            obj_id = list(obj_ids)[0]
            objs = [O.objects[obj_id]]
            glyphs = [i for i in range(nh.GLYPH_OBJ_OFF, nh.NUM_OBJECTS + nh.GLYPH_OBJ_OFF)
                      if O.objects[i - nh.GLYPH_OBJ_OFF] is not None and objs[0] in O.possibilities_from_glyph(i)]
        else:
            glyphs = [obj_id + nh.GLYPH_OBJ_OFF for obj_id in appearance_ids]
            obj_id = list(appearance_ids)[0]
            glyph = obj_id + nh.GLYPH_OBJ_OFF
            objs = sorted(set.union(*[set(O.possibilities_from_glyph(i)) for i in glyphs]))
            assert name == 'runed broadsword' or \
                   all(map(lambda i: sorted(O.possibilities_from_glyph(i + nh.GLYPH_OBJ_OFF)) == objs, appearance_ids)), \
                   name

        return objs, glyphs


class InventoryItems:
    def __init__(self, agent):
        self.agent = agent
        self._previous_inv_strs = None

        self._clear()

    def _clear(self):
        self.main_hand = None
        self.off_hand = None
        self.suit = None
        self.helm = None
        self.gloves = None
        self.boots = None
        self.cloak = None
        self.shirt = None

        self.total_weight = 0

        self.all_items = []
        self.all_letters = []

        self._recheck_containers = True

    def __iter__(self):
        return iter(self.all_items)

    def __str__(self):
        return (
            f'main_hand: {self.main_hand}\n'
            f'off_hand : {self.off_hand}\n'
            f'suit     : {self.suit}\n'
            f'helm     : {self.helm}\n'
            f'gloves   : {self.gloves}\n'
            f'boots    : {self.boots}\n'
            f'cloak    : {self.cloak}\n'
            f'shirt    : {self.shirt}\n'
            f'Items:\n' +
            '\n'.join([f' {l} - {i}' for l, i in zip(self.all_letters, self.all_items)])
        )

    def total_nutrition(self):
        ret = 0
        for item in self:
            if item.is_food():
                ret += item.object.nutrition * item.count
        return ret

    def on_panic(self):
        self._recheck_containers = True

    def update(self):
        if not (self._previous_inv_strs is not None and (self.agent.last_observation['inv_strs'] == self._previous_inv_strs).all()):
            self._clear()

            for item_name, category, glyph, letter in zip(
                    self.agent.last_observation['inv_strs'],
                    self.agent.last_observation['inv_oclasses'],
                    self.agent.last_observation['inv_glyphs'],
                    self.agent.last_observation['inv_letters']):
                item_name = bytes(item_name).decode().strip('\0')
                letter = chr(letter)
                if not item_name:
                    continue
                item = self.agent.inventory.item_manager.get_item_from_text(item_name, category=category,
                        glyph=glyph if not nh.glyph_is_body(glyph) and not nh.glyph_is_statue(glyph) else None,
                        position=None)

                self.all_items.append(item)
                self.all_letters.append(letter)

                if item.equipped:
                    for types, sub, name in [
                        ((O.Weapon, O.WepTool), None,         'main_hand'),
                        (O.Armor,               O.ARM_SHIELD, 'off_hand'), # TODO: twoweapon support
                        (O.Armor,               O.ARM_SUIT,   'suit'),
                        (O.Armor,               O.ARM_HELM,   'helm'),
                        (O.Armor,               O.ARM_GLOVES, 'gloves'),
                        (O.Armor,               O.ARM_BOOTS,  'boots'),
                        (O.Armor,               O.ARM_CLOAK,  'cloak'),
                        (O.Armor,               O.ARM_SHIRT,  'shirt'),
                    ]:
                        if isinstance(item.objs[0], types) and (sub is None or sub == item.objs[0].sub):
                            assert getattr(self, name) is None
                            setattr(self, name, item)
                            break

                if item.is_possible_container() or (item.is_container() and self._recheck_containers):
                    self.agent.inventory.check_container_content(item)

                self.total_weight += item.weight()
                # weight is sometimes unambiguous for unidentified items. All exceptions:
                # {'helmet': 30, 'helm of brilliance': 50, 'helm of opposite alignment': 50, 'helm of telepathy': 50}
                # {'leather gloves': 10, 'gauntlets of fumbling': 10, 'gauntlets of power': 30, 'gauntlets of dexterity': 10}
                # {'speed boots': 20, 'water walking boots': 15, 'jumping boots': 20, 'elven boots': 15, 'fumble boots': 20, 'levitation boots': 15}
                # {'luckstone': 10, 'loadstone': 500, 'touchstone': 10, 'flint': 10}

            self._previous_inv_strs = self.agent.last_observation['inv_strs']
            self._recheck_containers = False

    def get_letter(self, item):
        assert item in self.all_items, (item, self.all_items)
        return self.all_letters[self.all_items.index(item)]


class ItemPriorityBase:
    def split(self, items, forced_items, weight_capacity):
        '''
        returns a list of counts to take corresponding to `items`

        Order of `items` matter. First items are more important.
        Otherwise the agent will drop and pickup items repeatedly.

        The function should motonic (i.e. removing an item from the argument,
        shouldn't decrease counts of other items). Otherwise the agent will
        go to the item, don't take it, and repeat infinitely
        '''
        raise NotImplementedError()


class Inventory:
    def __init__(self, agent):
        self.agent = agent
        self.item_manager = ItemManager(self.agent)
        self.items = InventoryItems(self.agent)

        self._previous_blstats = None
        self.items_below_me = None
        self.letters_below_me = None

    def on_panic(self):
        self.items_below_me = None
        self.letters_below_me = None
        self._previous_blstats = None

        self.item_manager.on_panic()

    def update(self):
        self.item_manager.update()
        self.items.update()

        if self._previous_blstats is None or \
                (self._previous_blstats.y, self._previous_blstats.x, \
                 self._previous_blstats.level_number, self._previous_blstats.dungeon_number) != \
                (self.agent.blstats.y, self.agent.blstats.x, \
                 self.agent.blstats.level_number, self.agent.blstats.dungeon_number):

            assume_appropriate_message = self._previous_blstats is not None

            self._previous_blstats = self.agent.blstats
            self.items_below_me = None
            self.letters_below_me = None

            self.get_items_below_me(assume_appropriate_message=assume_appropriate_message)

        assert self.items_below_me is not None and self.letters_below_me is not None

    @contextlib.contextmanager
    def panic_if_items_below_me_change(self):
        old_items_below_me = self.items_below_me
        old_letters_below_me = self.letters_below_me

        def f(self):
            if (
                [(l, i.text) for i, l in zip(old_items_below_me, old_letters_below_me)] !=
                [(l, i.text) for i, l in zip(self.items_below_me, self.letters_below_me)]
            ):
                raise AgentPanic('items below me changed')

        fun = partial(f, self)

        self.agent.on_update.append(fun)

        try:
            yield
        finally:
            assert fun in self.agent.on_update
            self.agent.on_update.pop(self.agent.on_update.index(fun))

    ####### ACTIONS

    def wield(self, item):
        if item is None: # fists
            letter = '-'
        else:
            letter = self.items.get_letter(item)

        if item is not None and item.equipped:
            return True

        if self.agent.character.prop.polymorph:
            # TODO: depends on kind of a monster
            return False

        if (self.items.main_hand is not None and self.items.main_hand.status == Item.CURSED) or \
                (item is not None and item.objs[0].bi and self.items.off_hand is not None):
            return False

        with self.agent.atom_operation():
            self.agent.step(A.Command.WIELD)
            if "Don't be ridiculous" in self.agent.message:
                return False
            assert 'What do you want to wield' in self.agent.message, self.agent.message
            self.agent.type_text(letter)
            if 'You cannot wield a two-handed sword while wearing a shield.' in self.agent.message or \
                    'You cannot wield a two-handed weapon while wearing a shield.' in self.agent.message or \
                    ' welded to your hand' in self.agent.message:
                return False
            assert re.search(r'(You secure the tether\.  )?([a-zA-z] - |welds?( itself| themselves| ) to|'
                             r'You are already wielding that|You are empty handed|You are already empty handed)', \
                             self.agent.message), (self.agent.message, self.agent.popup)

        return True

    def wear(self, item):
        assert item is not None
        letter = self.items.get_letter(item)

        if item.equipped:
            return True

        for i in self.items:
            assert not isinstance(i, O.Armor) or i.sub != item.sub or not i.equipped, (i, item)

        with self.agent.atom_operation():
            self.agent.step(A.Command.WEAR)
            if "Don't even bother." in self.agent.message:
                return False
            assert 'What do you want to wear?' in self.agent.message, self.agent.message
            self.agent.type_text(letter)
            assert 'You finish your dressing maneuver.' in self.agent.message or \
                   'You are now wearing ' in self.agent.message, self.agent.message

        return True

    def takeoff(self, item):
        assert item is not None and item.equipped, item
        letter = self.items.get_letter(item)
        assert item.status != Item.CURSED, item

        equipped_armors = [i for i in self.items if i.is_armor() and i.equipped]
        assert item in equipped_armors

        with self.agent.atom_operation():
            self.agent.step(A.Command.TAKEOFF)

            if len(equipped_armors) > 1:
                assert 'What do you want to take off?' in self.agent.message, self.agent.message
                self.agent.type_text(letter)
            if 'It is cursed.' in self.agent.message or 'They are cursed.' in self.agent.message:
                return False
            assert 'You finish taking off ' in self.agent.message or \
                   'You were wearing ' in self.agent.message or \
                   'You feel that monsters no longer have difficulty pinpointing your location.' in self.agent.message \
                   , self.agent.message

        return True

    def check_container_content(self, item):
        assert not self.agent.character.prop.polymorph  # TODO: only handless
        assert item.is_possible_container() or item.is_container()
        assert item in self.items.all_items or item in self.items_below_me

        is_bag_of_tricks = False
        if item.content is not None:
            content = item.content
            content.reset()
        else:
            content = ContainerContent()

        def gen():
            nonlocal content, is_bag_of_tricks

            if 'You carefully open ' in self.agent.message or 'You open ' in self.agent.message:
                yield ' '

            if 'It develops a huge set of teeth and bites you!' in self.agent.message:
                is_bag_of_tricks = True
                return

            if 'Hmmm, it turns out to be locked.' in self.agent.message or 'It is locked.' in self.agent.message:
                content.locked = True
                return

            yield ':'

            if ' is empty' in self.agent.message:
                return

            if self.agent.popup and 'Contents of ' in self.agent.popup[0]:
                for text in self.agent.popup[1:]:
                    if not text:
                        continue
                    content.items.append(self.item_manager.get_item_from_text(text, position=None))
                return

            assert 0, (self.agent.message, self.agent.popup)

        with self.agent.atom_operation():
            if item in self.items.all_items:
                self.agent.step(A.Command.APPLY)
                self.agent.step(self.items.get_letter(item), gen())
            else:
                self.agent.step(A.Command.LOOT)
                while self.agent.message != f'There is {item.text} here, loot it? [ynq] (q)':
                    assert 'There is ' in self.agent.message and ', loot it?' in self.agent.message, self.agent.message
                    self.agent.step('n')
                self.agent.step('y', gen())

            if is_bag_of_tricks:
                assert item.content is None
                raise AgentPanic('bag of tricks bites')

            if item in self.items.all_items and item.comment != item.container_id:
                self.call_item(item, item.container_id)

            if item.content is None:
                assert item.container_id is not None
                assert item.container_id not in self.item_manager.container_contents
                self.item_manager.container_contents[item.container_id] = content
                item.content = content

            # TODO: make it more elegant
            if len(item.glyphs) == 1 and item.glyphs[0] not in self.item_manager._is_not_bag_of_tricks:
                self.item_manager._is_not_bag_of_tricks.add(item.glyphs[0])
                self.item_manager.update_possible_objects(item)

    def get_items_below_me(self, assume_appropriate_message=False):
        with self.agent.panic_if_position_changes():
            with self.agent.atom_operation():
                if not assume_appropriate_message:
                    self.agent.step(A.Command.LOOK)
                elif 'Things that are here:' in self.agent.popup or \
                        re.search('There are (several|many) objects here\.', self.agent.message):
                    # LOOK is necessary even when 'Things that are here' popup is present for some very rare cases
                    self.agent.step(A.Command.LOOK)

                if 'Things that are here:' not in self.agent.popup and 'There is ' not in '\n'.join(self.agent.popup):
                    if 'You see no objects here.' in self.agent.message:
                        items = []
                        letters = []
                    elif 'You see here ' in self.agent.message:
                        item_str = self.agent.message[self.agent.message.index('You see here ') + len('You see here '):]
                        item_str = item_str[:item_str.index('.')]
                        items = [self.item_manager.get_item_from_text(item_str,
                            position=(*self.agent.current_level().key(), self.agent.blstats.y, self.agent.blstats.x))]
                        letters = [None]
                    else:
                        items = []
                        letters = []
                else:
                    self.agent.step(A.Command.PICKUP) # FIXME: parse LOOK output, add this fragment to pickup method
                    if 'Pick up what?' not in self.agent.popup:
                        if 'You cannot reach the bottom of the pit.' in self.agent.message or \
                                'You cannot reach the floor.' in self.agent.message or \
                                'There is nothing here to pick up.' in self.agent.message or \
                                ' solidly fixed to the floor.' in self.agent.message or \
                                'You read:' in self.agent.message or \
                                "You don't see anything in here to pick up." in self.agent.message or \
                                'You cannot reach the ground.' in self.agent.message:
                            items = []
                            letters = []
                        else:
                            assert 0, (self.agent.message, self.agent.popup)
                    else:
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
                            items.append(self.item_manager.get_item_from_text(line, category,
                                position=(*self.agent.current_level().key(), self.agent.blstats.y, self.agent.blstats.x)))

                self.items_below_me = items
                self.letters_below_me = letters
                return items

    def pickup(self, items, counts=None):
        # TODO: if polyphormed, sometimes 'You are physically incapable of picking anything up.'
        if isinstance(items, Item):
            items = [items]
            if counts is not None:
                counts = [counts]
        if counts is None:
            counts = [i.count for i in items]
        assert len(items) > 0
        assert all(map(lambda item: item in self.items_below_me, items))
        assert len(counts) == len(items)
        assert sum(counts) > 0 and all((0 <= c <= i.count for c, i in zip(counts, items)))

        letters = [self.letters_below_me[self.items_below_me.index(item)] for item in items]
        screens = [max(self.letters_below_me[:self.items_below_me.index(item) + 1].count('a') - 1, 0) for item in items]

        with self.panic_if_items_below_me_change():
            self.get_items_below_me()

        one_item = len(self.items_below_me) == 1
        with self.agent.atom_operation():
            if one_item:
                assert all((s in [0, None] for s in screens))
                self.agent.step(A.Command.PICKUP)
                drop_count = items[0].count - counts[0]
            else:
                text = ' '.join((
                    ''.join([(str(count) if item.count != count else '') + letter
                             for letter, item, count, screen in zip(letters, items, counts, screens)
                             if count != 0 and screen == current_screen])
                    for current_screen in range(max(screens) + 1)))
                self.agent.step(A.Command.PICKUP, iter(list(text) + [A.MiscAction.MORE]))

            if re.search('You have [a-z ]+ lifting ', self.agent.message) and \
                    'Continue?' in self.agent.message:
                self.agent.type_text('y')
            if one_item and drop_count:
                letter = re.search(r'([a-zA-Z$]) - ', self.agent.message)
                assert letter is not None, self.agent.message
                letter = letter[1]

        if one_item and drop_count:
            self.drop(self.items.all_items[self.items.all_letters.index(letter)], drop_count)

        self.get_items_below_me()

        return True

    def drop(self, items, counts=None):
        if isinstance(items, Item):
            items = [items]
            if counts is not None:
                counts = [counts]
        if counts is None:
            counts = [i.count for i in items]
        assert all(map(lambda x: isinstance(x, (int, np.int32, np.int64)), counts)), list(map(type, counts))
        assert len(items) > 0
        assert all(map(lambda item: item in self.items.all_items, items))
        assert len(counts) == len(items)
        assert sum(counts) > 0 and all((0 <= c <= i.count for c, i in zip(counts, items)))

        letters = [self.items.all_letters[self.items.all_items.index(item)] for item in items]
        texts_to_type = [(str(count) if item.count != count else '') + letter
                         for letter, item, count in zip(letters, items, counts) if count != 0]

        if all((not i.can_be_dropped_from_inventory() for i in items)):
            return False

        def key_gen():
            if 'Drop what type of items?' in '\n'.join(self.agent.popup):
                yield 'a'
                yield A.MiscAction.MORE
            assert 'What would you like to drop?' in '\n'.join(self.agent.popup), \
                   (self.agent.message, self.agent.popup)
            while texts_to_type:
                for text in list(texts_to_type):
                    letter = text[-1]
                    if f'{letter} - ' in '\n'.join(self.agent.popup):
                        yield from text
                        texts_to_type.remove(text)

                if texts_to_type:
                    yield A.TextCharacters.SPACE
            yield A.MiscAction.MORE

        with self.agent.atom_operation():
            self.agent.step(A.Command.DROPTYPE, key_gen())
        self.get_items_below_me()

        return True

    def call_item(self, item, name):
        assert item in self.items.all_items, item
        letter = self.items.get_letter(item)
        with self.agent.atom_operation():
            self.agent.step(A.Command.CALL, iter(f'i{letter}#{name}\r'))
        return True


    ######## STRATEGIES helpers

    def get_best_melee_weapon(self, items=None, *, return_dps=False, allow_unknown_status=False):
        if self.agent.character.role == Character.MONK:
            return None

        if items is None:
            items = self.items
        # select the best
        best_item = None
        best_dps = utils.calc_dps(*self.agent.character.get_melee_bonus(None, large_monster=False))
        for item in items:
            if item.is_weapon() and \
                    (item.status in [Item.UNCURSED, Item.BLESSED] or
                     (allow_unknown_status and item.status == Item.UNKNOWN)):
                to_hit, dmg = self.agent.character.get_melee_bonus(item, large_monster=False)
                dps = utils.calc_dps(to_hit, dmg)
                # dps = item.get_dps(large_monster=False)  # TODO: what about monster size
                if best_dps < dps:
                    best_dps = dps
                    best_item = item
        if return_dps:
            return best_item, best_dps
        return best_item

    def get_ranged_combinations(self, items=None, throwing=True, allow_best_melee=False, allow_unknown_status=False):
        if items is None:
            items = self.items
        launchers = [i for i in items if i.is_launcher()]
        ammo_list = [i for i in items if i.is_fired_projectile()]
        valid_combinations = []

        # TODO: should this condition be used here
        if any(l.equipped and l.status == Item.CURSED for l in launchers):
            launchers = [l for l in launchers if l.equipped]

        for launcher in launchers:
            for ammo in ammo_list:
                if ammo.is_fired_projectile(launcher):
                    if launcher.status in [Item.UNCURSED, Item.BLESSED] or \
                            (allow_unknown_status and launcher.status == Item.UNKNOWN):
                        valid_combinations.append((launcher, ammo))

        if throwing:
            best_melee_weapon = None
            if not allow_best_melee:
                best_melee_weapon = self.get_best_melee_weapon()
            valid_combinations.extend([(None, i) for i in items
                                       if i.is_thrown_projectile() and i != best_melee_weapon])

        return valid_combinations

    def get_best_ranged_set(self, items=None, *, throwing=True, allow_best_melee=False,
                            return_dps=False, allow_unknown_status=False):
        if items is None:
            items = self.items
        best_launcher, best_ammo = None, None
        best_dps = -float('inf')
        for launcher, ammo in self.get_ranged_combinations(items, throwing, allow_best_melee, allow_unknown_status):
            to_hit, dmg = self.agent.character.get_ranged_bonus(launcher, ammo)
            dps = utils.calc_dps(to_hit, dmg)
            if dps > best_dps:
                best_launcher, best_ammo, best_dps = launcher, ammo, dps
        if return_dps:
            return best_launcher, best_ammo, best_dps
        return best_launcher, best_ammo

    def get_best_armorset(self, items=None, *, return_ac=False, allow_unknown_status=False):
        if items is None:
            items = self.items

        best_items = [None] * O.ARM_NUM
        best_ac = [None] * O.ARM_NUM
        for item in items:
            if item.is_armor() and item.is_unambiguous() and \
                    (item.status in [Item.UNCURSED, Item.BLESSED] or
                     (allow_unknown_status and item.status == Item.UNKNOWN)):
                slot = item.object.sub
                ac = item.get_ac()

                if self.agent.character.role == Character.MONK and slot == O.ARM_SUIT:
                    continue

                if best_ac[slot] is None or best_ac[slot] > ac:
                    best_ac[slot] = ac
                    best_items[slot] = item

        if return_ac:
            return best_items, best_ac
        return best_items


    ######## LOW-LEVEL STRATEGIES

    def gather_items(self):
        return (
            self.pickup_and_drop_items()
            .before(self.check_containers())
            .before(self.wear_best_stuff())
            .before(self.go_to_unchecked_containers())
            .before(self.go_to_item_to_pickup()
                    .before(self.check_items()).repeat().every(5)
                    .preempt(self.agent, [
                        self.pickup_and_drop_items(),
                        self.check_containers(),
                    ])).repeat()
        )

    @utils.debug_log('inventory.wear_best_stuff')
    @Strategy.wrap
    def wear_best_stuff(self):
        yielded = False
        while 1:
            best_armorset = self.get_best_armorset()

            # TODO: twoweapon
            for slot, name in [(O.ARM_SHIELD, 'off_hand'), (O.ARM_HELM, 'helm'), (O.ARM_GLOVES, 'gloves'), (O.ARM_BOOTS, 'boots'), \
                               (O.ARM_SHIRT, 'shirt'), (O.ARM_SUIT, 'suit'), (O.ARM_CLOAK, 'cloak')]:
                if best_armorset[slot] == getattr(self.items, name) or \
                        (getattr(self.items, name) is not None and getattr(self.items, name).status == Item.CURSED):
                    continue
                additional_cond = True
                if slot == O.ARM_SHIELD:
                    additional_cond &= self.items.main_hand is None or not self.items.main_hand.objs[0].bi
                if slot == O.ARM_GLOVES:
                    additional_cond &= self.items.main_hand is None or self.items.main_hand.status != Item.CURSED
                if slot == O.ARM_SHIRT or slot == O.ARM_SUIT:
                    additional_cond &= self.items.cloak is None or self.items.cloak.status != Item.CURSED
                if slot == O.ARM_SHIRT:
                    additional_cond &= self.items.suit is None or self.items.suit.status != Item.CURSED

                if additional_cond:
                    if not yielded:
                        yielded = True
                        yield True
                    if (slot == O.ARM_SHIRT or slot == O.ARM_SUIT) and self.items.cloak is not None:
                        self.takeoff(self.items.cloak)
                        break
                    if slot == O.ARM_SHIRT and self.items.suit is not None:
                        self.takeoff(self.items.suit)
                        break
                    if getattr(self.items, name) is not None:
                        self.takeoff(getattr(self.items, name))
                        break
                    assert best_armorset[slot] is not None
                    self.wear(best_armorset[slot])
                    break
            else:
                break

        if not yielded:
            yield False


    @utils.debug_log('inventory.check_items')
    @Strategy.wrap
    def check_items(self):
        mask = utils.isin(self.agent.glyphs, G.OBJECTS, G.BODIES, G.STATUES)
        if not mask.any():
            yield False

        dis = self.agent.bfs()

        mask &= self.agent.current_level().item_count == 0
        if not mask.any():
            yield False

        mask &= dis > 0
        if not mask.any():
            yield False
        yield True

        nonzero_y, nonzero_x = (mask & (dis == dis[mask].min())).nonzero()
        i = self.agent.rng.randint(len(nonzero_y))
        target_y, target_x = nonzero_y[i], nonzero_x[i]

        with self.agent.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.go_to_unchecked_containers')
    @Strategy.wrap
    def go_to_unchecked_containers(self):
        mask = self.agent.current_level().item_count != 0
        if not mask.any():
            yield False

        dis = self.agent.bfs()
        mask &= dis > 0
        if not mask.any():
            yield False

        for y, x in zip(*mask.nonzero()):
            for item in self.agent.current_level().items[y, x]:
                if not item.is_possible_container():
                    mask[y, x] = False

        if not mask.any():
            yield False
        yield True

        nonzero_y, nonzero_x = (mask & (dis == dis[mask].min())).nonzero()
        i = self.agent.rng.randint(len(nonzero_y))
        target_y, target_x = nonzero_y[i], nonzero_x[i]

        with self.agent.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.check_containers')
    @Strategy.wrap
    def check_containers(self):
        yielded = False
        for item in self.agent.inventory.items_below_me:
            if item.is_possible_container():
                if not yielded:
                    yielded = True
                    yield True
                self.check_container_content(item)

        if not yielded:
            yield False

    @utils.debug_log('inventory.go_to_item_to_pickup')
    @Strategy.wrap
    def go_to_item_to_pickup(self):
        level = self.agent.current_level()
        dis = self.agent.bfs()

        # TODO: free (no charge) items
        mask = ~level.shop_interior & (dis > 0)
        if not mask.any():
            yield False

        mask[mask] = self.agent.current_level().item_count[mask] != 0

        items = {}
        for y, x in sorted(zip(*mask.nonzero()), key=lambda p: dis[p]):
            for i in level.items[y, x]:
                assert i not in items
                items[i] = (y, x)

        if not items:
            yield False  # TODO: what about just dropping items?

        free_items = list(filter(lambda i: i.can_be_dropped_from_inventory(), self.items))
        forced_items = list(filter(lambda i: not i.can_be_dropped_from_inventory(), self.items))
        counts = self.agent.global_logic.item_priority.split(
                free_items + list(items.keys()), forced_items,
                self.agent.character.carrying_capacity)

        counts = counts[len(free_items):]
        assert len(counts) == len(items)
        if sum(counts) == 0:
            yield False
        yield True

        for i, c in zip(items, counts):
            if c != 0:
                target_y, target_x = items[i]
                break
        else:
            assert 0

        with self.agent.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.pickup_and_drop_items')
    @Strategy.wrap
    def pickup_and_drop_items(self):
        # TODO: free (no charge) items
        self.item_manager.price_identification()
        if self.agent.current_level().shop_interior[self.agent.blstats.y, self.agent.blstats.x]:
            yield False
        if len(self.items_below_me) == 0:
            yield False

        yielded = False
        while 1:
            if not self.items_below_me:
                raise AgentPanic('items below me vanished')

            free_items = list(filter(lambda i: i.can_be_dropped_from_inventory(), self.items))
            forced_items = list(filter(lambda i: not i.can_be_dropped_from_inventory(), self.items))
            counts = self.agent.global_logic.item_priority.split(
                    free_items + self.items_below_me, forced_items,
                    self.agent.character.carrying_capacity)
            to_drop = [item.count - count for item, count in zip(free_items, counts)]
            if sum(to_drop) > 0:
                if not yielded:
                    yielded = True
                    yield True
                assert self.drop(free_items, to_drop)
                continue

            to_pickup = counts[len(free_items):]
            if sum(to_pickup) > 0:
                if not yielded:
                    yielded = True
                    yield True
                assert self.pickup(self.items_below_me, to_pickup)
            break

        if not yielded:
            yield False
