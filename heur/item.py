import re

import nle.nethack as nh

from glyph import WEA


class Item:
    UNKNOWN = 0
    CURSED = 1
    UNCURSED = 2
    BLESSED = 3

    def __init__(self, glyphs, count=1, status=UNKNOWN, modifier=None, worn=False, at_ready=False):
        # glyphs is a list of possible glyphs for this item
        assert isinstance(glyphs, list)
        assert all(map(nh.glyph_is_object, glyphs))
        assert len(glyphs)

        self.glyphs = glyphs
        self.count = count
        self.status = status
        self.modifier = modifier
        self.worn = worn
        self.at_ready = at_ready
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
            worn = True
            at_ready = False
        elif info in {'at the ready', 'in quiver', 'in quiver pouch'}:
            worn = False
            at_ready = True
        elif info in {'', 'alternate weapon; not wielded'}:
            worn = False
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
                count, status, modifier, worn, at_ready)
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
        return Item([r + nh.GLYPH_OBJ_OFF for r in ret], count, status, modifier, worn, at_ready)
