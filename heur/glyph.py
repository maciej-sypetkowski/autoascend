import nle.nethack as nh


class SS: # screen_symbols
    S_stone     =nh.GLYPH_CMAP_OFF+  0#
    S_vwall     =nh.GLYPH_CMAP_OFF+  1#
    S_hwall     =nh.GLYPH_CMAP_OFF+  2#
    S_tlcorn    =nh.GLYPH_CMAP_OFF+  3#
    S_trcorn    =nh.GLYPH_CMAP_OFF+  4#
    S_blcorn    =nh.GLYPH_CMAP_OFF+  5#
    S_brcorn    =nh.GLYPH_CMAP_OFF+  6#
    S_crwall    =nh.GLYPH_CMAP_OFF+  7#
    S_tuwall    =nh.GLYPH_CMAP_OFF+  8#
    S_tdwall    =nh.GLYPH_CMAP_OFF+  9#
    S_tlwall    =nh.GLYPH_CMAP_OFF+ 10#
    S_trwall    =nh.GLYPH_CMAP_OFF+ 11#
    S_ndoor     =nh.GLYPH_CMAP_OFF+ 12#
    S_vodoor    =nh.GLYPH_CMAP_OFF+ 13#
    S_hodoor    =nh.GLYPH_CMAP_OFF+ 14#
    S_vcdoor    =nh.GLYPH_CMAP_OFF+ 15# /* closed door, vertical wall */
    S_hcdoor    =nh.GLYPH_CMAP_OFF+ 16# /* closed door, horizontal wall */
    S_bars      =nh.GLYPH_CMAP_OFF+ 17# /* KMH -- iron bars */
    S_tree      =nh.GLYPH_CMAP_OFF+ 18# /* KMH */
    S_room      =nh.GLYPH_CMAP_OFF+ 19#
    S_darkroom  =nh.GLYPH_CMAP_OFF+ 20#
    S_corr      =nh.GLYPH_CMAP_OFF+ 21#
    S_litcorr   =nh.GLYPH_CMAP_OFF+ 22#
    S_upstair   =nh.GLYPH_CMAP_OFF+ 23#
    S_dnstair   =nh.GLYPH_CMAP_OFF+ 24#
    S_upladder  =nh.GLYPH_CMAP_OFF+ 25#
    S_dnladder  =nh.GLYPH_CMAP_OFF+ 26#
    S_altar     =nh.GLYPH_CMAP_OFF+ 27#
    S_grave     =nh.GLYPH_CMAP_OFF+ 28#
    S_throne    =nh.GLYPH_CMAP_OFF+ 29#
    S_sink      =nh.GLYPH_CMAP_OFF+ 30#
    S_fountain  =nh.GLYPH_CMAP_OFF+ 31#
    S_pool      =nh.GLYPH_CMAP_OFF+ 32#
    S_ice       =nh.GLYPH_CMAP_OFF+ 33#
    S_lava      =nh.GLYPH_CMAP_OFF+ 34#
    S_vodbridge =nh.GLYPH_CMAP_OFF+ 35#
    S_hodbridge =nh.GLYPH_CMAP_OFF+ 36#
    S_vcdbridge =nh.GLYPH_CMAP_OFF+ 37# /* closed drawbridge, vertical wall */
    S_hcdbridge =nh.GLYPH_CMAP_OFF+ 38# /* closed drawbridge, horizontal wall */
    S_air       =nh.GLYPH_CMAP_OFF+ 39#
    S_cloud     =nh.GLYPH_CMAP_OFF+ 40#
    S_water     =nh.GLYPH_CMAP_OFF+ 41#

#/* end dungeon characters, begin traps */

    S_arrow_trap           =nh.GLYPH_CMAP_OFF+ 42#
    S_dart_trap            =nh.GLYPH_CMAP_OFF+ 43#
    S_falling_rock_trap    =nh.GLYPH_CMAP_OFF+ 44#
    S_squeaky_board        =nh.GLYPH_CMAP_OFF+ 45#
    S_bear_trap            =nh.GLYPH_CMAP_OFF+ 46#
    S_land_mine            =nh.GLYPH_CMAP_OFF+ 47#
    S_rolling_boulder_trap =nh.GLYPH_CMAP_OFF+ 48#
    S_sleeping_gas_trap    =nh.GLYPH_CMAP_OFF+ 49#
    S_rust_trap            =nh.GLYPH_CMAP_OFF+ 50#
    S_fire_trap            =nh.GLYPH_CMAP_OFF+ 51#
    S_pit                  =nh.GLYPH_CMAP_OFF+ 52#
    S_spiked_pit           =nh.GLYPH_CMAP_OFF+ 53#
    S_hole                 =nh.GLYPH_CMAP_OFF+ 54#
    S_trap_door            =nh.GLYPH_CMAP_OFF+ 55#
    S_teleportation_trap   =nh.GLYPH_CMAP_OFF+ 56#
    S_level_teleporter     =nh.GLYPH_CMAP_OFF+ 57#
    S_magic_portal         =nh.GLYPH_CMAP_OFF+ 58#
    S_web                  =nh.GLYPH_CMAP_OFF+ 59#
    S_statue_trap          =nh.GLYPH_CMAP_OFF+ 60#
    S_magic_trap           =nh.GLYPH_CMAP_OFF+ 61#
    S_anti_magic_trap      =nh.GLYPH_CMAP_OFF+ 62#
    S_polymorph_trap       =nh.GLYPH_CMAP_OFF+ 63#
    S_vibrating_square     =nh.GLYPH_CMAP_OFF+ 64# /* for display rather than any trap effect */

#/* end traps, begin special effects */

    S_vbeam       =nh.GLYPH_CMAP_OFF+ 65# /* The 4 zap beam symbols.  Do NOT separate. */
    S_hbeam       =nh.GLYPH_CMAP_OFF+ 66# /* To change order or add, see function      */
    S_lslant      =nh.GLYPH_CMAP_OFF+ 67# /* zapdir_to_glyph() in display.c.           */
    S_rslant      =nh.GLYPH_CMAP_OFF+ 68#
    S_digbeam     =nh.GLYPH_CMAP_OFF+ 69# /* dig beam symbol */
    S_flashbeam   =nh.GLYPH_CMAP_OFF+ 70# /* camera flash symbol */
    S_boomleft    =nh.GLYPH_CMAP_OFF+ 71# /* thrown boomerang, open left, e.g ')'    */
    S_boomright   =nh.GLYPH_CMAP_OFF+ 72# /* thrown boomerang, open right, e.g. '('  */
    S_ss1         =nh.GLYPH_CMAP_OFF+ 73# /* 4 magic shield ("resistance sparkle") glyphs */
    S_ss2         =nh.GLYPH_CMAP_OFF+ 74#
    S_ss3         =nh.GLYPH_CMAP_OFF+ 75#
    S_ss4         =nh.GLYPH_CMAP_OFF+ 76#
    S_poisoncloud =nh.GLYPH_CMAP_OFF+ 77
    S_goodpos     =nh.GLYPH_CMAP_OFF+ 78# /* valid position for targeting via getpos() */

#/* The 8 swallow symbols.  Do NOT separate.  To change order or add, */
#/* see the function swallow_to_glyph() in display.c.                 */
    S_sw_tl     =nh.GLYPH_CMAP_OFF+ 79# /* swallow top left [1]             */
    S_sw_tc     =nh.GLYPH_CMAP_OFF+ 80# /* swallow top center [2]    Order: */
    S_sw_tr     =nh.GLYPH_CMAP_OFF+ 81# /* swallow top right [3]            */
    S_sw_ml     =nh.GLYPH_CMAP_OFF+ 82# /* swallow middle left [4]   1 2 3  */
    S_sw_mr     =nh.GLYPH_CMAP_OFF+ 83# /* swallow middle right [6]  4 5 6  */
    S_sw_bl     =nh.GLYPH_CMAP_OFF+ 84# /* swallow bottom left [7]   7 8 9  */
    S_sw_bc     =nh.GLYPH_CMAP_OFF+ 85# /* swallow bottom center [8]        */
    S_sw_br     =nh.GLYPH_CMAP_OFF+ 86# /* swallow bottom right [9]         */

    S_explode1  =nh.GLYPH_CMAP_OFF+ 87# /* explosion top left               */
    S_explode2  =nh.GLYPH_CMAP_OFF+ 88# /* explosion top center             */
    S_explode3  =nh.GLYPH_CMAP_OFF+ 89# /* explosion top right        Ex.   */
    S_explode4  =nh.GLYPH_CMAP_OFF+ 90# /* explosion middle left            */
    S_explode5  =nh.GLYPH_CMAP_OFF+ 91# /* explosion middle center    /-\   */
    S_explode6  =nh.GLYPH_CMAP_OFF+ 92# /* explosion middle right     |@|   */
    S_explode7  =nh.GLYPH_CMAP_OFF+ 93# /* explosion bottom left      \-/   */
    S_explode8  =nh.GLYPH_CMAP_OFF+ 94# /* explosion bottom center          */
    S_explode9  =nh.GLYPH_CMAP_OFF+ 95# /* explosion bottom right           */

#/* end effects */

    MAXPCHARS   = 96# /* maximum number of mapped characters */

    @classmethod
    def find(cls, glyph):
        for k, v in vars(cls).items():
            if k.startswith('S_') and v == glyph:
                return f'SS.{k}'


class MON: # monsters, pets
    @staticmethod
    def is_monster(glyph):
        return nh.glyph_is_monster(glyph)

    @staticmethod
    def is_pet(glyph):
        return nh.glyph_is_pet(glyph)

    @staticmethod
    def permonst(glyph):
        if nh.glyph_is_monster(glyph):
            return nh.permonst(nh.glyph_to_mon(glyph))
        elif nh.glyph_is_pet(glyph):
            return nh.permonst(nh.glyph_to_pet(glyph))
        else:
            assert 0

    @staticmethod
    def find(glyph):
        if glyph in MON.ALL_MONS or glyph in MON.ALL_PETS:
            return f'MON.fn({repr(MON.permonst(glyph).mname)})'

    @staticmethod
    def from_name(name):
        for i in range(nh.NUMMONS):
            if nh.permonst(i).mname == name:
                return nh.GLYPH_MON_OFF + i
        assert 0

    fn = from_name

    ALL_MONS = [nh.GLYPH_MON_OFF + i for i in range(nh.NUMMONS)]
    ALL_PETS = [nh.GLYPH_PET_OFF + i for i in range(nh.NUMMONS)]


class WEA:
    # Taken from: https://nethackwiki.com/wiki/Weapon#Table_of_weapons_and_their_properties
    data = [
        ('orcish dagger', 'dagger', 'd3', 'd3'),
        ('dagger', 'dagger', 'd4', 'd3'),
        ('silver dagger', 'dagger', 'd4', 'd3'),
        ('athame', 'dagger', 'd4', 'd3'),
        ('elven dagger', 'dagger', 'd5', 'd3'),
        ('worm tooth', 'knife', 'd2', 'd2'),
        ('knife', 'knife', 'd3', 'd2'),
        ('stiletto', 'knife', 'd3', 'd2'),
        ('scalpel', 'knife', 'd3', 'd3'),
        ('crysknife', 'knife', 'd10', 'd10'),
        ('axe', 'axe', 'd6', 'd4'),
        ('battle-axe', 'axe', 'd8+d4', 'd6+2d4'),
        ('pick-axe', 'pick-axe', 'd6', 'd3'),
        ('dwarvish mattock', 'pick-axe', 'd12', 'd8+2d6'),
        ('orcish short sword', 'short sword', 'd5', 'd8'),
        ('short sword', 'short sword', 'd6', 'd8'),
        ('dwarvish short sword', 'short sword', 'd7', 'd8'),
        ('elven short sword', 'short sword', 'd8', 'd8'),
        ('broadsword', 'broadsword', '2d4', 'd6+1'),
        ('runesword', 'broadsword', '2d4', 'd6+1'),
        ('elven broadsword', 'broadsword', 'd6+d4', 'd6+1'),
        ('long sword', 'long sword', 'd8', 'd12'),
        ('katana', 'long sword', 'd10', 'd12'),
        ('two-handed sword', 'two-handed sword', 'd12', '3d6'),
        ('tsurugi', 'two-handed sword', 'd16', 'd8+2d6'),
        ('scimitar', 'scimitar', 'd8', 'd8'),
        ('silver saber', 'saber', 'd8', 'd8'),
        ('club', 'club', 'd6', 'd3'),
        ('aklys', 'club', 'd6', 'd3'),
        ('mace', 'mace', 'd6+1', 'd6'),
        ('morning star', 'morning star', '2d4', 'd6+1'),
        ('flail', 'flail', 'd6+1', '2d4'),
        ('grappling hook', 'flail', 'd2', 'd6'),
        ('war hammer', 'hammer', 'd4+1', 'd4'),
        ('quarterstaff', 'quarterstaff', 'd6', 'd6'),
        ('partisan', 'polearms', 'd6', 'd6+1'),
        ('fauchard', 'polearms', 'd6', 'd8'),
        ('glaive', 'polearms', 'd6', 'd10'),
        ('bec-de-corbin', 'polearms', 'd8', 'd6'),
        ('spetum', 'polearms', 'd6+1', '2d6'),
        ('lucern hammer', 'polearms', '2d4', 'd6'),
        ('guisarme', 'polearms', '2d4', 'd8'),
        ('ranseur', 'polearms', '2d4', '2d4'),
        ('voulge', 'polearms', '2d4', '2d4'),
        ('bill-guisarme', 'polearms', '2d4', 'd10'),
        ('bardiche', 'polearms', '2d4', '3d4'),
        ('halberd', 'polearms', 'd10', '2d6'),
        ('orcish spear', 'spear', 'd5', 'd8'),
        ('spear', 'spear', 'd6', 'd8'),
        ('silver spear', 'spear', 'd6', 'd8'),
        ('elven spear', 'spear', 'd7', 'd8'),
        ('dwarvish spear', 'spear', 'd8', 'd8'),
        ('javelin', 'spear', 'd6', 'd6'),
        ('trident', 'trident', 'd6+1', '3d4'),
        ('lance', 'lance', 'd6', 'd8'),
        ('orcish bow', 'bow', 'd2', 'd2'),
        ('orcish arrow', 'bow', 'd5', 'd6'),
        ('bow', 'bow', 'd2', 'd2'),
        ('arrow', 'bow', 'd6', 'd6'),
        ('elven bow', 'bow', 'd2', 'd2'),
        ('elven arrow', 'bow', 'd7', 'd6'),
        ('yumi', 'bow', 'd2', 'd2'),
        ('ya', 'bow', 'd7', 'd7'),
        ('silver arrow', 'bow', 'd6', 'd6'),
        ('sling', 'sling', 'd2', 'd2'),
        ('flint stone', 'sling', 'd6', 'd6'),
        ('crossbow', 'crossbow', 'd2', 'd2'),
        ('crossbow bolt', 'crossbow', 'd4+1', 'd6+1'),
        ('dart', 'dart', 'd3', 'd2'),
        ('shuriken', 'shuriken', 'd8', 'd6'),
        ('boomerang', 'boomerang', 'd9', 'd9'),
        ('bullwhip', 'whip', 'd2', '1'),
        ('rubber hose', 'whip', 'd4', 'd3'),
        ('unicorn horn', 'unicorn horn', 'd12', 'd12'),


        # synonyms
        ('shito', 'knife', 'd3', 'd2'),
        ('wakizashi', 'short sword', 'd6', 'd8'),
        ('ninja-to', 'broadsword', '2d4', 'd6+1'),
        ('nunchaku', 'flail', 'd6+1', '2d4'),
        ('naginata', 'polearms', 'd6', 'd10'),
        ('bec de corbin', 'polearms', 'd8', 'd6'),
    ]

    @staticmethod
    def expected_damage(damage_str):
        if '-' in damage_str:
            raise NotImplementedError()
        ret = 0
        for word in damage_str.split('+'):
            if 'd' in word:
                sides = int(word[word.find('d') + 1:])
                mult = word[:word.find('d')]
                if not mult:
                    mult = 1
                else:
                    mult = int(mult)
            else:
                sides = 1
                mult = int(word)
            ret += mult * (1 + sides) / 2
        return ret


    @classmethod
    def get_dps(cls, glyph, large_monster):
        assert nh.glyph_is_object(glyph), glyph
        obj = nh.objclass(nh.glyph_to_obj(glyph))
        objname = nh.objdescr.from_idx(nh.glyph_to_obj(glyph)).oc_name
        assert ord(obj.oc_class) == nh.WEAPON_CLASS
        for name, _, small_dps, large_dps in cls.data:
            if name == objname:
                if large_monster:
                    return cls.expected_damage(large_dps)
                else:
                    return cls.expected_damage(small_dps)
        assert 0, objname

class ALL:
    @staticmethod
    def find(glyph):
        x = None
        x = SS.find(glyph) if x is None else x
        x = MON.find(glyph) if x is None else x
        return x

class C:
    SIZE_X = 79
    SIZE_Y = 21
