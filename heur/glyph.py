import functools

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
    # /include/monflag.h
    MS_SILENT = 0     #/* makes no sound */
    MS_BARK = 1       #/* if full moon, may howl */
    MS_MEW = 2        #/* mews or hisses */
    MS_ROAR = 3       #/* roars */
    MS_GROWL = 4      #/* growls */
    MS_SQEEK = 5      #/* squeaks, as a rodent */
    MS_SQAWK = 6      #/* squawks, as a bird */
    MS_HISS = 7       #/* hisses */
    MS_BUZZ = 8       #/* buzzes (killer bee) */
    MS_GRUNT = 9      #/* grunts (or speaks own language) */
    MS_NEIGH = 10     #/* neighs, as an equine */
    MS_WAIL = 11      #/* wails, as a tortured soul */
    MS_GURGLE = 12    #/* gurgles, as liquid or through saliva */
    MS_BURBLE = 13    #/* burbles (jabberwock) */
    MS_ANIMAL = 13    #/* up to here are animal noises */
    MS_SHRIEK = 15    #/* wakes up others */
    MS_BONES = 16     #/* rattles bones (skeleton) */
    MS_LAUGH = 17     #/* grins, smiles, giggles, and laughs */
    MS_MUMBLE = 18    #/* says something or other */
    MS_IMITATE = 19   #/* imitates others (leocrotta) */
    MS_ORC = MS_GRUNT #/* intelligent brutes */
    MS_HUMANOID = 20  #/* generic traveling companion */
    MS_ARREST = 21    #/* "Stop in the name of the law!" (Kops) */
    MS_SOLDIER = 22   #/* army and watchmen expressions */
    MS_GUARD = 23     #/* "Please drop that gold and follow me." */
    MS_DJINNI = 24    #/* "Thank you for freeing me!" */
    MS_NURSE = 25     #/* "Take off your shirt, please." */
    MS_SEDUCE = 26    #/* "Hello, sailor." (Nymphs) */
    MS_VAMPIRE = 27   #/* vampiric seduction, Vlad's exclamations */
    MS_BRIBE = 28     #/* asks for money, or berates you */
    MS_CUSS = 29      #/* berates (demons) or intimidates (Wiz) */
    MS_RIDER = 30     #/* astral level special monsters */
    MS_LEADER = 31    #/* your class leader */
    MS_NEMESIS = 32   #/* your nemesis */
    MS_GUARDIAN = 33  #/* your leader's guards */
    MS_SELL = 34      #/* demand payment, complain about shoplifters */
    MS_ORACLE = 35    #/* do a consultation */
    MS_PRIEST = 36    #/* ask for contribution; do cleansing */
    MS_SPELL = 37     #/* spellcaster not matching any of the above */
    MS_WERE = 38      #/* lycanthrope in human form */
    MS_BOAST = 39     #/* giants */

    MR_FIRE = 0x01   #/* resists fire */
    MR_COLD = 0x02   #/* resists cold */
    MR_SLEEP = 0x04  #/* resists sleep */
    MR_DISINT = 0x08 #/* resists disintegration */
    MR_ELEC = 0x10   #/* resists electricity */
    MR_POISON = 0x20 #/* resists poison */
    MR_ACID = 0x40   #/* resists acid */
    MR_STONE = 0x80  #/* resists petrification */
    #/* other resistances: magic, sickness */
    #/* other conveyances: teleport, teleport control, telepathy */

    #/* individual resistances */
    MR2_SEE_INVIS = 0x0100 #/* see invisible */
    MR2_LEVITATE = 0x0200  #/* levitation */
    MR2_WATERWALK = 0x0400 #/* water walking */
    MR2_MAGBREATH = 0x0800 #/* magical breathing */
    MR2_DISPLACED = 0x1000 #/* displaced */
    MR2_STRENGTH = 0x2000  #/* gauntlets of power */
    MR2_FUMBLING = 0x4000  #/* clumsy */

    M1_FLY = 0x00000001         #/* can fly or float */
    M1_SWIM = 0x00000002        #/* can traverse water */
    M1_AMORPHOUS = 0x00000004   #/* can flow under doors */
    M1_WALLWALK = 0x00000008    #/* can phase thru rock */
    M1_CLING = 0x00000010       #/* can cling to ceiling */
    M1_TUNNEL = 0x00000020      #/* can tunnel thru rock */
    M1_NEEDPICK = 0x00000040    #/* needs pick to tunnel */
    M1_CONCEAL = 0x00000080     #/* hides under objects */
    M1_HIDE = 0x00000100        #/* mimics, blends in with ceiling */
    M1_AMPHIBIOUS = 0x00000200  #/* can survive underwater */
    M1_BREATHLESS = 0x00000400  #/* doesn't need to breathe */
    M1_NOTAKE = 0x00000800      #/* cannot pick up objects */
    M1_NOEYES = 0x00001000      #/* no eyes to gaze into or blind */
    M1_NOHANDS = 0x00002000     #/* no hands to handle things */
    M1_NOLIMBS = 0x00006000     #/* no arms/legs to kick/wear on */
    M1_NOHEAD = 0x00008000      #/* no head to behead */
    M1_MINDLESS = 0x00010000    #/* has no mind--golem, zombie, mold */
    M1_HUMANOID = 0x00020000    #/* has humanoid head/arms/torso */
    M1_ANIMAL = 0x00040000      #/* has animal body */
    M1_SLITHY = 0x00080000      #/* has serpent body */
    M1_UNSOLID = 0x00100000     #/* has no solid or liquid body */
    M1_THICK_HIDE = 0x00200000  #/* has thick hide or scales */
    M1_OVIPAROUS = 0x00400000   #/* can lay eggs */
    M1_REGEN = 0x00800000       #/* regenerates hit points */
    M1_SEE_INVIS = 0x01000000   #/* can see invisible creatures */
    M1_TPORT = 0x02000000       #/* can teleport */
    M1_TPORT_CNTRL = 0x04000000 #/* controls where it teleports to */
    M1_ACID = 0x08000000        #/* acidic to eat */
    M1_POIS = 0x10000000        #/* poisonous to eat */
    M1_CARNIVORE = 0x20000000   #/* eats corpses */
    M1_HERBIVORE = 0x40000000   #/* eats fruits */
    M1_OMNIVORE = 0x60000000    #/* eats both */
    #ifdef NHSTDC
    #define M1_METALLIVORE 0x80000000UL /* eats metal */
    #else
    M1_METALLIVORE = 0x80000000 #/* eats metal */
    #endif

    M2_NOPOLY = 0x00000001       #/* players mayn't poly into one */
    M2_UNDEAD = 0x00000002       #/* is walking dead */
    M2_WERE = 0x00000004         #/* is a lycanthrope */
    M2_HUMAN = 0x00000008        #/* is a human */
    M2_ELF = 0x00000010          #/* is an elf */
    M2_DWARF = 0x00000020        #/* is a dwarf */
    M2_GNOME = 0x00000040        #/* is a gnome */
    M2_ORC = 0x00000080          #/* is an orc */
    M2_DEMON = 0x00000100        #/* is a demon */
    M2_MERC = 0x00000200         #/* is a guard or soldier */
    M2_LORD = 0x00000400         #/* is a lord to its kind */
    M2_PRINCE = 0x00000800       #/* is an overlord to its kind */
    M2_MINION = 0x00001000       #/* is a minion of a deity */
    M2_GIANT = 0x00002000        #/* is a giant */
    M2_SHAPESHIFTER = 0x00004000 #/* is a shapeshifting species */
    M2_MALE = 0x00010000         #/* always male */
    M2_FEMALE = 0x00020000       #/* always female */
    M2_NEUTER = 0x00040000       #/* neither male nor female */
    M2_PNAME = 0x00080000        #/* monster name is a proper name */
    M2_HOSTILE = 0x00100000      #/* always starts hostile */
    M2_PEACEFUL = 0x00200000     #/* always starts peaceful */
    M2_DOMESTIC = 0x00400000     #/* can be tamed by feeding */
    M2_WANDER = 0x00800000       #/* wanders randomly */
    M2_STALK = 0x01000000        #/* follows you to other levels */
    M2_NASTY = 0x02000000        #/* extra-nasty monster (more xp) */
    M2_STRONG = 0x04000000       #/* strong (or big) monster */
    M2_ROCKTHROW = 0x08000000    #/* throws boulders */
    M2_GREEDY = 0x10000000       #/* likes gold */
    M2_JEWELS = 0x20000000       #/* likes gems */
    M2_COLLECT = 0x40000000      #/* picks up weapons and food */
    #ifdef NHSTDC
    #define M2_MAGIC 0x80000000UL /* picks up magic items */
    #else
    M2_MAGIC = 0x80000000 #/* picks up magic items */
    #endif

    M3_WANTSAMUL = 0x0001 #/* would like to steal the amulet */
    M3_WANTSBELL = 0x0002 #/* wants the bell */
    M3_WANTSBOOK = 0x0004 #/* wants the book */
    M3_WANTSCAND = 0x0008 #/* wants the candelabrum */
    M3_WANTSARTI = 0x0010 #/* wants the quest artifact */
    M3_WANTSALL = 0x001f  #/* wants any major artifact */
    M3_WAITFORU = 0x0040  #/* waits to see you or get attacked */
    M3_CLOSE = 0x0080     #/* lets you close unless attacked */

    M3_COVETOUS = 0x001f #/* wants something */
    M3_WAITMASK = 0x00c0 #/* waiting... */

    #/* Infravision is currently implemented for players only */
    M3_INFRAVISION = 0x0100  #/* has infravision */
    M3_INFRAVISIBLE = 0x0200 #/* visible by infravision */

    M3_DISPLACES = 0x0400 #/* moves monsters out of its way */

    MZ_TINY = 0          #/* < 2' */
    MZ_SMALL = 1         #/* 2-4' */
    MZ_MEDIUM = 2        #/* 4-7' */
    MZ_HUMAN = MZ_MEDIUM #/* human-sized */
    MZ_LARGE = 3         #/* 7-12' */
    MZ_HUGE = 4          #/* 12-25' */
    MZ_GIGANTIC = 7      #/* off the scale */

    #/* Monster races -- must stay within ROLE_RACEMASK */
    #/* Eventually this may become its own field */
    MH_HUMAN = M2_HUMAN
    MH_ELF = M2_ELF
    MH_DWARF = M2_DWARF
    MH_GNOME = M2_GNOME
    MH_ORC = M2_ORC

    #/* for mons[].geno (constant during game) */
    G_UNIQ = 0x1000     #/* generated only once */
    G_NOHELL = 0x0800   #/* not generated in "hell" */
    G_HELL = 0x0400     #/* generated only in "hell" */
    G_NOGEN = 0x0200    #/* generated only specially */
    G_SGROUP = 0x0080   #/* appear in small groups normally */
    G_LGROUP = 0x0040   #/* appear in large groups normally */
    G_GENO = 0x0020     #/* can be genocided */
    G_NOCORPSE = 0x0010 #/* no corpse left ever */
    G_FREQ = 0x0007     #/* creation frequency mask */

    #/* for mvitals[].mvflags (variant during game), along with G_NOCORPSE */
    G_KNOWN = 0x0004 #/* have been encountered */
    G_GENOD = 0x0002 #/* have been genocided */
    G_EXTINCT = 0x0001 #/* have been extinguished as population control */
    G_GONE = (G_GENOD | G_EXTINCT)
    MV_KNOWS_EGG = 0x0008 #/* player recognizes egg of this monster type */

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
        elif nh.glyph_is_body(glyph):
            return nh.permonst(glyph - nh.GLYPH_BODY_OFF)
        else:
            assert 0

    @staticmethod
    def find(glyph):
        if glyph in MON.ALL_MONS or glyph in MON.ALL_PETS:
            return f'MON.fn({repr(MON.permonst(glyph).mname)})'

    @staticmethod
    @functools.lru_cache(nh.NUMMONS)
    def from_name(name):
        for i in range(nh.NUMMONS):
            if nh.permonst(i).mname == name:
                return nh.GLYPH_MON_OFF + i
        assert 0, name

    @staticmethod
    def body_from_name(name):
        return MON.from_name(name) - nh.GLYPH_MON_OFF + nh.GLYPH_BODY_OFF

    fn = from_name

    ALL_MONS = [nh.GLYPH_MON_OFF + i for i in range(nh.NUMMONS)]
    ALL_PETS = [nh.GLYPH_PET_OFF + i for i in range(nh.NUMMONS)]


class WEA:
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

class ALL:
    @staticmethod
    def find(glyph):
        x = None
        x = SS.find(glyph) if x is None else x
        x = MON.find(glyph) if x is None else x
        return x


class Hunger:
    SATIATED = 0
    NOT_HUNGRY = 1
    HUNGRY = 2
    WEAK = 3
    FAINTING = 4


class C:
    SIZE_X = 79
    SIZE_Y = 21


class G:  # Glyphs
    FLOOR: ['.'] = {SS.S_room, SS.S_ndoor, SS.S_darkroom}
    VISIBLE_FLOOR: ['.'] = {SS.S_room}
    STONE: [' '] = {SS.S_stone}
    WALL: ['|', '-'] = {SS.S_vwall, SS.S_hwall, SS.S_tlcorn, SS.S_trcorn, SS.S_blcorn, SS.S_brcorn,
                        SS.S_crwall, SS.S_tuwall, SS.S_tdwall, SS.S_tlwall, SS.S_trwall}
    CORRIDOR: ['#'] = {SS.S_corr}
    STAIR_UP: ['<'] = {SS.S_upstair}
    STAIR_DOWN: ['>'] = {SS.S_dnstair}
    ALTAR: ['_'] = {SS.S_altar}
    FOUNTAIN = {SS.S_fountain}

    DOOR_CLOSED: ['+'] = {SS.S_vcdoor, SS.S_hcdoor}
    DOOR_OPENED: ['-', '|'] = {SS.S_vodoor, SS.S_hodoor}
    DOORS = set.union(DOOR_CLOSED, DOOR_OPENED)

    BARS = {SS.S_bars}

    MONS = set(MON.ALL_MONS)
    PETS = set(MON.ALL_PETS)
    INVISIBLE_MON = {nh.GLYPH_INVISIBLE}
    PEACEFUL_MONS = {i + nh.GLYPH_MON_OFF for i in range(nh.NUMMONS) if nh.permonst(i).mflags2 & MON.M2_PEACEFUL}

    STATUES = {i + nh.GLYPH_STATUE_OFF for i in range(nh.NUMMONS)}

    BODIES = {nh.GLYPH_BODY_OFF + i for i in range(nh.NUMMONS)}
    OBJECTS = {nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) != nh.ROCK_CLASS}
    BOULDER = {nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) == nh.ROCK_CLASS}

    NORMAL_OBJECTS = {i for i in range(nh.MAX_GLYPH) if nh.glyph_is_normal_object(i)}
    FOOD_OBJECTS = {i for i in NORMAL_OBJECTS if ord(nh.objclass(nh.glyph_to_obj(i)).oc_class) == nh.FOOD_CLASS}

    TRAPS = {SS.S_arrow_trap, SS.S_dart_trap, SS.S_falling_rock_trap, SS.S_squeaky_board, SS.S_bear_trap,
             SS.S_land_mine, SS.S_rolling_boulder_trap, SS.S_sleeping_gas_trap, SS.S_rust_trap, SS.S_fire_trap,
             SS.S_pit, SS.S_spiked_pit, SS.S_hole, SS.S_trap_door, SS.S_teleportation_trap, SS.S_level_teleporter,
             SS.S_magic_portal, SS.S_web, SS.S_statue_trap, SS.S_magic_trap, SS.S_anti_magic_trap, SS.S_polymorph_trap}

    DICT = {k: v for k, v in locals().items() if not k.startswith('_')}

    @classmethod
    def assert_map(cls, glyphs, chars):
        for glyph, char in zip(glyphs.reshape(-1), chars.reshape(-1)):
            char = bytes([char]).decode()
            for k, v in cls.__annotations__.items():
                assert glyph not in cls.DICT[k] or char in v, f'{k} {v} {glyph} {char}'


G.INV_DICT = {i: [k for k, v in G.DICT.items() if i in v]
              for i in set.union(*map(set, G.DICT.values()))}
