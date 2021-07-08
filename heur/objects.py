from collections import namedtuple
from enum import Enum

Weapon = namedtuple('Weapon', 'name,desc,kn,mg,bi,prob,wt,cost,sdam,ldam,hitbon,typ,sub,metal,color,damage_small,damage_large'.split(','))
Armor = namedtuple('Armor', 'name,desc,kn,mgc,blk,power,prob,delay,wt,cost,ac,can,sub,metal,c'.split(','))


#################### https://github.com/facebookresearch/nle/blob/master/include/skills.h

#/* Code to denote that no skill is applicable */
P_NONE = 0#,

P_DAGGER             =  1#,
P_KNIFE              =  2#,
P_AXE                =  3#,
P_PICK_AXE           =  4#,
P_SHORT_SWORD        =  5#,
P_BROAD_SWORD        =  6#,
P_LONG_SWORD         =  7#,
P_TWO_HANDED_SWORD   =  8#,
P_SCIMITAR           =  9#,
P_SABER              = 10#,
P_CLUB               = 11#, /* Heavy-shafted bludgeon */
P_MACE               = 12#,
P_MORNING_STAR       = 13#, /* Spiked bludgeon */
P_FLAIL              = 14#, /* Two pieces hinged or chained together */
P_HAMMER             = 15#, /* Heavy head on the end */
P_QUARTERSTAFF       = 16#, /* Long-shafted bludgeon */
P_POLEARMS           = 17#, /* attack two or three steps away */
P_SPEAR              = 18#, /* includes javelin */
P_TRIDENT            = 19#,
P_LANCE              = 20#,
P_BOW                = 21#, /* launchers */
P_SLING              = 22#,
P_CROSSBOW           = 23#,
P_DART               = 24#, /* hand-thrown missiles */
P_SHURIKEN           = 25#,
P_BOOMERANG          = 26#,
P_WHIP               = 27#, /* flexible, one-handed */
P_UNICORN_HORN       = 28#, /* last weapon, two-handed */

#/* Spell Skills added by Larry Stewart-Zerba */
P_ATTACK_SPELL       = 29#,
P_HEALING_SPELL      = 30#,
P_DIVINATION_SPELL   = 31#,
P_ENCHANTMENT_SPELL  = 32#,
P_CLERIC_SPELL       = 33#,
P_ESCAPE_SPELL       = 34#,
P_MATTER_SPELL       = 35#,

#/* Other types of combat */
P_BARE_HANDED_COMBAT = 36#, /* actually weaponless; gloves are ok */
P_TWO_WEAPON_COMBAT  = 37#, /* pair of weapons, one in each hand */
P_RIDING             = 38#, /* How well you control your steed */

P_NUM_SKILLS         = 39#
####################


#################### https://github.com/facebookresearch/nle/blob/master/include/objclass.h
LIQUID      =  1#, /* currently only for venom */
WAX         =  2#,
VEGGY       =  3#, /* foodstuffs */
FLESH       =  4#, /*   ditto    */
PAPER       =  5#,
CLOTH       =  6#,
LEATHER     =  7#,
WOOD        =  8#,
BONE        =  9#,
DRAGON_HIDE = 10#, /* not leather! */
IRON        = 11#, /* Fe - includes steel */
METAL       = 12#, /* Sn, &c. */
COPPER      = 13#, /* Cu - includes brass */
SILVER      = 14#, /* Ag */
GOLD        = 15#, /* Au */
PLATINUM    = 16#, /* Pt */
MITHRIL     = 17#,
PLASTIC     = 18#,
GLASS       = 19#,
GEMSTONE    = 20#,
MINERAL     = 21#


PIERCE=1 #/* for weapons & tools used as weapons */
SLASH=2  #/* (latter includes iron ball & chain) */
WHACK=0

ARM_SUIT   = 0#,
ARM_SHIELD = 1#,        /* needed for special wear function */
ARM_HELM   = 2#,
ARM_GLOVES = 3#,
ARM_BOOTS  = 4#,
ARM_CLOAK  = 5#,
ARM_SHIRT  = 6#
####################


#################### https://github.com/facebookresearch/nle/blob/master/include/prop.h
FIRE_RES          =  1#,
COLD_RES          =  2#,
SLEEP_RES         =  3#,
DISINT_RES        =  4#,
SHOCK_RES         =  5#,
POISON_RES        =  6#,
ACID_RES          =  7#,
STONE_RES         =  8#,
#/* note: for the first eight properties, MR_xxx == (1 << (xxx_RES - 1)) */
DRAIN_RES         =  9#,
SICK_RES          = 10#,
INVULNERABLE      = 11#,
ANTIMAGIC         = 12#,
#/* Troubles */
STUNNED           = 13#,
CONFUSION         = 14#,
BLINDED           = 15#,
DEAF              = 16#,
SICK              = 17#,
STONED            = 18#,
STRANGLED         = 19#,
VOMITING          = 20#,
GLIB              = 21#,
SLIMED            = 22#,
HALLUC            = 23#,
HALLUC_RES        = 24#,
FUMBLING          = 25#,
WOUNDED_LEGS      = 26#,
SLEEPY            = 27#,
HUNGER            = 28#,
#/* Vision and senses */
SEE_INVIS         = 29#,
TELEPAT           = 30#,
WARNING           = 31#,
WARN_OF_MON       = 32#,
WARN_UNDEAD       = 33#,
SEARCHING         = 34#,
CLAIRVOYANT       = 35#,
INFRAVISION       = 36#,
DETECT_MONSTERS   = 37#,
#/* Appearance and behavior */
ADORNED           = 38#,
INVIS             = 39#,
DISPLACED         = 40#,
STEALTH           = 41#,
AGGRAVATE_MONSTER = 42#,
CONFLICT          = 43#,
#/* Transportation */
JUMPING           = 44#,
TELEPORT          = 45#,
TELEPORT_CONTROL  = 46#,
LEVITATION        = 47#,
FLYING            = 48#,
WWALKING          = 49#,
SWIMMING          = 50#,
MAGICAL_BREATHING = 51#,
PASSES_WALLS      = 52#,
#/* Physical attributes */
SLOW_DIGESTION    = 53#,
HALF_SPDAM        = 54#,
HALF_PHDAM        = 55#,
REGENERATION      = 56#,
ENERGY_REGENERATION = 57#,
PROTECTION        = 58#,
PROT_FROM_SHAPE_CHANGERS = 59#,
POLYMORPH         = 60#,
POLYMORPH_CONTROL = 61#,
UNCHANGING        = 62#,
FAST              = 63#,
REFLECTING        = 64#,
FREE_ACTION       = 65#,
FIXED_ABIL        = 66#,
LIFESAVED         = 67#
####################

#################### https://github.com/facebookresearch/nle/blob/master/include/color.h
# these are already in nle.nethack.objclass(x).oc_color

CLR_BLACK=0
CLR_RED=1
CLR_GREEN=2
CLR_BROWN=3 #/* on IBM, low-intensity yellow is brown */
CLR_BLUE=4
CLR_MAGENTA=5
CLR_CYAN=6
CLR_GRAY=7 #/* low-intensity white */
NO_COLOR=8
CLR_ORANGE=9
CLR_BRIGHT_GREEN=10
CLR_YELLOW=11
CLR_BRIGHT_BLUE=12
CLR_BRIGHT_MAGENTA=13
CLR_BRIGHT_CYAN=14
CLR_WHITE=15
CLR_MAX=16

HI_OBJ=CLR_MAGENTA
HI_METAL=CLR_CYAN
HI_COPPER=CLR_YELLOW
HI_SILVER=CLR_GRAY
HI_GOLD=CLR_YELLOW
HI_LEATHER=CLR_BROWN
HI_CLOTH=CLR_BROWN
HI_ORGANIC=CLR_BROWN
HI_WOOD=CLR_BROWN
HI_PAPER=CLR_WHITE
HI_GLASS=CLR_BRIGHT_CYAN
HI_MINERAL=CLR_GRAY
DRAGON_SILVER=CLR_BRIGHT_CYAN
HI_ZAP=CLR_BRIGHT_BLUE
####################



##define WEAPON(name,desc,kn,mg,bi,prob,wt,                \
#               cost,sdam,ldam,hitbon,typ,sub,metal,color) \
#    OBJECT(OBJ(name,desc),                                          \
#           BITS(kn, mg, 1, 0, 0, 1, 0, 0, bi, 0, typ, sub, metal),  \
#           0, WEAPON_CLASS, prob, 0, wt,                            \
#           cost, sdam, ldam, hitbon, 0, wt, color)
def WEAPON(*args):
    return Weapon(*args, damage_small=None, damage_large=None)

##define PROJECTILE(name,desc,kn,prob,wt,                  \
#                   cost,sdam,ldam,hitbon,metal,sub,color) \
#    OBJECT(OBJ(name,desc),                                          \
#           BITS(kn, 1, 1, 0, 0, 1, 0, 0, 0, 0, PIERCE, sub, metal), \
#           0, WEAPON_CLASS, prob, 0, wt,                            \
#           cost, sdam, ldam, hitbon, 0, wt, color)
def PROJECTILE(name,desc,kn,prob,wt,cost,sdam,ldam,hitbon,metal,sub,color):
    return WEAPON(name, desc, kn, 1, 0, prob, wt, cost, sdam, ldam, hitbon, PIERCE, sub, metal, color)

##define BOW(name,desc,kn,prob,wt,cost,hitbon,metal,sub,color) \
#    OBJECT(OBJ(name,desc),                                          \
#           BITS(kn, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, sub, metal),      \
#           0, WEAPON_CLASS, prob, 0, wt,                            \
#           cost, 2, 2, hitbon, 0, wt, color)
def BOW(name,desc,kn,prob,wt,cost,hitbon,metal,sub,color):
    return WEAPON(name, desc, kn, 0, 0, prob, wt, cost, 2, 2, hitbon, 0, sub, metal, color)

##define ARMOR(name,desc,kn,mgc,blk,power,prob,delay,wt,  \
#            cost,ac,can,sub,metal,c)                   \
#    OBJECT(OBJ(name, desc),                                         \
#        BITS(kn, 0, 1, 0, mgc, 1, 0, 0, blk, 0, 0, sub, metal),  \
#        power, ARMOR_CLASS, prob, delay, wt,                     \
#        cost, 0, 0, 10 - ac, can, wt, c)
def ARMOR(*args):
    return Armor(*args)

##define HELM(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c)  \
#    ARMOR(name, desc, kn, mgc, 0, power, prob, delay, wt,  \
#        cost, ac, can, ARM_HELM, metal, c)
def HELM(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c):
    return ARMOR(name, desc, kn, mgc, 0, power, prob, delay, wt, \
                 cost, ac, can, ARM_HELM, metal, c)

##define CLOAK(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c)  \
#    ARMOR(name, desc, kn, mgc, 0, power, prob, delay, wt,  \
#        cost, ac, can, ARM_CLOAK, metal, c)
def CLOAK(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c):
    return ARMOR(name, desc, kn, mgc, 0, power, prob, delay, wt, \
                 cost, ac, can, ARM_CLOAK, metal, c)

##define SHIELD(name,desc,kn,mgc,blk,power,prob,delay,wt,cost,ac,can,metal,c) \
#    ARMOR(name, desc, kn, mgc, blk, power, prob, delay, wt, \
#        cost, ac, can, ARM_SHIELD, metal, c)
def SHIELD(name,desc,kn,mgc,blk,power,prob,delay,wt,cost,ac,can,metal,c):
    return ARMOR(name, desc, kn, mgc, blk, power, prob, delay, wt, \
                 cost, ac, can, ARM_SHIELD, metal, c)

##define GLOVES(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c)  \
#    ARMOR(name, desc, kn, mgc, 0, power, prob, delay, wt,  \
#        cost, ac, can, ARM_GLOVES, metal, c)
def GLOVES(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c):
    return ARMOR(name, desc, kn, mgc, 0, power, prob, delay, wt, \
                 cost, ac, can, ARM_GLOVES, metal, c)

##define BOOTS(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c)  \
#    ARMOR(name, desc, kn, mgc, 0, power, prob, delay, wt,  \
#        cost, ac, can, ARM_BOOTS, metal, c)
def BOOTS(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c):
    return ARMOR(name, desc, kn, mgc, 0, power, prob, delay, wt, \
                 cost, ac, can, ARM_BOOTS, metal, c)

##define DRGN_ARMR(name,mgc,power,cost,ac,color)  \
#    ARMOR(name, None, 1, mgc, 1, power, 0, 5, 40,  \
#        cost, ac, 0, ARM_SUIT, DRAGON_HIDE, color)
def DRGN_ARMR(name,mgc,power,cost,ac,color):
    return ARMOR(name, None, 1, mgc, 1, power, 0, 5, 40, \
                 cost, ac, 0, ARM_SUIT, DRAGON_HIDE, color)

#################### https://github.com/facebookresearch/nle/blob/master/src/objects.c
P=PIERCE
S=SLASH
B=WHACK

objects = [
    None,

    PROJECTILE("arrow", None,
            1, 55, 1, 2, 6, 6, 0,        IRON, -P_BOW, HI_METAL),
    PROJECTILE("elven arrow", "runed arrow",
            0, 20, 1, 2, 7, 6, 0,        WOOD, -P_BOW, HI_WOOD),
    PROJECTILE("orcish arrow", "crude arrow",
            0, 20, 1, 2, 5, 6, 0,        IRON, -P_BOW, CLR_BLACK),
    PROJECTILE("silver arrow", None,
            1, 12, 1, 5, 6, 6, 0,        SILVER, -P_BOW, HI_SILVER),
    PROJECTILE("ya", "bamboo arrow",
            0, 15, 1, 4, 7, 7, 1,        METAL, -P_BOW, HI_METAL),
    PROJECTILE("crossbow bolt", None,
            1, 55, 1, 2, 4, 6, 0,        IRON, -P_CROSSBOW, HI_METAL),

    #/* missiles that don't use a launcher */
    WEAPON("dart", None,
        1, 1, 0, 60,   1,   2,  3,  2, 0, P,   -P_DART, IRON, HI_METAL),
    WEAPON("shuriken", "throwing star",
        0, 1, 0, 35,   1,   5,  8,  6, 2, P,   -P_SHURIKEN, IRON, HI_METAL),
    WEAPON("boomerang", None,
        1, 1, 0, 15,   5,  20,  9,  9, 0, 0,   -P_BOOMERANG, WOOD, HI_WOOD),

    #/* spears [note: javelin used to have a separate skill from spears,
    #   because the latter are primarily stabbing weapons rather than
    #   throwing ones; but for playability, they've been merged together
    #   under spear skill and spears can now be thrown like javelins] */
    WEAPON("spear", None,
        1, 1, 0, 50,  30,   3,  6,  8, 0, P,   P_SPEAR, IRON, HI_METAL),
    WEAPON("elven spear", "runed spear",
        0, 1, 0, 10,  30,   3,  7,  8, 0, P,   P_SPEAR, WOOD, HI_WOOD),
    WEAPON("orcish spear", "crude spear",
        0, 1, 0, 13,  30,   3,  5,  8, 0, P,   P_SPEAR, IRON, CLR_BLACK),
    WEAPON("dwarvish spear", "stout spear",
        0, 1, 0, 12,  35,   3,  8,  8, 0, P,   P_SPEAR, IRON, HI_METAL),
    WEAPON("silver spear", None,
        1, 1, 0,  2,  36,  40,  6,  8, 0, P,   P_SPEAR, SILVER, HI_SILVER),
    WEAPON("javelin", "throwing spear",
        0, 1, 0, 10,  20,   3,  6,  6, 0, P,   P_SPEAR, IRON, HI_METAL),

    #/* spearish; doesn't stack, not intended to be thrown */
    WEAPON("trident", None,
        1, 0, 0,  8,  25,   5,  6,  4, 0, P,   P_TRIDENT, IRON, HI_METAL),
            #/* +1 small, +2d4 large */

    #/* blades; all stack */
    WEAPON("dagger", None,
        1, 1, 0, 30,  10,   4,  4,  3, 2, P,   P_DAGGER, IRON, HI_METAL),
    WEAPON("elven dagger", "runed dagger",
        0, 1, 0, 10,  10,   4,  5,  3, 2, P,   P_DAGGER, WOOD, HI_WOOD),
    WEAPON("orcish dagger", "crude dagger",
        0, 1, 0, 12,  10,   4,  3,  3, 2, P,   P_DAGGER, IRON, CLR_BLACK),
    WEAPON("silver dagger", None,
        1, 1, 0,  3,  12,  40,  4,  3, 2, P,   P_DAGGER, SILVER, HI_SILVER),
    WEAPON("athame", None,
        1, 1, 0,  0,  10,   4,  4,  3, 2, S,   P_DAGGER, IRON, HI_METAL),
    WEAPON("scalpel", None,
        1, 1, 0,  0,   5,   6,  3,  3, 2, S,   P_KNIFE, METAL, HI_METAL),
    WEAPON("knife", None,
        1, 1, 0, 20,   5,   4,  3,  2, 0, P|S, P_KNIFE, IRON, HI_METAL),
    WEAPON("stiletto", None,
        1, 1, 0,  5,   5,   4,  3,  2, 0, P|S, P_KNIFE, IRON, HI_METAL),
    #/* 3.6: worm teeth and crysknives now stack;
    #   when a stack of teeth is enchanted at once, they fuse into one crysknife;
    #   when a stack of crysknives drops, the whole stack reverts to teeth */
    WEAPON("worm tooth", None,
        1, 1, 0,  0,  20,   2,  2,  2, 0, 0,   P_KNIFE, 0, CLR_WHITE),
    WEAPON("crysknife", None,
        1, 1, 0,  0,  20, 100, 10, 10, 3, P,   P_KNIFE, MINERAL, CLR_WHITE),

    #/* axes */
    WEAPON("axe", None,
        1, 0, 0, 40,  60,   8,  6,  4, 0, S,   P_AXE, IRON, HI_METAL),
    WEAPON("battle-axe", "double-headed axe",       #/* "double-bitted"? */
        0, 0, 1, 10, 120,  40,  8,  6, 0, S,   P_AXE, IRON, HI_METAL),

    #/* swords */
    WEAPON("short sword", None,
        1, 0, 0,  8,  30,  10,  6,  8, 0, P,   P_SHORT_SWORD, IRON, HI_METAL),
    WEAPON("elven short sword", "runed short sword",
        0, 0, 0,  2,  30,  10,  8,  8, 0, P,   P_SHORT_SWORD, WOOD, HI_WOOD),
    WEAPON("orcish short sword", "crude short sword",
        0, 0, 0,  3,  30,  10,  5,  8, 0, P,   P_SHORT_SWORD, IRON, CLR_BLACK),
    WEAPON("dwarvish short sword", "broad short sword",
        0, 0, 0,  2,  30,  10,  7,  8, 0, P,   P_SHORT_SWORD, IRON, HI_METAL),
    WEAPON("scimitar", "curved sword",
        0, 0, 0, 15,  40,  15,  8,  8, 0, S,   P_SCIMITAR, IRON, HI_METAL),
    WEAPON("silver saber", None,
        1, 0, 0,  6,  40,  75,  8,  8, 0, S,   P_SABER, SILVER, HI_SILVER),
    WEAPON("broadsword", None,
        1, 0, 0,  8,  70,  10,  4,  6, 0, S,   P_BROAD_SWORD, IRON, HI_METAL),
            #/* +d4 small, +1 large */
    WEAPON("elven broadsword", "runed broadsword",
        0, 0, 0,  4,  70,  10,  6,  6, 0, S,   P_BROAD_SWORD, WOOD, HI_WOOD),
            #/* +d4 small, +1 large */
    WEAPON("long sword", None,
        1, 0, 0, 50,  40,  15,  8, 12, 0, S,   P_LONG_SWORD, IRON, HI_METAL),
    WEAPON("two-handed sword", None,
        1, 0, 1, 22, 150,  50, 12,  6, 0, S,   P_TWO_HANDED_SWORD,
                                                                IRON, HI_METAL),
            #/* +2d6 large */
    WEAPON("katana", "samurai sword",
        0, 0, 0,  4,  40,  80, 10, 12, 1, S,   P_LONG_SWORD, IRON, HI_METAL),
    #/* special swords set up for artifacts */
    WEAPON("tsurugi", "long samurai sword",
        0, 0, 1,  0,  60, 500, 16,  8, 2, S,   P_TWO_HANDED_SWORD,
                                                                METAL, HI_METAL),
            #/* +2d6 large */
    WEAPON("runesword", "runed broadsword",
        0, 0, 0,  0,  40, 300,  4,  6, 0, S,   P_BROAD_SWORD, IRON, CLR_BLACK),
            #/* +d4 small, +1 large; Stormbringer: +5d2 +d8 from level drain */

    #/* polearms */
    #/* spear-type */
    WEAPON("partisan", "vulgar polearm",
        0, 0, 1,  5,  80,  10,  6,  6, 0, P,   P_POLEARMS, IRON, HI_METAL),
            #/* +1 large */
    WEAPON("ranseur", "hilted polearm",
        0, 0, 1,  5,  50,   6,  4,  4, 0, P,   P_POLEARMS, IRON, HI_METAL),
            #/* +d4 both */
    WEAPON("spetum", "forked polearm",
        0, 0, 1,  5,  50,   5,  6,  6, 0, P,   P_POLEARMS, IRON, HI_METAL),
            #/* +1 small, +d6 large */
    WEAPON("glaive", "single-edged polearm",
        0, 0, 1,  8,  75,   6,  6, 10, 0, S,   P_POLEARMS, IRON, HI_METAL),
    WEAPON("lance", None,
        1, 0, 0,  4, 180,  10,  6,  8, 0, P,   P_LANCE, IRON, HI_METAL),
            #/* +2d10 when jousting with lance as primary weapon */
    #/* axe-type */
    WEAPON("halberd", "angled poleaxe",
        0, 0, 1,  8, 150,  10, 10,  6, 0, P|S, P_POLEARMS, IRON, HI_METAL),
            #/* +1d6 large */
    WEAPON("bardiche", "long poleaxe",
        0, 0, 1,  4, 120,   7,  4,  4, 0, S,   P_POLEARMS, IRON, HI_METAL),
            #/* +1d4 small, +2d4 large */
    WEAPON("voulge", "pole cleaver",
        0, 0, 1,  4, 125,   5,  4,  4, 0, S,   P_POLEARMS, IRON, HI_METAL),
            #/* +d4 both */
    WEAPON("dwarvish mattock", "broad pick",
        0, 0, 1, 13, 120,  50, 12,  8, -1, B,  P_PICK_AXE, IRON, HI_METAL),
    #/* curved/hooked */
    WEAPON("fauchard", "pole sickle",
        0, 0, 1,  6,  60,   5,  6,  8, 0, P|S, P_POLEARMS, IRON, HI_METAL),
    WEAPON("guisarme", "pruning hook",
        0, 0, 1,  6,  80,   5,  4,  8, 0, S,   P_POLEARMS, IRON, HI_METAL),
            #/* +1d4 small */
    WEAPON("bill-guisarme", "hooked polearm",
        0, 0, 1,  4, 120,   7,  4, 10, 0, P|S, P_POLEARMS, IRON, HI_METAL),
            #/* +1d4 small */
    #/* other */
    WEAPON("lucern hammer", "pronged polearm",
        0, 0, 1,  5, 150,   7,  4,  6, 0, B|P, P_POLEARMS, IRON, HI_METAL),
            #/* +1d4 small */
    WEAPON("bec de corbin", "beaked polearm",
        0, 0, 1,  4, 100,   8,  8,  6, 0, B|P, P_POLEARMS, IRON, HI_METAL),

    #/* bludgeons */
    WEAPON("mace", None,
        1, 0, 0, 40,  30,   5,  6,  6, 0, B,   P_MACE, IRON, HI_METAL),
            #/* +1 small */
    WEAPON("morning star", None,
        1, 0, 0, 12, 120,  10,  4,  6, 0, B,   P_MORNING_STAR, IRON, HI_METAL),
            #/* +d4 small, +1 large */
    WEAPON("war hammer", None,
        1, 0, 0, 15,  50,   5,  4,  4, 0, B,   P_HAMMER, IRON, HI_METAL),
            #/* +1 small */
    WEAPON("club", None,
        1, 0, 0, 12,  30,   3,  6,  3, 0, B,   P_CLUB, WOOD, HI_WOOD),
    WEAPON("rubber hose", None,
        1, 0, 0,  0,  20,   3,  4,  3, 0, B,   P_WHIP, PLASTIC, CLR_BROWN),
    WEAPON("quarterstaff", "staff",
        0, 0, 1, 11,  40,   5,  6,  6, 0, B,   P_QUARTERSTAFF, WOOD, HI_WOOD),
    #/* two-piece */
    WEAPON("aklys", "thonged club",
        0, 0, 0,  8,  15,   4,  6,  3, 0, B,   P_CLUB, IRON, HI_METAL),
    WEAPON("flail", None,
        1, 0, 0, 40,  15,   4,  6,  4, 0, B,   P_FLAIL, IRON, HI_METAL),
            #/* +1 small, +1d4 large */

    #/* misc */
    WEAPON("bullwhip", None,
        1, 0, 0,  2,  20,   4,  2,  1, 0, 0,   P_WHIP, LEATHER, CLR_BROWN),

    #/* bows */
    BOW("bow", None,               1, 24, 30, 60, 0, WOOD, P_BOW, HI_WOOD),
    BOW("elven bow", "runed bow",  0, 12, 30, 60, 0, WOOD, P_BOW, HI_WOOD),
    BOW("orcish bow", "crude bow", 0, 12, 30, 60, 0, WOOD, P_BOW, CLR_BLACK),
    BOW("yumi", "long bow",        0,  0, 30, 60, 0, WOOD, P_BOW, HI_WOOD),
    BOW("sling", None,             1, 40,  3, 20, 0, LEATHER, P_SLING, HI_LEATHER),
    BOW("crossbow", None,          1, 45, 50, 40, 0, WOOD, P_CROSSBOW, HI_WOOD),






    #/* helmets */
    HELM("elven leather helm", "leather hat",
        0, 0,           0,  6, 1,  3,  8,  9, 0, LEATHER, HI_LEATHER),
    HELM("orcish helm", "iron skull cap",
        0, 0,           0,  6, 1, 30, 10,  9, 0, IRON, CLR_BLACK),
    HELM("dwarvish iron helm", "hard hat",
        0, 0,           0,  6, 1, 40, 20,  8, 0, IRON, HI_METAL),
    HELM("fedora", None,
        1, 0,           0,  0, 0,  3,  1, 10, 0, CLOTH, CLR_BROWN),
    HELM("cornuthaum", "conical hat",
        0, 1, CLAIRVOYANT,  3, 1,  4, 80, 10, 1, CLOTH, CLR_BLUE),
            #/* name coined by devteam; confers clairvoyance for wizards,
            #blocks clairvoyance if worn by role other than wizard */
    HELM("dunce cap", "conical hat",
        0, 1,           0,  3, 1,  4,  1, 10, 0, CLOTH, CLR_BLUE),
    HELM("dented pot", None,
        1, 0,           0,  2, 0, 10,  8,  9, 0, IRON, CLR_BLACK),
    #/* with shuffled appearances... */
    HELM("helmet", "plumed helmet",
        0, 0,           0, 10, 1, 30, 10,  9, 0, IRON, HI_METAL),
    HELM("helm of brilliance", "etched helmet",
        0, 1,           0,  6, 1, 50, 50,  9, 0, IRON, CLR_GREEN),
    HELM("helm of opposite alignment", "crested helmet",
        0, 1,           0,  6, 1, 50, 50,  9, 0, IRON, HI_METAL),
    HELM("helm of telepathy", "visored helmet",
        0, 1,     TELEPAT,  2, 1, 50, 50,  9, 0, IRON, HI_METAL),

    #/* suits of armor */
    #/*
    # * There is code in polyself.c that assumes (1) and (2).
    # * There is code in obj.h, objnam.c, mon.c, read.c that assumes (2).
    # *      (1) The dragon scale mails and the dragon scales are together.
    # *      (2) That the order of the dragon scale mail and dragon scales
    # *          is the same as order of dragons defined in monst.c.
    # */
    #/* 3.4.1: dragon scale mail reclassified as "magic" since magic is
    #needed to create them */
    DRGN_ARMR("gray dragon scale mail",    1, ANTIMAGIC,  1200, 1, CLR_GRAY),
    DRGN_ARMR("silver dragon scale mail",  1, REFLECTING, 1200, 1, DRAGON_SILVER),
    ##if 0 /* DEFERRED */
    #DRGN_ARMR("shimmering dragon scale mail", 1, DISPLACED, 1200, 1, CLR_CYAN),
    ##endif
    DRGN_ARMR("red dragon scale mail",     1, FIRE_RES,    900, 1, CLR_RED),
    DRGN_ARMR("white dragon scale mail",   1, COLD_RES,    900, 1, CLR_WHITE),
    DRGN_ARMR("orange dragon scale mail",  1, SLEEP_RES,   900, 1, CLR_ORANGE),
    DRGN_ARMR("black dragon scale mail",   1, DISINT_RES, 1200, 1, CLR_BLACK),
    DRGN_ARMR("blue dragon scale mail",    1, SHOCK_RES,   900, 1, CLR_BLUE),
    DRGN_ARMR("green dragon scale mail",   1, POISON_RES,  900, 1, CLR_GREEN),
    DRGN_ARMR("yellow dragon scale mail",  1, ACID_RES,    900, 1, CLR_YELLOW),
    #/* For now, only dragons leave these. */
    #/* 3.4.1: dragon scales left classified as "non-magic"; they confer
    #   magical properties but are produced "naturally" */
    DRGN_ARMR("gray dragon scales",        0, ANTIMAGIC,   700, 7, CLR_GRAY),
    DRGN_ARMR("silver dragon scales",      0, REFLECTING,  700, 7, DRAGON_SILVER),
    ##if 0 /* DEFERRED */
    #DRGN_ARMR("shimmering dragon scales",  0, DISPLACED,   700, 7, CLR_CYAN),
    ##endif
    DRGN_ARMR("red dragon scales",         0, FIRE_RES,    500, 7, CLR_RED),
    DRGN_ARMR("white dragon scales",       0, COLD_RES,    500, 7, CLR_WHITE),
    DRGN_ARMR("orange dragon scales",      0, SLEEP_RES,   500, 7, CLR_ORANGE),
    DRGN_ARMR("black dragon scales",       0, DISINT_RES,  700, 7, CLR_BLACK),
    DRGN_ARMR("blue dragon scales",        0, SHOCK_RES,   500, 7, CLR_BLUE),
    DRGN_ARMR("green dragon scales",       0, POISON_RES,  500, 7, CLR_GREEN),
    DRGN_ARMR("yellow dragon scales",      0, ACID_RES,    500, 7, CLR_YELLOW),
    #undef DRGN_ARMR
    #/* other suits */
    ARMOR("plate mail", None,
        1, 0, 1,  0, 44, 5, 450, 600,  3, 2,  ARM_SUIT, IRON, HI_METAL),
    ARMOR("crystal plate mail", None,
        1, 0, 1,  0, 10, 5, 450, 820,  3, 2,  ARM_SUIT, GLASS, CLR_WHITE),
    ARMOR("bronze plate mail", None,
        1, 0, 1,  0, 25, 5, 450, 400,  4, 1,  ARM_SUIT, COPPER, HI_COPPER),
    ARMOR("splint mail", None,
        1, 0, 1,  0, 62, 5, 400,  80,  4, 1,  ARM_SUIT, IRON, HI_METAL),
    ARMOR("banded mail", None,
        1, 0, 1,  0, 72, 5, 350,  90,  4, 1,  ARM_SUIT, IRON, HI_METAL),
    ARMOR("dwarvish mithril-coat", None,
        1, 0, 0,  0, 10, 1, 150, 240,  4, 2,  ARM_SUIT, MITHRIL, HI_SILVER),
    ARMOR("elven mithril-coat", None,
        1, 0, 0,  0, 15, 1, 150, 240,  5, 2,  ARM_SUIT, MITHRIL, HI_SILVER),
    ARMOR("chain mail", None,
        1, 0, 0,  0, 72, 5, 300,  75,  5, 1,  ARM_SUIT, IRON, HI_METAL),
    ARMOR("orcish chain mail", "crude chain mail",
        0, 0, 0,  0, 20, 5, 300,  75,  6, 1,  ARM_SUIT, IRON, CLR_BLACK),
    ARMOR("scale mail", None,
        1, 0, 0,  0, 72, 5, 250,  45,  6, 1,  ARM_SUIT, IRON, HI_METAL),
    ARMOR("studded leather armor", None,
        1, 0, 0,  0, 72, 3, 200,  15,  7, 1,  ARM_SUIT, LEATHER, HI_LEATHER),
    ARMOR("ring mail", None,
        1, 0, 0,  0, 72, 5, 250, 100,  7, 1,  ARM_SUIT, IRON, HI_METAL),
    ARMOR("orcish ring mail", "crude ring mail",
        0, 0, 0,  0, 20, 5, 250,  80,  8, 1,  ARM_SUIT, IRON, CLR_BLACK),
    ARMOR("leather armor", None,
        1, 0, 0,  0, 82, 3, 150,   5,  8, 1,  ARM_SUIT, LEATHER, HI_LEATHER),
    ARMOR("leather jacket", None,
        1, 0, 0,  0, 12, 0,  30,  10,  9, 0,  ARM_SUIT, LEATHER, CLR_BLACK),

    #/* shirts */
    ARMOR("Hawaiian shirt", None,
        1, 0, 0,  0,  8, 0,   5,   3, 10, 0,  ARM_SHIRT, CLOTH, CLR_MAGENTA),
    ARMOR("T-shirt", None,
        1, 0, 0,  0,  2, 0,   5,   2, 10, 0,  ARM_SHIRT, CLOTH, CLR_WHITE),

    #/* cloaks */
    CLOAK("mummy wrapping", None,
        1, 0,          0,  0, 0,  3,  2, 10, 1,  CLOTH, CLR_GRAY),
            #/* worn mummy wrapping blocks invisibility */
    CLOAK("elven cloak", "faded pall",
        0, 1,    STEALTH,  8, 0, 10, 60,  9, 1,  CLOTH, CLR_BLACK),
    CLOAK("orcish cloak", "coarse mantelet",
        0, 0,          0,  8, 0, 10, 40, 10, 1,  CLOTH, CLR_BLACK),
    CLOAK("dwarvish cloak", "hooded cloak",
        0, 0,          0,  8, 0, 10, 50, 10, 1,  CLOTH, HI_CLOTH),
    CLOAK("oilskin cloak", "slippery cloak",
        0, 0,          0,  8, 0, 10, 50,  9, 2,  CLOTH, HI_CLOTH),
    CLOAK("robe", None,
        1, 1,          0,  3, 0, 15, 50,  8, 2,  CLOTH, CLR_RED),
            #/* robe was adopted from slash'em, where it's worn as a suit
            #rather than as a cloak and there are several variations */
    CLOAK("alchemy smock", "apron",
        0, 1, POISON_RES,  9, 0, 10, 50,  9, 1,  CLOTH, CLR_WHITE),
    CLOAK("leather cloak", None,
        1, 0,          0,  8, 0, 15, 40,  9, 1,  LEATHER, CLR_BROWN),
    #/* with shuffled appearances... */
    CLOAK("cloak of protection", "tattered cape",
        0, 1, PROTECTION,  9, 0, 10, 50,  7, 3,  CLOTH, HI_CLOTH),
            #/* cloak of protection is now the only item conferring MC 3 */
    CLOAK("cloak of invisibility", "opera cloak",
        0, 1,      INVIS, 10, 0, 10, 60,  9, 1,  CLOTH, CLR_BRIGHT_MAGENTA),
    CLOAK("cloak of magic resistance", "ornamental cope",
        0, 1,  ANTIMAGIC,  2, 0, 10, 60,  9, 1,  CLOTH, CLR_WHITE),
            #/*  'cope' is not a spelling mistake... leave it be */
    CLOAK("cloak of displacement", "piece of cloth",
        0, 1,  DISPLACED, 10, 0, 10, 50,  9, 1,  CLOTH, HI_CLOTH),

    #/* shields */
    SHIELD("small shield", None,
        1, 0, 0,          0, 6, 0,  30,  3, 9, 0,  WOOD, HI_WOOD),
    SHIELD("elven shield", "blue and green shield",
        0, 0, 0,          0, 2, 0,  40,  7, 8, 0,  WOOD, CLR_GREEN),
    SHIELD("Uruk-hai shield", "white-handed shield",
        0, 0, 0,          0, 2, 0,  50,  7, 9, 0,  IRON, HI_METAL),
    SHIELD("orcish shield", "red-eyed shield",
        0, 0, 0,          0, 2, 0,  50,  7, 9, 0,  IRON, CLR_RED),
    SHIELD("large shield", None,
        1, 0, 1,          0, 7, 0, 100, 10, 8, 0,  IRON, HI_METAL),
    SHIELD("dwarvish roundshield", "large round shield",
        0, 0, 0,          0, 4, 0, 100, 10, 8, 0,  IRON, HI_METAL),
    SHIELD("shield of reflection", "polished silver shield",
        0, 1, 0, REFLECTING, 3, 0,  50, 50, 8, 0,  SILVER, HI_SILVER),

    #/* gloves */
    #/* These have their color but not material shuffled, so the IRON must
    # * stay CLR_BROWN (== HI_LEATHER) even though it's normally either
    # * HI_METAL or CLR_BLACK.  All have shuffled descriptions.
    # */
    GLOVES("leather gloves", "old gloves",
        0, 0,        0, 16, 1, 10,  8, 9, 0,  LEATHER, HI_LEATHER),
    GLOVES("gauntlets of fumbling", "padded gloves",
        0, 1, FUMBLING,  8, 1, 10, 50, 9, 0,  LEATHER, HI_LEATHER),
    GLOVES("gauntlets of power", "riding gloves",
        0, 1,        0,  8, 1, 30, 50, 9, 0,  IRON, CLR_BROWN),
    GLOVES("gauntlets of dexterity", "fencing gloves",
        0, 1,        0,  8, 1, 10, 50, 9, 0,  LEATHER, HI_LEATHER),

    #/* boots */
    BOOTS("low boots", "walking shoes",
        0, 0,          0, 25, 2, 10,  8, 9, 0, LEATHER, HI_LEATHER),
    BOOTS("iron shoes", "hard shoes",
        0, 0,          0,  7, 2, 50, 16, 8, 0, IRON, HI_METAL),
    BOOTS("high boots", "jackboots",
        0, 0,          0, 15, 2, 20, 12, 8, 0, LEATHER, HI_LEATHER),
    #/* with shuffled appearances... */
    BOOTS("speed boots", "combat boots",
          0, 1,       FAST, 12, 2, 20, 50, 9, 0, LEATHER, HI_LEATHER),
    BOOTS("water walking boots", "jungle boots",
        0, 1,   WWALKING, 12, 2, 15, 50, 9, 0, LEATHER, HI_LEATHER),
    BOOTS("jumping boots", "hiking boots",
        0, 1,    JUMPING, 12, 2, 20, 50, 9, 0, LEATHER, HI_LEATHER),
    BOOTS("elven boots", "mud boots",
        0, 1,    STEALTH, 12, 2, 15,  8, 9, 0, LEATHER, HI_LEATHER),
    BOOTS("kicking boots", "buckled boots",
        0, 1,          0, 12, 2, 50,  8, 9, 0, IRON, CLR_BROWN),
            #/* CLR_BROWN for same reason as gauntlets of power */
    BOOTS("fumble boots", "riding boots",
        0, 1,   FUMBLING, 12, 2, 20, 30, 9, 0, LEATHER, HI_LEATHER),
    BOOTS("levitation boots", "snow boots",
        0, 1, LEVITATION, 12, 2, 15, 30, 9, 0, LEATHER, HI_LEATHER),
    ]
####################

# Taken from: https://nethackwiki.com/wiki/Weapon#Table_of_weapons_and_their_properties
weapon_damage = [
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
    ('bec de corbin', 'polearms', 'd8', 'd6'),
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
]

assert len(set([w.name for w in objects if isinstance(w, Weapon)]) - set([w[0] for w in weapon_damage])) == 0
for name, _, damage_small, damage_large in weapon_damage:
    for i in range(len(objects)):
        if objects[i] is None:
            continue

        if objects[i].name == name:
            objects[i] = objects[i]._replace(damage_small=damage_small)
            objects[i] = objects[i]._replace(damage_large=damage_large)
            break


# consistency check
import nle.nethack as nh
for i in range(nh.NUM_OBJECTS):
    if ord(nh.objclass(i).oc_class) == nh.WEAPON_CLASS:
        assert isinstance(objects[i], Weapon)
        d = nh.objdescr.from_idx(i)
        assert d.oc_name == objects[i].name
        assert d.oc_descr == objects[i].desc
        assert objects[i].damage_small is not None and objects[i].damage_large is not None
    elif ord(nh.objclass(i).oc_class) == nh.ARMOR_CLASS:
        assert isinstance(objects[i], Armor)
        d = nh.objdescr.from_idx(i)
        assert d.oc_name == objects[i].name, (d.oc_name, objects[i].name)
        assert d.oc_descr == objects[i].desc
    else:
        assert i >= len(objects) or objects[i] is None


def weapon_from_glyph(i):
    assert nh.glyph_is_object(i)
    assert ord(nh.objclass(nh.glyph_to_obj(i)).oc_class) == nh.WEAPON_CLASS
    return objects[nh.glyph_to_obj(i)]

def armor_from_glyph(i):
    assert nh.glyph_is_object(i)
    assert ord(nh.objclass(nh.glyph_to_obj(i)).oc_class) == nh.ARMOR_CLASS
    return objects[nh.glyph_to_obj(i)]
