import re

import numpy as np
from nle.nethack import actions as A

import objects as O


class Character:
    ARCHEOLOGIST = 0
    BARBARIAN = 1
    CAVEMAN = 2
    HEALER = 3
    KNIGHT = 4
    MONK = 5
    PRIEST = 6
    RANGER = 7
    ROGUE = 8
    SAMURAI = 9
    TOURIST = 10
    VALKYRIE = 11
    WIZARD = 12

    name_to_role = {
        'Archeologist': ARCHEOLOGIST,
        'Barbarian': BARBARIAN,
        'Caveman': CAVEMAN,
        'Cavewoman': CAVEMAN,
        'Healer': HEALER,
        'Knight': KNIGHT,
        'Monk': MONK,
        'Priest': PRIEST,
        'Priestess': PRIEST,
        'Ranger': RANGER,
        'Rogue': ROGUE,
        'Samurai': SAMURAI,
        'Tourist': TOURIST,
        'Valkyrie': VALKYRIE,
        'Wizard': WIZARD,
    }

    CHAOTIC = 0
    NEUTRAL = 1
    LAWFUL = 2

    name_to_alignment = {
        'chaotic': CHAOTIC,
        'neutral': NEUTRAL,
        'lawful': LAWFUL,
    }

    HUMAN = 0
    DWARF = 1
    ELF = 2
    GNOME = 3
    ORC = 4

    name_to_race = {
        'human': HUMAN,
        'dwarf': DWARF,
        'dwarven': DWARF,
        'elf': ELF,
        'elven': ELF,
        'gnome': GNOME,
        'gnomish': GNOME,
        'orc': ORC,
        'orcish': ORC,
    }

    MALE = 0
    FEMALE = 1

    name_to_gender = {
        'male': MALE,
        'female': FEMALE,
    }

    name_to_skill_type = {
        # "": O.P_NONE,
        "dagger": O.P_DAGGER,
        "knife": O.P_KNIFE,
        "axe": O.P_AXE,
        "pick-axe": O.P_PICK_AXE,
        "short sword": O.P_SHORT_SWORD,
        "broadsword": O.P_BROAD_SWORD,
        "long sword": O.P_LONG_SWORD,
        "two-handed sword": O.P_TWO_HANDED_SWORD,
        "scimitar": O.P_SCIMITAR,
        "saber": O.P_SABER,
        "club": O.P_CLUB,
        "mace": O.P_MACE,
        "morning star": O.P_MORNING_STAR,
        "flail": O.P_FLAIL,
        "hammer": O.P_HAMMER,
        "quarterstaff": O.P_QUARTERSTAFF,
        "polearms": O.P_POLEARMS,
        "spear": O.P_SPEAR,
        "trident": O.P_TRIDENT,
        "lance": O.P_LANCE,
        "bow": O.P_BOW,
        "sling": O.P_SLING,
        "crossbow": O.P_CROSSBOW,
        "dart": O.P_DART,
        "shuriken": O.P_SHURIKEN,
        "boomerang": O.P_BOOMERANG,
        "whip": O.P_WHIP,
        "unicorn horn": O.P_UNICORN_HORN,

        "attack spells": O.P_ATTACK_SPELL,
        "healing spells": O.P_HEALING_SPELL,
        "divination spells": O.P_DIVINATION_SPELL,
        "enchantment spells": O.P_ENCHANTMENT_SPELL,
        "clerical spells": O.P_CLERIC_SPELL,
        "escape spells": O.P_ESCAPE_SPELL,
        "matter spells": O.P_MATTER_SPELL,

        "bare handed combat": O.P_BARE_HANDED_COMBAT,
        "martial arts": O.P_BARE_HANDED_COMBAT,
        "two weapon combat": O.P_TWO_WEAPON_COMBAT,
        "riding": O.P_RIDING,
    }

    possible_skill_types = ['Fighting Skills', 'Weapon Skills', 'Spellcasting Skills']
    possible_skill_levels = ['Unskilled', 'Basic', 'Skilled', 'Expert', 'Master', 'Grand Master']

    SKILL_LEVEL_RESTRICTED = 0
    SKILL_LEVEL_UNSKILLED = 1
    SKILL_LEVEL_BASIC = 2
    SKILL_LEVEL_SKILLED = 3
    SKILL_LEVEL_EXPERT = 4
    SKILL_LEVEL_MASTER = 5
    SKILL_LEVEL_GRAND_MASTER = 6

    name_to_skill_level = {k: v for k, v in zip(['Restricted'] + possible_skill_levels,
                                                [SKILL_LEVEL_RESTRICTED,
                                                 SKILL_LEVEL_UNSKILLED, SKILL_LEVEL_BASIC, SKILL_LEVEL_SKILLED,
                                                 SKILL_LEVEL_EXPERT,
                                                 SKILL_LEVEL_MASTER, SKILL_LEVEL_GRAND_MASTER])}

    def __init__(self, agent):
        self.agent = agent
        self.role = None
        self.alignment = None
        self.race = None
        self.gender = None
        self.skill_levels = np.zeros(max(self.name_to_skill_type.values()) + 1, dtype=int)
        self.upgradable_skills = dict()

    def parse(self):
        with self.agent.atom_operation():
            self.agent.step(A.Command.ATTRIBUTES)
            text = ' '.join(self.agent.popup)
            self._parse(text)

    def _parse(self, text):
        matches = re.findall('You are a ([a-z]+) (([a-z]+) )?([a-z]+) ([A-Z][a-z]+).', text)
        if len(matches) == 1:
            alignment, _, gender, race, role = matches[0]
        else:
            matches = re.findall(
                'You are an? ([a-zA-Z ]+), a level (\d+) (([a-z]+) )?([a-z]+) ([A-Z][a-z]+). *You are ([a-z]+)',
                text)
            assert len(matches) == 1, repr(text)
            _, _, _, gender, race, role, alignment = matches[0]

        if not gender:
            if role == 'Priestess':
                gender = 'female'
            elif role == 'Priest':
                gender = 'male'
            elif role == 'Caveman':
                gender = 'male'
            elif role == 'Cavewoman':
                gender = 'female'
            elif role == 'Valkyrie':
                gender = 'female'
            else:
                assert 0, repr(text)

        self.role = self.name_to_role[role]
        self.alignment = self.name_to_alignment[alignment]
        self.race = self.name_to_race[race]
        self.gender = self.name_to_gender[gender]

    def parse_enhance_view(self):
        with self.agent.atom_operation():
            self.agent.step(A.Command.ENHANCE)
            self._parse_enhance_view()
            while self.upgradable_skills:
                to_upgrade = self.select_skill_to_upgrade()
                old_skill_level = self.skill_levels.copy()
                self.agent.step(A.Command.ENHANCE, iter([self.upgradable_skills[to_upgrade]]))
                self.agent.step(A.Command.ENHANCE)
                self._parse_enhance_view()
                assert (old_skill_level != self.skill_levels).any(), (old_skill_level, self.skill_levels)

    def select_skill_to_upgrade(self):
        assert self.upgradable_skills
        # TODO: logic
        return next(iter(self.upgradable_skills.keys()))

    def _parse_enhance_view(self):
        if self.agent.popup[0] not in ('Current skills:', 'Pick a skill to advance:'):
            raise ValueError('Invalid ehance popup text format.' + str(self.agent.popup))
        self.upgradable_skills = dict()
        for line in self.agent.popup[1:]:
            if line.strip() in self.possible_skill_types:
                continue
            matches = re.findall(r'^([a-zA-Z] -)? *' +
                                 r'(' + '|'.join(self.name_to_skill_type.keys()) + ') *' +
                                 r'\[(' + '|'.join(self.possible_skill_levels) + ')\]', line)
            assert len(matches) == 1, (matches, line)
            letter, skill_type, skill_level = matches[0]
            if letter:
                letter = letter[0]
                assert letter not in self.upgradable_skills.values()
                self.upgradable_skills[self.name_to_skill_type[skill_type]] = letter
            self.skill_levels[self.name_to_skill_type[skill_type]] = self.name_to_skill_level[skill_level]

    def __str__(self):
        inv_skill_type = {v: k for k, v in self.name_to_skill_type.items()}
        inv_skill_level = {v: k for k, v in self.name_to_skill_level.items()}
        if self.role is None:
            return 'unparsed_character'
        skill_str = '| '.join(inv_skill_type[skill_type] + '-' + inv_skill_level[level]
                              for skill_type, level in enumerate(self.skill_levels)
                              if level in inv_skill_level and skill_type in inv_skill_type and level != 0)
        if self.upgradable_skills:
            skill_str += '\n Upgradable: ' + '|'.join(letter + '-' + inv_skill_type[skill]
                                                      for skill, letter in self.upgradable_skills.items())
        return '-'.join([f'{list(d.keys())[list(d.values()).index(v)][:3].lower()}'
                         for d, v in [(self.name_to_role, self.role),
                                      (self.name_to_race, self.race),
                                      (self.name_to_gender, self.gender),
                                      (self.name_to_alignment, self.alignment),
                                      ]]) + '\n Skills: ' + skill_str
