from grao_tables_processing.common.singleton import Singleton


class RegexPatternWrapper(metaclass=Singleton):
  def __init__(self):
    # Building regex strings
    cap_letter = '\p{Lu}'
    low_letter = '\p{Ll}'
    separator = '[\||\!]\s*'
    number = '\d+'

    self.year_group = '(\d{4})'
    self.date_group = '(\d{2}-\d{4})'
    self.full_date_group = '(\d{2}-\d{2}-\d{4})'

    name_part = f'[\s|-]*{cap_letter}*'
    name_part_old = f'[\.|\s|-]{cap_letter}*'
    type_abbr = f'{cap_letter}{{1,2}}\.'
    name = f'{cap_letter}+{name_part * 3}'
    name_old = f'{cap_letter}+{name_part_old * 3}'
    word = f'{low_letter}+'
    number_group = f'{separator}({number})\s*'

    self.old_reg = f'ОБЛАСТ:({name_old})'
    self.old_mun = f'ОБЩИНА:({name_old})'
    self.region_name_new = f'{word} ({name}) {word} ({name})'
    self.settlement_info_quarterly = f'({type_abbr}{name})\s*{number_group * 3}'
    self.settlement_info_yearly = f'({type_abbr}{name})\s*{number_group * 6}'
