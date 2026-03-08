from typing import List, Dict


def interval_constraint_seeds(ast: Dict, dfg: Dict) -> List[str]:
    """
    Шаблонные правила для интервалов.
    Пока простые, но достаточно выразительные для LLM.
    В будущем можно выбирать по типу операции ( +, -, * ).
    """
    seeds: List[str] = []

    # базовый шаблон "сдвиг интервала на константу"
    seeds.append(
        "Seed 1 (interval shift by constant): "
        "forall min, max, c, x. (min <= x && x <= max) -> (min+c <= x+c && x+c <= max+c)."
    )

    # частный случай для +1
    seeds.append(
        "Seed 2 (interval shift by 1): "
        "forall min, max, x. (min <= x && x <= max) -> (min+1 <= x+1 && x+1 <= max+1)."
    )

    # шаблон для передачи аннотации через присваивание
    seeds.append(
        "Seed 3 (interval assignment propagation): "
        "if y = x and x : Interval(min,max), then y : Interval(min,max)."
    )

    return seeds
