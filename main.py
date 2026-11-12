#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Планування інформаційної кампанії — курсовий проєкт, розділи 1–3.

Єдина програма, стандартна бібліотека Python, без зовнішніх залежностей.
Два досліджувані алгоритми: жадібний та генетичний (відбір рулеткою).
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from fractions import Fraction
from typing import Callable
import csv
import html
import math
from pathlib import Path
import random
import statistics
import sys
import secrets
from time import perf_counter_ns


# ---------------------------------------------------------------------------
# МОДЕЛЬ ЗАДАЧІ: n, T, c_i, r_i, a_it, B, L; F = sum(r_i * d_i * x_i)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Influencer:
    cost: int                    # c_i (грн)
    reach_per_post: int          # r_i (користувачі)
    days: tuple[int, ...]        # D_i; номери днів з 1 до T


@dataclass(frozen=True)
class Problem:
    name: str
    days_count: int              # T
    budget: int                  # B
    daily_limit: int             # L
    influencers: tuple[Influencer, ...]

    @property
    def n(self) -> int:
        return len(self.influencers)


@dataclass(frozen=True)
class Solution:
    selected: tuple[int, ...]    # X*; 1-based номери інфлюенсерів
    coverage: int                # F*
    cost: int                    # C*
    daily_posts: tuple[int, ...] # q_t, включно з нульовими днями


@dataclass(frozen=True)
class GAParameters:
    population: int = 40         # P
    generations: int = 100       # G
    crossover: float = 0.80     # pc
    mutation: float = 0.05      # pm


@dataclass(frozen=True)
class GAResult:
    solution: Solution
    history: tuple[int, ...]     # рекорд: покоління 0 .. G
    last_improvement: int        # номер покоління, де востаннє оновився рекорд
    repairs: int                 # кількість змінених процедурою repair хромосом
    seed: int


def validate_problem(problem: Problem) -> None:
    if problem.days_count < 1 or problem.budget < 0 or problem.daily_limit < 1:
        raise ValueError('T >= 1, B >= 0, L >= 1 — обов’язкові умови.')
    if problem.n < 1:
        raise ValueError('Потрібен принаймні один інфлюенсер.')
    for i, person in enumerate(problem.influencers, 1):
        if person.cost <= 0 or person.reach_per_post < 0:
            raise ValueError(f'Інфлюенсер {i}: вартість > 0, охоплення >= 0.')
        if not person.days or len(set(person.days)) != len(person.days):
            raise ValueError(f'Інфлюенсер {i}: список днів має бути непорожнім без повторень.')
        if any(t < 1 or t > problem.days_count for t in person.days):
            raise ValueError(f'Інфлюенсер {i}: день поза межами 1..T.')


def validate_ga_parameters(p: GAParameters) -> None:
    if p.population < 2 or p.generations < 1:
        raise ValueError('P >= 2, G >= 1.')
    if not (0 <= p.crossover <= 1 and 0 <= p.mutation <= 1):
        raise ValueError('pc і pm повинні належати [0,1].')


class EvaluationContext:
    """Дані з задачі, попередньо підготовлені до багаторазового оцінювання."""
    def __init__(self, problem: Problem):
        validate_problem(problem)
        self.problem = problem
        self.costs = tuple(p.cost for p in problem.influencers)
        self.values = tuple(p.reach_per_post * len(p.days) for p in problem.influencers)
        self.days = tuple(tuple(t - 1 for t in p.days) for p in problem.influencers)
        self.ratios = tuple(Fraction(v, c) for v, c in zip(self.values, self.costs))

    def evaluate(self, chromosome: list[int] | tuple[int, ...]):
        """EvaluateChromosome(x): C(x), F(x), q_t(x), допустимість."""
        cost = 0
        coverage = 0
        q = [0] * self.problem.days_count
        for i, bit in enumerate(chromosome):
            if bit:
                cost += self.costs[i]
                coverage += self.values[i]
                for t in self.days[i]:
                    q[t] += 1
        feasible = cost <= self.problem.budget and all(v <= self.problem.daily_limit for v in q)
        return cost, coverage, q, feasible

    def repair(self, chromosome: list[int]) -> bool:
        """RepairChromosome(x) із підрозділу 2.3.1, in place.

        Якщо є перевантажені дні, кандидатами на видалення стають ТІЛЬКИ
        обрані інфлюенсери, які публікують у ці дні. Інакше — всі обрані.
        Порядок: найменше Ri/ci, за рівності — менше Ri, потім більший №.
        """
        cost, _, q, feasible = self.evaluate(chromosome)
        if feasible:
            return False
        changed = False
        while cost > self.problem.budget or any(v > self.problem.daily_limit for v in q):
            overloaded = {t for t, count in enumerate(q) if count > self.problem.daily_limit}
            if overloaded:
                candidates = [i for i, bit in enumerate(chromosome)
                              if bit and any(t in overloaded for t in self.days[i])]
            else:
                candidates = [i for i, bit in enumerate(chromosome) if bit]
            if not candidates:
                raise RuntimeError('Неможливо виправити хромосому: порушено інваріант.')
            i = min(candidates, key=lambda j: (self.ratios[j], self.values[j], -j))
            chromosome[i] = 0
            cost -= self.costs[i]
            for t in self.days[i]:
                q[t] -= 1
            changed = True
        return changed

    def solution(self, chromosome: list[int] | tuple[int, ...]) -> Solution:
        cost, value, q, feasible = self.evaluate(chromosome)
        if not feasible:
            raise ValueError('Недопустимий розв’язок — перевищено B або L.')
        return Solution(tuple(i + 1 for i, bit in enumerate(chromosome) if bit),
                        value, cost, tuple(q))


def check_solution(problem: Problem, solution: Solution) -> bool:
    """Незалежна перевірка, що результат допустимий і має правильні F, C, q."""
    if len(set(solution.selected)) != len(solution.selected):
        return False
    if any(i < 1 or i > problem.n for i in solution.selected):
        return False
    x = [int(i + 1 in solution.selected) for i in range(problem.n)]
    try:
        calculated = EvaluationContext(problem).solution(x)
    except ValueError:
        return False
    return calculated == Solution(tuple(sorted(solution.selected)), solution.coverage,
                                  solution.cost, solution.daily_posts)


# ---------------------------------------------------------------------------
# ЖАДІБНИЙ АЛГОРИТМ
# ---------------------------------------------------------------------------
def solve_greedy(problem: Problem, detailed: bool = False):
    ctx = EvaluationContext(problem)
    order = sorted(range(problem.n),
                   key=lambda i: (-ctx.ratios[i], -ctx.values[i], ctx.costs[i], i))
    x = [0] * problem.n
    spent = 0
    q = [0] * problem.days_count
    steps = []
    for i in order:
        if spent + ctx.costs[i] > problem.budget:
            if detailed:
                steps.append((i + 1, 'відхилено: бюджет'))
            continue
        if any(q[t] + 1 > problem.daily_limit for t in ctx.days[i]):
            if detailed:
                steps.append((i + 1, 'відхилено: денний ліміт'))
            continue
        x[i] = 1
        spent += ctx.costs[i]
        for t in ctx.days[i]:
            q[t] += 1
        if detailed:
            steps.append((i + 1, 'обрано'))
    answer = ctx.solution(x)
    if not check_solution(problem, answer):
        raise AssertionError('Жадібний алгоритм повернув некоректний результат.')
    return (answer, steps) if detailed else answer


# ---------------------------------------------------------------------------
# ГЕНЕТИЧНИЙ АЛГОРИТМ
# ---------------------------------------------------------------------------
def roulette_indices(fitness: list[int], rng: random.Random) -> tuple[int, int]:
    """Імовірність p_j = F(x_j) / sum(F(x_k)); якщо сума нуль — рівномірно."""
    total = sum(fitness)
    if total == 0:
        return rng.randrange(len(fitness)), rng.randrange(len(fitness))
    cumulative = []
    subtotal = 0
    for f in fitness:
        subtotal += f
        cumulative.append(subtotal)
    return (bisect_right(cumulative, rng.random() * total),
            bisect_right(cumulative, rng.random() * total))


def solve_genetic(problem: Problem, parameters: GAParameters | None = None,
                  seed: int = 42,
                  on_generation: Callable[[int, int, tuple[int, ...], bool], None] | None = None
                  ) -> GAResult:
    """Розв’язання ГА; необов’язковий callback спрацьовує після кожного покоління.

    on_generation(g, best_f, selected, improved):
      g — номер покоління (0 = початкове), best_f — рекордне охоплення,
      selected — номери інфлюенсерів у рекордному розв’язку (від 1),
      improved — чи поліпшився рекорд від попереднього покоління.
    """
    params = parameters or GAParameters()
    validate_ga_parameters(params)
    ctx = EvaluationContext(problem)
    rng = random.Random(seed)
    repairs = 0

    # GenerateInitialPopulation(P, n): випадкові біти, оцінка та repair.
    population: list[list[int]] = []
    for _ in range(params.population):
        x = [rng.randrange(2) for _ in range(problem.n)]
        if ctx.repair(x):
            repairs += 1
        population.append(x)

    best_f = -1
    best_x: list[int] = [0] * problem.n
    best_cost = 0
    last_improvement = 0
    fitness = []
    for x in population:
        c, f, _, feasible = ctx.evaluate(x)
        assert feasible
        fitness.append(f)
        if f > best_f:
            best_f, best_cost, best_x = f, c, x.copy()
    history = [best_f]  # g=0: початкова популяція
    if on_generation is not None:
        on_generation(0, best_f, tuple(i + 1 for i, bit in enumerate(best_x) if bit), True)

    for g in range(1, params.generations + 1):
        previous_best_f = best_f
        new_population: list[list[int]] = []
        while len(new_population) < params.population:
            parent1, parent2 = roulette_indices(fitness, rng)
            x, y = population[parent1].copy(), population[parent2].copy()
            # Одноточкове схрещування. Для n=1 схема коректно пропускається.
            if problem.n > 1 and rng.random() < params.crossover:
                cut = rng.randrange(1, problem.n)
                x, y = x[:cut] + y[cut:], y[:cut] + x[cut:]
            for child in (x, y):
                # Мутація кожного гена з імовірністю pm.
                for j in range(problem.n):
                    if rng.random() < params.mutation:
                        child[j] = 1 - child[j]
                if ctx.repair(child):
                    repairs += 1
                c, f, _, feasible = ctx.evaluate(child)
                assert feasible
                # Оновлюємо X*, F*, C* негайно після появи нащадка.
                if f > best_f:
                    best_f, best_cost, best_x = f, c, child.copy()
                    last_improvement = g
                new_population.append(child)
                if len(new_population) >= params.population:
                    break
        population = new_population
        fitness = [ctx.evaluate(x)[1] for x in population]
        history.append(best_f)
        if on_generation is not None:
            on_generation(g, best_f, tuple(i + 1 for i, bit in enumerate(best_x) if bit),
                          best_f > previous_best_f)

    result = GAResult(ctx.solution(best_x), tuple(history), last_improvement, repairs, seed)
    if result.solution.coverage != best_f or result.solution.cost != best_cost:
        raise AssertionError('Помилка збереження рекордного розв’язку ГА.')
    if not check_solution(problem, result.solution):
        raise AssertionError('ГА повернув недопустимий результат.')
    return result


# ---------------------------------------------------------------------------
# ГЕНЕРАТОР ЗАДАЧ
# ---------------------------------------------------------------------------
def generate_problem(n: int, T: int, beta: float = 0.35, L: int = 2,
                     seed: int = 42, name: str | None = None) -> Problem:
    if n < 1 or T < 1 or L < 1 or not (0 < beta <= 1):
        raise ValueError('n,T,L >= 1 та 0 < beta <= 1.')
    rng = random.Random(seed)
    influencers = []
    for _ in range(n):
        cost = rng.randint(5000, 18000)
        reach = rng.randint(8000, 25000)
        d = rng.randint(1, min(3, T))
        days = tuple(sorted(rng.sample(range(1, T + 1), d)))
        influencers.append(Influencer(cost, reach, days))
    B = max(min(p.cost for p in influencers),
            math.floor(beta * sum(p.cost for p in influencers)))
    problem = Problem(name or f'Згенерована задача (seed={seed})', T, B, L,
                      tuple(influencers))
    validate_problem(problem)
    return problem


def change_constraints(problem: Problem, beta: float, L: int) -> Problem:
    if not (0 < beta <= 1) or L < 1:
        raise ValueError('0 < beta <= 1, L >= 1.')
    costs = [p.cost for p in problem.influencers]
    return replace(problem, budget=max(min(costs), math.floor(beta * sum(costs))),
                   daily_limit=L)


def display_problem(problem: Problem) -> None:
    print(f'\n{problem.name} | n={problem.n}, T={problem.days_count}, '
          f'B={problem.budget}, L={problem.daily_limit}')
    print(' №  | Вартість | Охоплення посту | Дні | Загальне Ri')
    for i, p in enumerate(problem.influencers, 1):
        print(f'{i:>3} | {p.cost:>8} | {p.reach_per_post:>15} | '
              f'{",".join(map(str,p.days)):<12} | {p.reach_per_post*len(p.days)}')


def display_solution(problem: Problem, title: str, solution: Solution,
                     extra: str = '') -> None:
    status = 'ТАК' if check_solution(problem, solution) else 'НІ'
    print(f'\n{title}: X*={list(solution.selected)}, F*={solution.coverage}, '
          f'C*={solution.cost}, використано бюджету='
          f'{solution.cost / problem.budget * 100 if problem.budget else 0:.1f}%, '
          f'допустимий={status}')
    print('Графік q_t: ' + ', '.join(f'день {i}: {v}'
                                     for i, v in enumerate(solution.daily_posts, 1)))
    if extra:
        print(extra)


# ---------------------------------------------------------------------------
# ЕКСПЕРИМЕНТИ; CSV, HTML, відтворюваність
# ---------------------------------------------------------------------------
GROUPS = ((6, 7), (12, 10), (18, 14), (36, 21))
BASE_PARAMS = GAParameters()


def timed_greedy(problem: Problem) -> tuple[Solution, float]:
    begin = perf_counter_ns()
    result = solve_greedy(problem)
    return result, (perf_counter_ns() - begin) / 1_000_000


def timed_genetic(problem: Problem, params: GAParameters, seed: int) -> tuple[GAResult, float]:
    begin = perf_counter_ns()
    result = solve_genetic(problem, params, seed)
    return result, (perf_counter_ns() - begin) / 1_000_000


def sample_problems(n: int, T: int, count: int, root_seed: int) -> list[Problem]:
    return [generate_problem(n, T, seed=root_seed + i, name=f'n={n},T={T},#{i+1}')
            for i in range(count)]


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def avg(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def med(values) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def stdev(values) -> float:
    values = list(values)
    return statistics.pstdev(values) if values else 0.0


def summarize_runs(rows: list[dict], label: str, option: str) -> dict:
    return {'серія': label, 'варіант': option, 'запусків': len(rows),
            'F_середнє': round(avg(r['F_ga'] for r in rows), 3),
            'F_максимум': max(r['F_ga'] for r in rows),
            'F_стандартне_відхилення': round(stdev(r['F_ga'] for r in rows), 3),
            'час_ГА_середній_мс': round(avg(r['ga_ms'] for r in rows), 5),
            'час_ГА_медіана_мс': round(med(r['ga_ms'] for r in rows), 5),
            'перемог_ГА_%': round(100 * avg(r['F_ga'] > r['F_greedy'] for r in rows), 3),
            'рівність_%': round(100 * avg(r['F_ga'] == r['F_greedy'] for r in rows), 3),
            'останнє_покращення_середнє': round(avg(r['last_improvement'] for r in rows), 3),
            'усі_допустимі': all(r['feasible'] for r in rows)}


def one_ga_row(problem: Problem, task_i: int, repeat_i: int, greedy: Solution,
               params: GAParameters, seed: int) -> dict:
    ga, elapsed = timed_genetic(problem, params, seed)
    valid = check_solution(problem, greedy) and check_solution(problem, ga.solution)
    if not valid:
        raise AssertionError('Виявлено недопустимий результат експерименту.')
    return {'task': task_i, 'repeat': repeat_i, 'seed': seed,
            'n': problem.n, 'T': problem.days_count, 'B': problem.budget,
            'L': problem.daily_limit, 'P': params.population, 'G': params.generations,
            'pc': params.crossover, 'pm': params.mutation,
            'F_greedy': greedy.coverage, 'F_ga': ga.solution.coverage,
            'C_ga': ga.solution.cost, 'budget_share': round(ga.solution.cost / problem.budget, 6),
            'last_improvement': ga.last_improvement, 'repairs': ga.repairs,
            'ga_ms': round(elapsed, 6), 'feasible': valid,
            'history': ';'.join(map(str, ga.history))}


def should_show_progress(index: int, total: int) -> bool:
    """Показуємо проміжні відповіді, не друкуючи сотні майже однакових рядків."""
    step = max(1, (total + 3) // 4)
    return index == 1 or index == total or index % step == 0


def print_generated_sample(problem: Problem) -> None:
    """Фактичні значення вхідних параметрів для контрольного прикладу групи."""
    print(f'  Згенеровано приклад: n={problem.n}, T={problem.days_count}, '
          f'B={problem.budget}, L={problem.daily_limit}', flush=True)
    for i, influencer in enumerate(problem.influencers[:3], 1):
        print(f'    №{i}: c={influencer.cost}, r={influencer.reach_per_post}, '
              f'дні={list(influencer.days)}', flush=True)
    if problem.n > 3:
        print(f'    ... ще {problem.n - 3} інфлюенсерів', flush=True)


def print_ga_trace(row: dict) -> None:
    history = [int(x) for x in row['history'].split(';')]
    last = len(history) - 1
    points = sorted({0, last // 4, last // 2, 3 * last // 4, last})
    print('    Зміна рекорду ГА: ' + ' -> '.join(
        f'g={g}: F*={history[g]}' for g in points), flush=True)


def show_live_ga_demo(problem: Problem, params: GAParameters, seed: int) -> None:
    """Показує реальну роботу ГА; демонстрація не враховується у замірах часу."""
    print('\nКОНТРОЛЬНИЙ ЗАПУСК ГА В РЕАЛЬНОМУ ЧАСІ', flush=True)
    print(f'  n={problem.n}, T={problem.days_count}, B={problem.budget}, '
          f'P={params.population}, G={params.generations}, seed={seed}', flush=True)
    print('  Цей демонстраційний запуск НЕ входить до статистики експериментів.',
          flush=True)

    def show_generation(g: int, coverage: int, selected: tuple[int, ...], improved: bool) -> None:
        if g == 0 or g == params.generations or improved or should_show_progress(g, params.generations):
            note = '  <- покращення' if improved and g > 0 else ''
            print(f'  Покоління {g:>3}/{params.generations}: '
                  f'F*={coverage}, X*={list(selected)}{note}', flush=True)

    result = solve_genetic(problem, params, seed, on_generation=show_generation)
    print(f'  Результат демонстрації: F*={result.solution.coverage}, '
          f'допустимий={"ТАК" if check_solution(problem, result.solution) else "НІ"}',
          flush=True)


def print_experiment_summary(summary: dict) -> None:
    """Відображення всіх підсумкових таблиць у консолі, а не тільки шляхів до файлів."""
    labels = {'generations': 'Кількість поколінь G', 'population': 'Розмір популяції P',
              'crossover': 'Імовірність схрещування pc', 'mutation': 'Імовірність мутації pm',
              'time': 'Час роботи', 'quality': 'Якість розв’язків',
              'constraints': 'Бюджет та денний ліміт'}
    print('\n' + '=' * 72)
    print('ПІДСУМКОВІ РЕЗУЛЬТАТИ ВСІХ ЕКСПЕРИМЕНТІВ')
    for key, rows in summary.items():
        print(f'\n{labels.get(key, key)} ({len(rows)} варіантів):')
        if key in ('generations', 'population', 'crossover', 'mutation'):
            print('  Варіант | середнє F ГА | макс. F ГА | перемог ГА % | час ГА, мс')
            for r in rows:
                print(f'  {r["варіант"]:>7} | {r["F_середнє"]:>12.0f} | '
                      f'{r["F_максимум"]:>11} | {r["перемог_ГА_%"]:>11.1f} | '
                      f'{r["час_ГА_середній_мс"]:>10.3f}')
        elif key == 'time':
            print('  n / T    | середній час ЖА, мс | середній час ГА, мс')
            for r in rows:
                print(f'  {r["n"]:>2} / {r["T"]:<2}  | {r["ЖА_середній_мс"]:>19.4f} | '
                      f'{r["ГА_середній_мс"]:>19.4f}')
        elif key == 'quality':
            print('  n / T    | середнє F ЖА | середнє F ГА | перемог ГА %')
            for r in rows:
                print(f'  {r["n"]:>2} / {r["T"]:<2}  | {r["F_ЖА_середнє"]:>12.0f} | '
                      f'{r["F_ГА_середнє"]:>12.0f} | {r["перемог_ГА_%"]:>11.1f}')
        elif key == 'constraints':
            print('   β   / L | середнє F ЖА | середнє F ГА | допустимі')
            for r in rows:
                print(f'  {r["beta"]:>4.2f} / {r["L"]:<2} | {r["F_ЖА_середнє"]:>12.0f} | '
                      f'{r["F_ГА_середнє"]:>12.0f} | '
                      f'{"так" if r["усі_допустимі"] else "НІ"}')
    print('\nПримітка: якість оцінено від найкращого спостереженого результату,')
    print('а не від математично доведеного глобального оптимуму.', flush=True)


def parameter_series(problems: list[Problem], repeats: int, root_seed: int,
                     out: Path, kind: str, choices: list[tuple[str, GAParameters]]) -> list[dict]:
    greedy_solutions = [solve_greedy(task) for task in problems]
    all_raw = []
    summaries = []
    for option, params in choices:
        rows = []
        for i, (problem, greedy) in enumerate(zip(problems, greedy_solutions)):
            task_rows = []
            for j in range(repeats):
                # Парні незалежні потоки для чесного порівняння різних P/G/pc/pm
                seed = root_seed + i * 1009 + j * 37
                row = one_ga_row(problem, i + 1, j + 1, greedy, params, seed)
                row['option'] = option
                rows.append(row)
                task_rows.append(row)
            if should_show_progress(i + 1, len(problems)):
                print(f'    [{i + 1}/{len(problems)}] {kind}={option} | '
                      f'F_ЖА={greedy.coverage}; F_ГА≈{avg(r["F_ga"] for r in task_rows):.0f}; '
                      f'найкраще F_ГА={max(r["F_ga"] for r in task_rows)}; '
                      f'seed={task_rows[0]["seed"]}; B={problem.budget}; '
                      f'останнє покращення g={task_rows[0]["last_improvement"]}', flush=True)
                if i == 0:
                    print_ga_trace(task_rows[0])
        summaries.append(summarize_runs(rows, kind, option))
        all_raw.extend(rows)
        print(f'  {kind}: {option} | F≈{summaries[-1]["F_середнє"]:.1f}', flush=True)
    csv_write(out / f'{kind}_raw.csv', all_raw)
    csv_write(out / f'{kind}_summary.csv', summaries)
    return summaries


def run_experiments(mode: str, tasks: int | None, repeats: int | None,
                    root_seed: int, out: Path, only: str = 'all') -> dict:
    if mode not in ('quick', 'full'):
        raise ValueError('mode = quick або full')
    count = tasks if tasks is not None else (100 if mode == 'full' else 4)
    reps = repeats if repeats is not None else (5 if mode == 'full' else 2)
    if count < 1 or reps < 1:
        raise ValueError('tasks і repeats мають бути додатними.')
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[dict]] = {}
    supported = ('all', 'parameters', 'generations', 'population',
                 'crossover', 'mutation', 'time', 'quality', 'constraints')
    if only not in supported:
        raise ValueError(f'Невідома серія експериментів: {only}')
    params_tasks = (sample_problems(18, 14, count, root_seed + 100000)
                    if only in ('all', 'parameters', 'generations',
                                'population', 'crossover', 'mutation', 'constraints')
                    else [])

    # Один контрольний запуск із виводом КОЖНОГО зафіксованого етапу
    # в момент його появи. Не змінює статистичних запусків і їх часу.
    if params_tasks:
        print_generated_sample(params_tasks[0])
        show_live_ga_demo(params_tasks[0], BASE_PARAMS, root_seed + 900000)

    if only in ('all', 'parameters', 'generations', 'population', 'crossover', 'mutation'):
        print('\nПАРАМЕТРИ ГЕНЕТИЧНОГО АЛГОРИТМУ', flush=True)
        if params_tasks:
            print_generated_sample(params_tasks[0])
        tests = {
            'generations': [(str(g), replace(BASE_PARAMS, generations=g))
                            for g in (25, 50, 100, 200)],
            'population': [(str(p), replace(BASE_PARAMS, population=p))
                           for p in (20, 40, 80)],
            'crossover': [(str(p), replace(BASE_PARAMS, crossover=p))
                          for p in (0.60, 0.80, 0.90)],
            'mutation': [(str(p), replace(BASE_PARAMS, mutation=p))
                         for p in (0.01, 0.05, 0.10)]
        }
        for kind, options in tests.items():
            if only not in ('all', 'parameters', kind):
                continue
            summary[kind] = parameter_series(params_tasks, reps, root_seed + 200000,
                                             out, kind, options)

    if only in ('all', 'time', 'quality'):
        titles = {'all': 'Час та якість',
                  'time': 'Порівняння часу',
                  'quality': 'Порівняння якості'}
        print(f'\n[{titles[only]}]', flush=True)
        time_raw = []
        time_summary = []
        quality_raw = []
        quality_summary = []
        for group_i, (n, T) in enumerate(GROUPS):
            problems = sample_problems(n, T, count, root_seed + 1000 * (group_i + 1))
            print_generated_sample(problems[0])
            if group_i == 0 and not params_tasks:
                show_live_ga_demo(problems[0], BASE_PARAMS, root_seed + 900000)
            rows = []
            for i, problem in enumerate(problems):
                greedy, greedy_ms = timed_greedy(problem)
                ga, ga_ms = timed_genetic(problem, BASE_PARAMS, root_seed + 500000 + 101 * group_i + i)
                if not (check_solution(problem, greedy) and check_solution(problem, ga.solution)):
                    raise AssertionError('Некоректний розв’язок у порівнянні.')
                reference = max(greedy.coverage, ga.solution.coverage)
                rows.append({'task': i + 1, 'n': n, 'T': T,
                             'F_greedy': greedy.coverage, 'F_ga': ga.solution.coverage,
                             'greedy_ms': round(greedy_ms, 6), 'ga_ms': round(ga_ms, 6),
                             'gap_greedy_pct': round(100 * (reference - greedy.coverage) / reference, 6)
                             if reference else 0,
                             'gap_ga_pct': round(100 * (reference - ga.solution.coverage) / reference, 6)
                             if reference else 0,
                             'ga_win': ga.solution.coverage > greedy.coverage,
                             'reference_best_observed': reference,
                             'feasible': True})
                if should_show_progress(i + 1, count):
                    print(f'    [{i + 1}/{count}] n={n},T={T}: F_ЖА={greedy.coverage}, '
                          f'F_ГА={ga.solution.coverage}, '
                          f'ЖА={greedy_ms:.3f} мс, ГА={ga_ms:.3f} мс, '
                          f'B={problem.budget}', flush=True)
            time_raw.extend(rows)
            quality_raw.extend(rows)
            time_summary.append({'n': n, 'T': T, 'задач': count,
                                 'ЖА_середній_мс': round(avg(x['greedy_ms'] for x in rows), 6),
                                 'ЖА_медіана_мс': round(med(x['greedy_ms'] for x in rows), 6),
                                 'ГА_середній_мс': round(avg(x['ga_ms'] for x in rows), 6),
                                 'ГА_медіана_мс': round(med(x['ga_ms'] for x in rows), 6),
                                 'відношення_часів': round(avg(x['ga_ms'] for x in rows) /
                                                           max(avg(x['greedy_ms'] for x in rows), 1e-12), 3)})
            quality_summary.append({'n': n, 'T': T, 'задач': count,
                                    'F_ЖА_середнє': round(avg(x['F_greedy'] for x in rows), 3),
                                    'F_ГА_середнє': round(avg(x['F_ga'] for x in rows), 3),
                                    'gap_ЖА_%': round(avg(x['gap_greedy_pct'] for x in rows), 4),
                                    'gap_ГА_%': round(avg(x['gap_ga_pct'] for x in rows), 4),
                                    'перемог_ГА_%': round(100 * avg(x['ga_win'] for x in rows), 3)})
            if only == 'quality':
                print(f'  n={n}, T={T} | середнє F ЖА≈{quality_summary[-1]["F_ЖА_середнє"]:.0f}; '
                      f'ГА≈{quality_summary[-1]["F_ГА_середнє"]:.0f}', flush=True)
            else:
                print(f'  n={n}, T={T} | ЖА {time_summary[-1]["ЖА_середній_мс"]:.4f} мс; '
                      f'ГА {time_summary[-1]["ГА_середній_мс"]:.4f} мс', flush=True)
        if only in ('all', 'time'):
            csv_write(out / 'time_raw.csv', time_raw)
            csv_write(out / 'time_summary.csv', time_summary)
            summary['time'] = time_summary
        if only in ('all', 'quality'):
            csv_write(out / 'quality_raw.csv', quality_raw)
            csv_write(out / 'quality_summary.csv', quality_summary)
            summary['quality'] = quality_summary

    if only in ('all', 'constraints'):
        print('\nВПЛИВ БЮДЖЕТУ β ТА ДЕННОГО ЛІМІТУ L', flush=True)
        rows_all = []
        summaries = []
        for beta in (0.20, 0.35, 0.50, 0.65):
            for L in (1, 2, 3, 4):
                rows = []
                for i, base in enumerate(params_tasks):
                    problem = change_constraints(base, beta, L)
                    gr, gr_ms = timed_greedy(problem)
                    ga, ga_ms = timed_genetic(problem, BASE_PARAMS, root_seed + 800000 + i)
                    rows.append({'beta': beta, 'L': L, 'task': i + 1,
                                 'n': 18, 'T': 14, 'B': problem.budget,
                                 'F_greedy': gr.coverage, 'F_ga': ga.solution.coverage,
                                 'C_greedy': gr.cost, 'C_ga': ga.solution.cost,
                                 'budget_share_greedy': round(gr.cost / problem.budget, 6),
                                 'budget_share_ga': round(ga.solution.cost / problem.budget, 6),
                                 'greedy_ms': round(gr_ms, 6), 'ga_ms': round(ga_ms, 6),
                                 'feasible': check_solution(problem, gr) and
                                             check_solution(problem, ga.solution)})
                    if should_show_progress(i + 1, count):
                        print(f'    [{i + 1}/{count}] β={beta},L={L},B={problem.budget}: '
                              f'F_ЖА={gr.coverage}, F_ГА={ga.solution.coverage}, '
                              f'C_ЖА={gr.cost}, C_ГА={ga.solution.cost}', flush=True)
                if not all(row['feasible'] for row in rows):
                    raise AssertionError('Недопустимий результат експерименту B/L.')
                rows_all.extend(rows)
                summaries.append({'beta': beta, 'L': L, 'задач': count,
                                  'F_ЖА_середнє': round(avg(x['F_greedy'] for x in rows), 3),
                                  'F_ГА_середнє': round(avg(x['F_ga'] for x in rows), 3),
                                  'C_ЖА_до_B': round(avg(x['budget_share_greedy'] for x in rows), 5),
                                  'C_ГА_до_B': round(avg(x['budget_share_ga'] for x in rows), 5),
                                  'ЖА_середній_мс': round(avg(x['greedy_ms'] for x in rows), 5),
                                  'ГА_середній_мс': round(avg(x['ga_ms'] for x in rows), 5),
                                  'усі_допустимі': True})
                print(f'  β={beta}, L={L} | F ЖА≈{summaries[-1]["F_ЖА_середнє"]:.0f}, '
                      f'F ГА≈{summaries[-1]["F_ГА_середнє"]:.0f}', flush=True)
        csv_write(out / 'constraints_raw.csv', rows_all)
        csv_write(out / 'constraints_summary.csv', summaries)
        summary['constraints'] = summaries

    metadata = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'python': sys.version.split()[0], 'platform': sys.platform,
        'mode': mode, 'tasks_per_group': count, 'repeats_per_parameter_case': reps,
        'seed': root_seed, 'series': only, 'GA_base_parameters': asdict(BASE_PARAMS),
        'groups': [{'n': n, 'T': T} for n, T in GROUPS],
        'reference_rule': 'У quality: max(F_ЖА, F_ГА) для цієї задачі, не глобальний оптимум.',
        'time_measurement': 'perf_counter_ns, мс; без генерації задач, файлів та виводу.'
    }
    (out / 'metadata.txt').write_text(
        '\n'.join(f'{key}: {value}' for key, value in metadata.items()) + '\n',
        encoding='utf-8')
    make_report_html(out, summary, metadata)
    print_experiment_summary(summary)
    print(f'\nЗавершено. CSV і HTML-звіт: {out.resolve()}', flush=True)
    return summary


def make_report_html(out: Path, data: dict, metadata: dict) -> None:
    sections = []
    titles = {'generations': 'Кількість поколінь G',
              'population': 'Розмір популяції P',
              'crossover': 'Імовірність схрещування pc',
              'mutation': 'Імовірність мутації pm',
              'time': 'Витрати часу',
              'quality': 'Якість розв’язків',
              'constraints': 'Вплив β і L'}
    for key, rows in data.items():
        if not rows:
            continue
        columns = list(rows[0])
        th = ''.join(f'<th>{html.escape(str(c))}</th>' for c in columns)
        body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(row.get(c,"")))}</td>'
                                        for c in columns) + '</tr>' for row in rows)
        sections.append(f'<section><h2>{html.escape(titles.get(key,key))}</h2>'
                        f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
                        f'<tbody>{body}</tbody></table></div></section>')
    note = ('Показник відхилення δ у таблиці «Якість» вимірюється від НАЙКРАЩОГО '
            'СПОСТЕРЕЖЕНОГО результату двох алгоритмів для конкретної задачі, '
            'а не від доведеного глобального оптимуму.')
    webpage = f'''<!doctype html><html lang="uk"><head><meta charset="utf-8">
<title>Експерименти: інформаційна кампанія</title><style>
body{{font:15px/1.5 system-ui,Arial,sans-serif;background:#f4f6fa;color:#182339;max-width:1200px;margin:35px auto;padding:0 20px}}
h1{{font-size:29px}}h2{{font-size:20px}}section{{background:white;padding:20px;margin:22px 0;border-radius:12px;box-shadow:0 2px 10px #17233910}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:9px 12px;border-bottom:1px solid #e7eaf1;text-align:left;white-space:nowrap}}
th{{background:#eaf0ff;position:sticky;top:0}}tr:nth-child(even){{background:#fafbfe}}.small{{color:#536178}}code{{background:#e9ecf4;padding:3px 6px}}
</style></head><body><h1>Планування інформаційної кампанії</h1>
<p class="small">Звіт експериментів · {html.escape(metadata['generated_at'])} · режим {html.escape(metadata['mode'])} · задач у групі: {metadata['tasks_per_group']} · повторів: {metadata['repeats_per_parameter_case']}</p>
<p><strong>Важливо:</strong> {html.escape(note)}</p>{''.join(sections)}
<p class="small">Повні числові дані доступні в CSV-файлах поруч зі звітом. Час залежить від комп’ютера, запуску та навантаження системи.</p></body></html>'''
    (out / 'report.html').write_text(webpage, encoding='utf-8')


# ---------------------------------------------------------------------------
# КОНСОЛЬНА ПРОГРАМА: дані -> порівняння, експерименти — окремими пунктами
# ---------------------------------------------------------------------------
def ask_int(label: str, default: int | None = None, minimum: int | None = None,
            maximum: int | None = None) -> int:
    """Повторюємо запит, доки користувач не введе ціле число у допустимому діапазоні."""
    while True:
        s = input(f'{label}' + (f' [{default}]' if default is not None else '') + ': ').strip()
        if not s and default is None:
            print('  Введіть ціле число.')
            continue
        try:
            value = int(s) if s else default
        except ValueError:
            print('  Потрібне ціле число, наприклад 6 або 30000.')
            continue
        if minimum is not None and value < minimum:
            print(f'  Значення має бути не менше {minimum}.')
            continue
        if maximum is not None and value > maximum:
            print(f'  Значення має бути не більше {maximum}.')
            continue
        return value


def ask_float(label: str, default: float, minimum: float | None = None,
              maximum: float | None = None, exclusive_min: bool = False) -> float:
    while True:
        s = input(f'{label} [{default}]: ').strip().replace(',', '.')
        try:
            value = float(s) if s else default
        except ValueError:
            print('  Потрібне число, наприклад 0.35.')
            continue
        if not math.isfinite(value):
            print('  Число має бути скінченним.')
            continue
        if (minimum is not None and (value <= minimum if exclusive_min else value < minimum)) \
           or (maximum is not None and value > maximum):
            lower = f'>{minimum}' if exclusive_min else f'>={minimum}'
            print(f'  Допустимий діапазон: {lower}, <={maximum}.')
            continue
        return value


def ask_days(T: int) -> tuple[int, ...]:
    while True:
        line = input(f'  Дні публікацій (від 1 до {T}; напр. 1,3,5): ').strip()
        try:
            days = tuple(int(x.strip()) for x in line.split(','))
        except ValueError:
            print('  Уведіть номери днів через кому, наприклад 1,3,5.')
            continue
        if not days or len(set(days)) != len(days) or any(t < 1 or t > T for t in days):
            print(f'  Потрібні унікальні номери днів від 1 до {T}.')
            continue
        return tuple(sorted(days))


def enter_problem() -> Problem:
    print('\n--- Введення даних вручну ---')
    print('Обмеження: Один інфлюенсер публікує не більше одного посту на день.')
    n = ask_int('Кількість доступних інфлюенсерів n (від 1 до 10000)', 6, 1, 10000)
    T = ask_int('Тривалість кампанії T, днів (від 1 до 366)', 7, 1, 366)
    B = ask_int('Загальний бюджет B, грн (>= 0)', 30000, 0)
    L = ask_int('Максимум публікацій за день L (> 0)', 2, 1)
    print('Для кожного учасника задайте вартість cᵢ, охоплення одного посту rᵢ та дні.')
    people = []
    for i in range(1, n + 1):
        print(f'\nІнфлюенсер №{i} із {n}')
        cost = ask_int('  Вартість співпраці, грн (напр. 8000)', minimum=1)
        reach = ask_int('  Охоплення одного посту, осіб (напр. 12000)', minimum=0)
        days = ask_days(T)
        people.append(Influencer(cost, reach, days))
    # Внутрішній підпис для display_problem; запиту назви кампанії немає.
    problem = Problem('Власна задача', T, B, L, tuple(people))
    validate_problem(problem)
    return problem


def random_problem_interactive() -> Problem:
    print('\n--- Генерація випадкової задачі ---')
    print('Стандартні значення: β=0.35, L=2. Для випадкового прикладу залиште seed порожнім.')
    n = ask_int('Кількість інфлюенсерів n', 6, 1, 10000)
    T = ask_int('Тривалість T, днів', 7, 1, 366)
    beta = ask_float('Частка бюджету β (0 < β <= 1)', 0.35, 0, 1, exclusive_min=True)
    limit = ask_int('Денний ліміт L', 2, 1)
    while True:
        value = input('Seed (Enter = новий випадковий, число = відтворюваний): ').strip()
        try:
            seed = int(value) if value else secrets.randbits(32)
            break
        except ValueError:
            print('  Seed має бути цілим числом або порожнім.')
    problem = generate_problem(n, T, beta, limit, seed)
    print(f'\nЗгенеровано задачу. Seed = {seed}; запишіть його для відтворення даних.')
    return problem


def run_comparison(problem: Problem, parameters: GAParameters = BASE_PARAMS,
                   seed: int = 42) -> None:
    """Єдина доступна дія розв’язання: порівняння ЖА та ГА на одних даних."""
    print('\n' + '-' * 67)
    print('ПОРІВНЯННЯ ЖАДІБНОГО ТА ГЕНЕТИЧНОГО АЛГОРИТМІВ')
    print(f'Налаштування ГА: P={parameters.population}, G={parameters.generations}, '
          f'pc={parameters.crossover:.2f}, pm={parameters.mutation:.2f}, seed={seed}')
    greedy, greedy_ms = timed_greedy(problem)
    ga, ga_ms = timed_genetic(problem, parameters, seed)
    display_solution(problem, 'Жадібний алгоритм', greedy)
    display_solution(problem, 'Генетичний алгоритм', ga.solution,
                     f'Покоління останнього покращення: {ga.last_improvement}; '
                     f'виконано repair: {ga.repairs}; seed: {ga.seed}.')
    delta = ga.solution.coverage - greedy.coverage
    print('\nПІДСУМОК ПОРІВНЯННЯ')
    if delta > 0:
        print(f'  Генетичний дав на {delta} користувачів більше охоплення.')
    elif delta < 0:
        print(f'  Жадібний дав на {-delta} користувачів більше охоплення.')
    else:
        print('  Обидва алгоритми дали однакове сумарне охоплення.')
    print(f'  Час: жадібний ≈ {greedy_ms:.4f} мс; генетичний ≈ {ga_ms:.4f} мс.')

def interactive_experiment() -> None:
    print('\nЕКСПЕРИМЕНТИ — ПОВНИЙ НАБІР ПОРІВНЯНЬ')
    print('Виконаються 7 груп досліджень: G, P, pc, pm, час, якість, бюджет/ліміт.')
    print('Пробний режим: 4 задачі на групу, 2 повтори ГА для налаштувань.')
    print('Повний режим: 100 задач на групу, 5 повторів ГА для налаштувань.')
    while True:
        answer = input('Режим [Enter=пробний / 2=повний]: ').strip()
        if answer in ('', '1', '2'):
            break
        print('  Введіть 2 для повного режиму або Enter для пробного.')
    mode = 'full' if answer == '2' else 'quick'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dest = Path('results') / f'eksperymenty_{mode}_{stamp}'
    print(f'\nПочинаємо.')
    print('обох алгоритмів та динаміка рекорду ГА для контрольних запусків.', flush=True)
    summary = run_experiments(mode, None, None, 2026, dest, only='all')
    print(f'\nЗбережено таблиці всіх {len(summary)} груп досліджень у {dest.resolve()}')


def pause_before_menu() -> None:
    input('\nНатисніть Enter, коли завершите перегляд результатів...')


def print_menu() -> None:
    print('\n' + '=' * 67)
    print('ПЛАНУВАННЯ ІНФОРМАЦІЙНОЇ КАМПАНІЇ — ПОРІВНЯННЯ АЛГОРИТМІВ')
    print('1. Ввести власні дані → порівняти ЖА і ГА')
    print('2. Згенерувати випадкові дані → порівняти ЖА і ГА')
    print('3. ЕКСПЕРИМЕНТИ (усі дослідження та результати)')
    print('0. Вихід')


def interactive() -> None:
    while True:
        print_menu()
        try:
            choice = input('Ваш вибір: ').strip()
        except (KeyboardInterrupt, EOFError):
            print('\nРоботу завершено.')
            break
        if choice == '0':
            print('До побачення!')
            break
        try:
            if choice in ('1', '2'):
                problem = enter_problem() if choice == '1' else random_problem_interactive()
                print('\nВХІДНІ ДАНІ ЗАДАЧІ:')
                display_problem(problem)
                run_comparison(problem)
            elif choice == '3':
                interactive_experiment()
            else:
                print('Невідомий пункт. Оберіть 0, 1, 2 або 3.')
        except (ValueError, KeyError, OSError, AssertionError) as exc:
            print('Помилка:', exc)
        except (KeyboardInterrupt, EOFError):
            print('\nОперацію перервано.')
        try:
            pause_before_menu()
        except (KeyboardInterrupt, EOFError):
            print('\nРоботу завершено.')
            break


def cli() -> None:
    parser = argparse.ArgumentParser(description='Порівняння ЖА та ГА; експерименти')
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('menu', help='Інтерактивне меню')
    experiments = sub.add_parser('experiments', help='Всі дослідження одним запуском')
    experiments.add_argument('--mode', choices=['quick', 'full'], default='quick')
    experiments.add_argument('--tasks', type=int, default=None,
                             help='Кількість задач на групу (для перевірок)')
    experiments.add_argument('--repeats', type=int, default=None,
                             help='Повтори ГА у дослідженні параметрів')
    experiments.add_argument('--seed', type=int, default=2026)
    experiments.add_argument('--out', default='results/experiments')
    args = parser.parse_args()
    if args.command is None or args.command == 'menu':
        interactive()
    elif args.command == 'experiments':
        run_experiments(args.mode, args.tasks, args.repeats, args.seed,
                        Path(args.out), only='all')


if __name__ == '__main__':
    try:
        cli()
    except (ValueError, OSError) as error:
        print('Помилка:', error, file=sys.stderr)
        sys.exit(2)
