"""Автоматичні тести; запуск: python -m unittest -v."""
import io
import itertools
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

from main import (EvaluationContext, GAParameters, Influencer, Problem,
                  built_in_examples, change_constraints, check_solution,
                  generate_problem, roulette_indices,
                  solve_genetic, solve_greedy, validate_problem, interactive,
                  run_experiments)
import random


class TestCourseProject(unittest.TestCase):
    def test_greedy_example_241(self):
        p = built_in_examples()['4']
        x = solve_greedy(p)
        self.assertEqual(x.selected, (2, 4))
        self.assertEqual((x.coverage, x.cost), (89000, 20000))
        self.assertEqual(x.daily_posts, (0, 1, 1, 2, 1))
        self.assertTrue(check_solution(p, x))

    def test_ga_example_242(self):
        p = built_in_examples()['5']
        result = solve_genetic(p, GAParameters(40, 100, 0.80, 0.05), seed=42)
        self.assertEqual(result.solution.selected, (1, 4))
        self.assertEqual((result.solution.coverage, result.solution.cost), (87000, 20000))
        self.assertEqual(result.solution.daily_posts, (1, 0, 2, 0, 1, 1, 1))
        self.assertEqual(len(result.history), 101)
        self.assertTrue(check_solution(p, result.solution))

    def test_all_examples_and_reproducibility(self):
        for key, p in built_in_examples().items():
            with self.subTest(example=key):
                g = solve_greedy(p)
                ga1 = solve_genetic(p, GAParameters(12, 25, 0.8, 0.05), 123)
                ga2 = solve_genetic(p, GAParameters(12, 25, 0.8, 0.05), 123)
                self.assertTrue(check_solution(p, g))
                self.assertTrue(check_solution(p, ga1.solution))
                self.assertEqual(ga1, ga2)
                self.assertTrue(all(x <= y for x, y in zip(ga1.history, ga1.history[1:])))

    def test_budget_repair(self):
        p = Problem('budget', 2, 10, 2, (
            Influencer(10, 30, (1,)),
            Influencer(7, 1, (2,)),
        ))
        ctx = EvaluationContext(p)
        x = [1, 1]
        self.assertTrue(ctx.repair(x))
        self.assertEqual(x, [1, 0])  # Видалення найменш ефективного
        self.assertTrue(ctx.evaluate(x)[3])

    def test_overload_repair_only_conflicting(self):
        p = Problem('calendar', 2, 100, 1, (
            Influencer(10, 10, (1,)),
            Influencer(10, 20, (1,)),
            Influencer(10, 1, (2,)),  # Найгірший k, але не причина перевантаження
        ))
        x = [1, 1, 1]
        ctx = EvaluationContext(p)
        ctx.repair(x)
        self.assertEqual(x, [0, 1, 1])

    def test_repair_tie_breaker(self):
        p = Problem('tie', 2, 10, 2, (
            Influencer(10, 10, (1,)),
            Influencer(10, 10, (2,)),
        ))
        x = [1, 1]
        EvaluationContext(p).repair(x)
        self.assertEqual(x, [1, 0])  # за рівності k і R більший номер видаляється

    def test_greedy_tie(self):
        p = Problem('tie', 2, 10, 1, (
            Influencer(10, 20, (1,)),
            Influencer(5, 10, (1,)),
        ))
        self.assertEqual(solve_greedy(p).selected, (1,))  # рівні k: більший Ri обирається першим

    def test_generation_and_constraints(self):
        p = generate_problem(12, 10, 0.35, 2, 999)
        self.assertEqual(p, generate_problem(12, 10, 0.35, 2, 999))
        self.assertGreaterEqual(p.budget, min(p2.cost for p2 in p.influencers))
        self.assertEqual(p.n, 12)
        tighter = change_constraints(p, 0.2, 1)
        self.assertEqual(tighter.daily_limit, 1)
        self.assertEqual(tighter.influencers, p.influencers)

    def test_roulette_zero(self):
        rng = random.Random(1)
        for _ in range(50):
            a, b = roulette_indices([0, 0, 0], rng)
            self.assertIn(a, range(3))
            self.assertIn(b, range(3))

    def test_roulette_nonzero(self):
        rng = random.Random(1)
        for _ in range(100):
            a, b = roulette_indices([0, 100, 0], rng)
            self.assertEqual((a, b), (1, 1))

    def test_ga_n1_and_empty_solution(self):
        p = Problem('one', 1, 0, 1, (Influencer(10, 20, (1,)),))
        res = solve_genetic(p, GAParameters(4, 10, 0.80, 0.05), seed=3)
        self.assertEqual(res.solution.selected, ())
        self.assertTrue(check_solution(p, res.solution))

    def test_random_feasibility(self):
        for seed in range(16):
            p = generate_problem(6 + seed, 4 + seed, beta=0.2, L=1, seed=seed)
            self.assertTrue(check_solution(p, solve_greedy(p)))
            ga = solve_genetic(p, GAParameters(10, 10, 0.8, 0.05), seed=seed)
            self.assertTrue(check_solution(p, ga.solution))


class TestAdditionalVerification(unittest.TestCase):
    """Не тільки наявність F*, а й математична правильність та реальний UI."""

    def test_compare_to_exhaustive_reference_small_random_tasks(self):
        """ЖА/ГА повертають допустимі розв'язки, які не перевищують точний максимум.

        Повний перебір тут використовується ЛИШЕ як контроль у тестах,
        не є третім алгоритмом програмного продукту / дослідження.
        """
        for seed in range(25):
            problem = generate_problem(n=4 + seed % 7, T=3 + seed % 8,
                                       beta=(0.2, 0.35, 0.65)[seed % 3],
                                       L=(1, 2, 3)[seed % 3], seed=seed)
            ctx = EvaluationContext(problem)
            exact = max(ctx.evaluate(x)[1] for x in itertools.product((0, 1), repeat=problem.n)
                        if ctx.evaluate(x)[3])
            greedy = solve_greedy(problem)
            genetic = solve_genetic(problem, GAParameters(15, 35, 0.8, 0.05), seed=seed)
            self.assertTrue(check_solution(problem, greedy))
            self.assertTrue(check_solution(problem, genetic.solution))
            self.assertLessEqual(greedy.coverage, exact)
            self.assertLessEqual(genetic.solution.coverage, exact)
            self.assertEqual(len(genetic.history), 36)
            self.assertEqual(tuple(sorted(genetic.history)), genetic.history)

    def test_roulette_empirical_proportions(self):
        rng = random.Random(2026)
        observed = [0, 0, 0]
        for _ in range(5000):
            a, b = roulette_indices([20, 30, 50], rng)
            observed[a] += 1
            observed[b] += 1
        total = sum(observed)
        for seen, target in zip(observed, (0.2, 0.3, 0.5)):
            self.assertAlmostEqual(seen / total, target, delta=0.02)

    def test_main_menu_random_auto_compare_and_pause(self):
        # Введення 6 n / 7 T / beta / L за замовчуванням, seed=123.
        responses = ['2', '', '', '', '', '123', '', '0']
        out = io.StringIO()
        answers = iter(responses)
        def typed(prompt):
            print(prompt, end='')
            return next(answers)
        with patch('builtins.input', side_effect=typed), redirect_stdout(out):
            interactive()
        text = out.getvalue()
        self.assertIn('Згенеровано задачу. Seed = 123', text)
        self.assertIn('ВХІДНІ ДАНІ ЗАДАЧІ', text)
        self.assertIn('ПОРІВНЯННЯ ЖАДІБНОГО ТА ГЕНЕТИЧНОГО', text)
        self.assertIn('допустимий=ТАК', text)
        self.assertEqual(text.count('ПЛАНУВАННЯ ІНФОРМАЦІЙНОЇ КАМПАНІЇ — ПОРІВНЯННЯ АЛГОРИТМІВ'), 2)
        self.assertLess(text.index('ПІДСУМОК ПОРІВНЯННЯ'),
                        text.index('Натисніть Enter, коли завершите перегляд результатів'))
        self.assertLess(text.index('Натисніть Enter, коли завершите перегляд результатів'),
                        text.rindex('ПЛАНУВАННЯ ІНФОРМАЦІЙНОЇ КАМПАНІЇ — ПОРІВНЯННЯ АЛГОРИТМІВ'))
        self.assertNotIn('Обрати готову задачу', text)
        self.assertNotIn('Показати вхідні дані', text)

    def test_main_menu_manual_auto_compare_and_pause(self):
        # Після вилучення поля назви одразу запитується n, а не назва кампанії.
        responses = ['1', '2', '3', '100', '1',
                     '50', '20', '1,2', '80', '100', '2,3', '', '0']
        out = io.StringIO()
        answers = iter(responses)
        def typed(prompt):
            print(prompt, end='')
            return next(answers)
        with patch('builtins.input', side_effect=typed), redirect_stdout(out):
            interactive()
        text = out.getvalue()
        self.assertIn('n=2, T=3, B=100, L=1', text)
        self.assertIn('Графік q_t:', text)
        self.assertIn('допустимий=ТАК', text)

    def test_genetic_generation_callback_live_and_reproducible(self):
        problem = built_in_examples()['4']
        events = []

        def on_generation(g, coverage, selected, improved):
            events.append((g, coverage, selected, improved))

        params = GAParameters(12, 20, 0.8, 0.05)
        with_progress = solve_genetic(problem, params, seed=123, on_generation=on_generation)
        without_progress = solve_genetic(problem, params, seed=123)
        self.assertEqual(with_progress, without_progress)
        self.assertEqual(len(events), params.generations + 1)
        self.assertEqual([event[0] for event in events], list(range(params.generations + 1)))
        self.assertEqual(tuple(event[1] for event in events), with_progress.history)
        self.assertTrue(events[0][3])
        for before, after in zip(events, events[1:]):
            self.assertEqual(after[3], after[1] > before[1])
            self.assertEqual(tuple(sorted(after[2])), after[2])

    def test_all_experiments_and_live_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            out = io.StringIO()
            with redirect_stdout(out):
                result = run_experiments('quick', tasks=2, repeats=1,
                                         root_seed=10, out=dest, only='all')
            self.assertEqual(set(result), {
                'generations','population','crossover','mutation',
                'time','quality','constraints'})
            self.assertEqual(len(list(dest.glob('*_raw.csv'))), 7)
            self.assertEqual(len(list(dest.glob('*_summary.csv'))), 7)
            self.assertTrue((dest / 'metadata.txt').is_file())
            self.assertTrue((dest / 'report.html').is_file())
            self.assertFalse(list(dest.rglob('*.json')))
            output = out.getvalue()
            self.assertIn('Згенеровано приклад:', output)
            self.assertIn('КОНТРОЛЬНИЙ ЗАПУСК ГА В РЕАЛЬНОМУ ЧАСІ', output)
            self.assertIn('Покоління   0/100', output)
            self.assertIn('Покоління 100/100', output)
            self.assertIn('Зміна рекорду ГА:', output)
            self.assertIn('F_ЖА=', output)
            self.assertIn('F_ГА=', output)
            self.assertIn('ПІДСУМКОВІ РЕЗУЛЬТАТИ ВСІХ ЕКСПЕРИМЕНТІВ', output)
            self.assertNotIn('розділ 3', output.lower())
            self.assertNotIn('розділом 3',
                             (dest / 'report.html').read_text(encoding='utf-8').lower())

    def test_menu_only_three_actions(self):
        out = io.StringIO()
        with patch('builtins.input', side_effect=['0']), redirect_stdout(out):
            interactive()
        text = out.getvalue()
        self.assertIn('1. Ввести власні дані', text)
        self.assertIn('2. Згенерувати випадкові дані', text)
        self.assertIn('3. ЕКСПЕРИМЕНТИ', text)
        self.assertNotIn('JSON', text)
        self.assertNotIn('Розв’язати лише', text)
        self.assertNotIn('розділ 3', text.lower())
        self.assertNotIn('4. ', text)

    def test_experiments_menu_pauses_after_results(self):
        with tempfile.TemporaryDirectory() as folder:
            out = io.StringIO()
            answers = iter(['3', '', '', '0'])
            def typed(prompt):
                print(prompt, end='')
                return next(answers)
            with patch('main.run_experiments', return_value={'time': [{}]}), \
                 patch('main.datetime') as dt, \
                 patch('builtins.input', side_effect=typed), \
                 redirect_stdout(out):
                dt.now.return_value.strftime.return_value = '20261029_170000'
                interactive()
            output = out.getvalue()
            self.assertIn('ЕКСПЕРИМЕНТИ — ПОВНИЙ НАБІР ПОРІВНЯНЬ', output)
            self.assertIn('Натисніть Enter, коли завершите перегляд результатів', output)
            self.assertEqual(output.count('1. Ввести власні дані'), 2)

    def test_progress_helper(self):
        from main import should_show_progress
        self.assertTrue(should_show_progress(1, 100))
        self.assertFalse(should_show_progress(2, 100))
        self.assertTrue(should_show_progress(25, 100))
        self.assertTrue(should_show_progress(100, 100))


if __name__ == '__main__':
    unittest.main()
