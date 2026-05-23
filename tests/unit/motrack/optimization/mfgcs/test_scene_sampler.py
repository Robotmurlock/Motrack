"""
Tests for ``motrack.optimization.mfgcs.scene_sampler``.
"""
import re
import unittest

from motrack.config_parser import FactorySpec
from motrack.optimization.mfgcs.scene_sampler import (
    RandomSceneSampler,
    SceneSampler,
    StratifiedSceneSampler,
    scene_sampler_factory,
)


class RandomSceneSamplerTest(unittest.TestCase):
    """RandomSceneSampler determinism + bounds."""

    def test_returns_subset_of_requested_size(self) -> None:
        sampler = RandomSceneSampler(n=3, seed=0)
        scenes = ['a', 'b', 'c', 'd', 'e']
        sample = sampler.sample(scenes)
        self.assertEqual(len(sample), 3)
        self.assertTrue(set(sample).issubset(set(scenes)))
        self.assertEqual(len(set(sample)), 3, 'sample must be unique')

    def test_clips_n_to_dataset_size(self) -> None:
        sampler = RandomSceneSampler(n=10, seed=0)
        sample = sampler.sample(['a', 'b'])
        self.assertEqual(set(sample), {'a', 'b'})

    def test_deterministic_for_fixed_seed(self) -> None:
        s1 = RandomSceneSampler(n=4, seed=123)
        s2 = RandomSceneSampler(n=4, seed=123)
        scenes = list('abcdefgh')
        self.assertEqual(s1.sample(scenes), s2.sample(scenes))

    def test_different_seeds_diverge(self) -> None:
        s1 = RandomSceneSampler(n=4, seed=1)
        s2 = RandomSceneSampler(n=4, seed=2)
        scenes = list('abcdefgh')
        self.assertNotEqual(s1.sample(scenes), s2.sample(scenes))

    def test_resamples_on_each_call(self) -> None:
        sampler = RandomSceneSampler(n=3, seed=0)
        scenes = list('abcdefghij')
        first = sampler.sample(scenes)
        second = sampler.sample(scenes)
        self.assertNotEqual(first, second, 'fresh sample expected per call')

    def test_rejects_invalid_n(self) -> None:
        with self.assertRaises(ValueError):
            RandomSceneSampler(n=0)

    def test_rejects_empty_scene_list(self) -> None:
        with self.assertRaises(ValueError):
            RandomSceneSampler(n=2).sample([])


SPORT_GROUPS = {
    'basketball': r'^bb_',
    'football': r'^fb_',
    'volleyball': r'^vb_',
}


def _balanced_scenes() -> list:
    return (
        [f'bb_{i}' for i in range(5)]
        + [f'fb_{i}' for i in range(5)]
        + [f'vb_{i}' for i in range(5)]
    )


class StratifiedSceneSamplerTest(unittest.TestCase):
    """StratifiedSceneSampler regex-based grouping + balance."""

    def test_balanced_sample_across_groups(self) -> None:
        sampler = StratifiedSceneSampler(SPORT_GROUPS, n_per_group=2, seed=0)
        sample = sampler.sample(_balanced_scenes())
        self.assertEqual(len(sample), 6)
        per_group = {g: 0 for g in SPORT_GROUPS}
        for s in sample:
            for g, prefix in (('basketball', 'bb_'), ('football', 'fb_'), ('volleyball', 'vb_')):
                if s.startswith(prefix):
                    per_group[g] += 1
                    break
        self.assertEqual(per_group, {'basketball': 2, 'football': 2, 'volleyball': 2})

    def test_clips_n_per_group_to_pool_size(self) -> None:
        scenes = ['bb_0'] + [f'fb_{i}' for i in range(5)] + [f'vb_{i}' for i in range(5)]
        sampler = StratifiedSceneSampler(SPORT_GROUPS, n_per_group=3, seed=0)
        sample = sampler.sample(scenes)
        # basketball clipped to 1, others get 3 each.
        self.assertEqual(len(sample), 1 + 3 + 3)
        self.assertEqual(sum(1 for s in sample if s.startswith('bb_')), 1)
        self.assertEqual(sum(1 for s in sample if s.startswith('fb_')), 3)
        self.assertEqual(sum(1 for s in sample if s.startswith('vb_')), 3)

    def test_first_match_wins_on_overlap(self) -> None:
        # 'foo' and 'foobar' both match 'foobar_*'; first wins.
        groups = {'foo': r'^foo', 'foobar': r'^foobar_'}
        sampler = StratifiedSceneSampler(groups, n_per_group=2, seed=0, strict=False)
        scenes = ['foobar_1', 'foobar_2', 'foo_1', 'foo_2']
        sample = sampler.sample(scenes)
        # All four scenes match 'foo' first, so 'foobar' bucket is empty;
        # lenient mode skips it. Sample size = 2 (n_per_group from 'foo').
        self.assertEqual(len(sample), 2)
        self.assertTrue(set(sample).issubset(set(scenes)))

    def test_strict_rejects_unmatched_scene(self) -> None:
        sampler = StratifiedSceneSampler(SPORT_GROUPS, n_per_group=1, seed=0)
        scenes = _balanced_scenes() + ['xx_unknown']
        with self.assertRaisesRegex(ValueError, 'no group regex'):
            sampler.sample(scenes)

    def test_lenient_drops_unmatched_scenes(self) -> None:
        sampler = StratifiedSceneSampler(SPORT_GROUPS, n_per_group=1, seed=0, strict=False)
        scenes = _balanced_scenes() + ['xx_unknown', 'yy_unknown']
        sample = sampler.sample(scenes)
        self.assertEqual(len(sample), 3)
        self.assertFalse(any(s.startswith(('xx_', 'yy_')) for s in sample))

    def test_strict_rejects_empty_group(self) -> None:
        # No volleyball scenes in input.
        scenes = [f'bb_{i}' for i in range(3)] + [f'fb_{i}' for i in range(3)]
        sampler = StratifiedSceneSampler(SPORT_GROUPS, n_per_group=1, seed=0)
        with self.assertRaisesRegex(ValueError, 'volleyball'):
            sampler.sample(scenes)

    def test_lenient_skips_empty_group(self) -> None:
        scenes = [f'bb_{i}' for i in range(3)] + [f'fb_{i}' for i in range(3)]
        sampler = StratifiedSceneSampler(SPORT_GROUPS, n_per_group=1, seed=0, strict=False)
        sample = sampler.sample(scenes)
        self.assertEqual(len(sample), 2)

    def test_resamples_on_each_call(self) -> None:
        sampler = StratifiedSceneSampler(SPORT_GROUPS, n_per_group=2, seed=0)
        scenes = _balanced_scenes()
        first = sampler.sample(scenes)
        second = sampler.sample(scenes)
        self.assertNotEqual(first, second, 'fresh sample expected per call')
        # Both still balanced.
        for sample in (first, second):
            self.assertEqual(sum(1 for s in sample if s.startswith('bb_')), 2)
            self.assertEqual(sum(1 for s in sample if s.startswith('fb_')), 2)
            self.assertEqual(sum(1 for s in sample if s.startswith('vb_')), 2)

    def test_deterministic_for_fixed_seed(self) -> None:
        s1 = StratifiedSceneSampler(SPORT_GROUPS, n_per_group=2, seed=123)
        s2 = StratifiedSceneSampler(SPORT_GROUPS, n_per_group=2, seed=123)
        scenes = _balanced_scenes()
        self.assertEqual(s1.sample(scenes), s2.sample(scenes))
        self.assertEqual(s1.sample(scenes), s2.sample(scenes))

    def test_invalid_regex_raises_at_construction(self) -> None:
        with self.assertRaises(re.error):
            StratifiedSceneSampler({'bad': '['}, n_per_group=1, seed=0)

    def test_empty_groups_dict_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StratifiedSceneSampler({}, n_per_group=1, seed=0)

    def test_rejects_invalid_n_per_group(self) -> None:
        with self.assertRaises(ValueError):
            StratifiedSceneSampler(SPORT_GROUPS, n_per_group=0, seed=0)

    def test_rejects_empty_scene_list(self) -> None:
        sampler = StratifiedSceneSampler(SPORT_GROUPS, n_per_group=1, seed=0)
        with self.assertRaises(ValueError):
            sampler.sample([])


class SceneSamplerFactoryTest(unittest.TestCase):
    """scene_sampler_factory dispatches by FactorySpec."""

    def test_builds_random_sampler(self) -> None:
        sampler = scene_sampler_factory(FactorySpec(type='random', params={'n': 5, 'seed': 1}))
        self.assertIsInstance(sampler, RandomSceneSampler)
        self.assertIsInstance(sampler, SceneSampler)

    def test_builds_stratified_sampler(self) -> None:
        sampler = scene_sampler_factory(FactorySpec(
            type='stratified',
            params={'groups': SPORT_GROUPS, 'n_per_group': 1, 'seed': 1},
        ))
        self.assertIsInstance(sampler, StratifiedSceneSampler)
        self.assertIsInstance(sampler, SceneSampler)
        sample = sampler.sample(_balanced_scenes())
        self.assertEqual(len(sample), 3)

    def test_unknown_type_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Unknown scene sampler type'):
            scene_sampler_factory(FactorySpec(type='nope', params={}))

    def test_unknown_param_key_raises(self) -> None:
        with self.assertRaises(TypeError):
            scene_sampler_factory(FactorySpec(type='random', params={'bogus': 1}))

    def test_unknown_stratified_param_key_raises(self) -> None:
        with self.assertRaises(TypeError):
            scene_sampler_factory(FactorySpec(
                type='stratified',
                params={'groups': SPORT_GROUPS, 'bogus': 1},
            ))

    def test_default_params_applied(self) -> None:
        # Empty params should use the variant's defaults (n=8, seed=42).
        sampler = scene_sampler_factory(FactorySpec(type='random', params={}))
        sample = sampler.sample([f's{i}' for i in range(20)])
        self.assertEqual(len(sample), 8)


if __name__ == '__main__':
    unittest.main()
