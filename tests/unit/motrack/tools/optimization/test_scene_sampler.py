"""
Tests for ``motrack.tools.optimization.mfgcs.scene_sampler``.
"""
import unittest

from motrack.config_parser import FactorySpec
from motrack.tools.optimization.mfgcs.scene_sampler import (
    RandomSceneSampler,
    SceneSampler,
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


class SceneSamplerFactoryTest(unittest.TestCase):
    """scene_sampler_factory dispatches by FactorySpec."""

    def test_builds_random_sampler(self) -> None:
        sampler = scene_sampler_factory(FactorySpec(type='random', params={'n': 5, 'seed': 1}))
        self.assertIsInstance(sampler, RandomSceneSampler)
        self.assertIsInstance(sampler, SceneSampler)

    def test_unknown_type_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Unknown scene sampler type'):
            scene_sampler_factory(FactorySpec(type='nope', params={}))

    def test_unknown_param_key_raises(self) -> None:
        with self.assertRaises(TypeError):
            scene_sampler_factory(FactorySpec(type='random', params={'bogus': 1}))

    def test_default_params_applied(self) -> None:
        # Empty params should use the variant's defaults (n=8, seed=42).
        sampler = scene_sampler_factory(FactorySpec(type='random', params={}))
        sample = sampler.sample([f's{i}' for i in range(20)])
        self.assertEqual(len(sample), 8)


if __name__ == '__main__':
    unittest.main()
