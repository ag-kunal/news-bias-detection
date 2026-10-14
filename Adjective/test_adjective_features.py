from __future__ import annotations

import unittest
import math
from types import SimpleNamespace

import pandas as pd

from analyze_and_plot import RATE_COLUMN, cluster_bootstrap_means
from extract_features import compute_title_features


def token(pos: str, is_alpha: bool = True) -> SimpleNamespace:
    return SimpleNamespace(pos_=pos, is_alpha=is_alpha)


class TitleFeatureTests(unittest.TestCase):
    def test_counts_adjectives_and_normalizes_by_alphabetic_tokens(self) -> None:
        features = compute_title_features(
            "Bright, bold leaders respond",
            [token("ADJ"), token("ADJ"), token("NOUN"), token("VERB"), token("PUNCT", False)],
        )
        self.assertEqual(features["title_adjective_count"], 2)
        self.assertEqual(features["title_pos_token_count"], 4)
        self.assertEqual(features["title_adjective_rate_per_100_tokens"], 50.0)
        self.assertTrue(features["title_has_adjective"])
        self.assertTrue(features["title_is_valid"])

    def test_empty_title_has_undefined_rate(self) -> None:
        features = compute_title_features("  ", [])
        self.assertEqual(features["title_adjective_count"], 0)
        self.assertEqual(features["title_pos_token_count"], 0)
        self.assertTrue(math.isnan(float(features["title_adjective_rate_per_100_tokens"])))
        self.assertFalse(features["title_has_adjective"])
        self.assertFalse(features["title_is_valid"])

    def test_cluster_bootstrap_fills_biases_absent_from_a_cluster(self) -> None:
        frame = pd.DataFrame({
            "_cluster_id": ["story-a", "story-a", "story-b"],
            "Bias": ["left", "right", "center"],
            RATE_COLUMN: [10.0, 20.0, 30.0],
        })
        means, differences, cluster_count = cluster_bootstrap_means(
            frame,
            "_cluster_id",
            replicates=100,
            seed=42,
        )
        self.assertEqual(cluster_count, 2)
        self.assertEqual(set(means), {"left", "center", "right"})
        self.assertEqual(len(differences), 3)
        for lower, upper in means.values():
            self.assertTrue(math.isfinite(lower))
            self.assertTrue(math.isfinite(upper))


if __name__ == "__main__":
    unittest.main()