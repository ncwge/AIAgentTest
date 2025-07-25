"""
ge_matcher.py
----------------

This module contains a simple demonstration of how you might implement a
"closest match" finder for GE Appliances products.  Given a competitor
SKU, the script retrieves (or mocks) the competitor's product details,
normalizes them into a common schema, and then compares those details
against a small catalogue of GE products.  Each GE product is scored
according to how well it matches the competitor on dimensions (width,
height, depth), key feature overlap, and numeric attributes such as
sound level.  The top‑scoring GE product is then returned.

The example below is deliberately simplified: it includes a hard‑coded
catalogue with just two GE products (a gas range and a built‑in
dishwasher) and a mocked competitor product.  In a real deployment you
would replace the ``get_competitor_data`` function with logic to
scrape or query a competitor's website for the product's attributes,
and expand the ``GE_PRODUCTS`` list by loading data from GE's product
feeds or API.  You would also enhance the similarity metric to
account for additional attributes (e.g. BTU ratings, cooktop style,
wash cycles) and tune the weights to your needs.

Usage::

    python ge_matcher.py GDSH4715AF

will produce a report showing the competitor details and the best GE
match.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class Product:
    """A simplified representation of an appliance used for matching."""

    sku: str
    category: str  # e.g. "dishwasher", "range"
    width: float  # in inches
    height: float  # in inches
    depth: float  # in inches
    # For ranges this could be oven capacity (cubic feet), for dishwashers it
    # might be place settings.  None if unknown.
    capacity: Optional[float]
    # A dictionary of binary features.  True indicates presence of a feature.
    features: Dict[str, bool] = field(default_factory=dict)
    # Additional numeric attributes.  For dishwashers this might include noise
    # level in dBA; for ranges it could hold BTU output.
    numerics: Dict[str, Optional[float]] = field(default_factory=dict)


def get_competitor_data(sku: str) -> Product:
    """
    Stubbed function that returns product details for a given competitor SKU.

    In a production system this function would scrape the competitor's
    website (or use a public API) to gather product specifications.  For
    example, for Frigidaire model GDSH4715AF, the product page lists
    features such as CleanBoost™, a 30‑minute fast wash cycle, a
    47‑dBA noise rating and sensor technology【103015234325894†L380-L404】.  Here
    we construct a Product object with those attributes hard‑coded for
    demonstration purposes.
    """
    sku = sku.upper()
    if sku == "GDSH4715AF":
        return Product(
            sku=sku,
            category="dishwasher",
            width=24.0,
            height=33.75,
            depth=26.69,
            capacity=14,  # place settings; approximate
            features={
                "cleanboost": True,
                "fast_wash": True,
                "quiet_operation": True,
                "sensor_technology": True,
                "leak_protection": True,
                "target_spray": True,
                "third_rack": True,
                "adjustable_rack": True,
                "energy_star": True,
            },
            numerics={"noise": 47.0},
        )
    # Fallback: return a generic product with unknown attributes.
    return Product(
        sku=sku,
        category="unknown",
        width=0.0,
        height=0.0,
        depth=0.0,
        capacity=None,
        features={},
        numerics={},
    )


# Example GE product catalogue.  In practice this data would be loaded
# from GE's product feeds or scraped from the GE Appliances website.
# Each product defines its primary dimensions, a few distinguishing
# features, and numeric attributes like noise level.  The dishwasher
# entry below loosely reflects the GDT650SYVFS model, which is a
# 24‑inch, 47‑dBA top‑control dishwasher with Dry Boost™, Active Flood
# Protect and a third rack.
GE_PRODUCTS: List[Product] = [
    Product(
        sku="JGBS66REKSS",
        category="range",
        width=30.0,
        height=47.25,
        depth=28.75,
        capacity=5.0,
        features={
            "edge_to_edge_cooktop": True,
            "integrated_griddle": True,
            "power_boil": True,
            "center_oval_burner": True,
            "steam_clean": True,
            "precise_simmer": True,
            "dishwasher_safe_grates": True,
        },
        numerics={"noise": None},
    ),
    Product(
        sku="GDT650SYVFS",
        category="dishwasher",
        width=23.75,
        height=34.0,
        depth=24.0,
        capacity=16,  # place settings
        features={
            "dry_boost": True,
            "sanitize_cycle": True,
            "third_rack": True,
            "active_flood_protect": True,
            "fast_wash": True,  # 1‑hour wash cycle
            "energy_star": True,
        },
        numerics={"noise": 47.0},
    ),
]


def jaccard_similarity(features_a: Set[str], features_b: Set[str]) -> float:
    """Return the Jaccard similarity between two sets of feature names."""
    if not features_a and not features_b:
        return 0.0
    intersection = features_a.intersection(features_b)
    union = features_a.union(features_b)
    return len(intersection) / len(union)


def compute_similarity(competitor: Product, ge: Product) -> float:
    """
    Compute a heuristic similarity score between two products.  The score
    ranges from 0.0 (no match) to 1.0 (perfect match).  Products of
    different categories automatically receive a score of 0.  The
    similarity is computed as a weighted sum of dimension similarity,
    feature overlap and numeric attribute similarity.
    """
    # Different categories are incomparable.
    if competitor.category != ge.category:
        return 0.0

    # Dimension similarity: use relative difference between widths, heights
    # and depths.  Values closer to 0 difference yield scores nearer 1.
    def dimension_score(a: float, b: float) -> float:
        # Avoid division by zero.
        if a <= 0 or b <= 0:
            return 0.0
        return max(0.0, 1.0 - abs(a - b) / a)

    width_sim = dimension_score(competitor.width, ge.width)
    height_sim = dimension_score(competitor.height, ge.height)
    depth_sim = dimension_score(competitor.depth, ge.depth)
    dimension_sim = (width_sim + height_sim + depth_sim) / 3.0

    # Feature similarity: use Jaccard index over feature names where
    # features are present (True).  Convert keys to lowercase to avoid
    # mismatches due to casing.
    comp_features = {k.lower() for k, v in competitor.features.items() if v}
    ge_features = {k.lower() for k, v in ge.features.items() if v}
    feature_sim = jaccard_similarity(comp_features, ge_features)

    # Numeric similarity: if both products have a noise rating, compute
    # similarity based on relative difference.  Otherwise fall back to 0.
    noise_comp = competitor.numerics.get("noise")
    noise_ge = ge.numerics.get("noise")
    if noise_comp is not None and noise_ge is not None and noise_comp > 0:
        noise_sim = max(0.0, 1.0 - abs(noise_comp - noise_ge) / noise_comp)
    else:
        noise_sim = 0.0

    # Weights can be tuned depending on the domain.  Here we give equal
    # importance to dimensions and features, with a smaller weight for
    # numeric attributes such as noise level.
    weight_dimensions = 0.4
    weight_features = 0.4
    weight_numerics = 0.2

    score = (
        weight_dimensions * dimension_sim
        + weight_features * feature_sim
        + weight_numerics * noise_sim
    )
    return score


def find_best_ge_match(competitor: Product, ge_products: List[Product]) -> Optional[Product]:
    """
    Return the GE product with the highest similarity score to the
    competitor.  If no products have a score greater than zero,
    ``None`` is returned.
    """
    best_score = -1.0
    best_product: Optional[Product] = None
    for ge in ge_products:
        score = compute_similarity(competitor, ge)
        if score > best_score:
            best_score = score
            best_product = ge
    if best_score <= 0.0:
        return None
    return best_product


def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python ge_matcher.py <competitor_sku>")
        return
    sku = argv[1]
    competitor = get_competitor_data(sku)
    if competitor.category == "unknown":
        print(f"No data available for competitor SKU {sku}.")
        return
    match = find_best_ge_match(competitor, GE_PRODUCTS)
    print(f"Competitor SKU: {competitor.sku}")
    print(f"Category: {competitor.category}")
    print(f"Dimensions (W×H×D): {competitor.width}\" × {competitor.height}\" × {competitor.depth}\"")
    noise = competitor.numerics.get("noise")
    if noise is not None:
        print(f"Noise level: {noise} dBA")
    print("Features:")
    for feat, present in competitor.features.items():
        if present:
            print(f"  - {feat}")
    if match is None:
        print("\nNo suitable GE match found.")
        return
    print("\nBest GE match:")
    print(f"  SKU: {match.sku}")
    print(f"  Category: {match.category}")
    print(f"  Dimensions (W×H×D): {match.width}\" × {match.height}\" × {match.depth}\"")
    if match.numerics.get("noise") is not None:
        print(f"  Noise level: {match.numerics['noise']} dBA")
    if match.capacity is not None:
        print(f"  Capacity: {match.capacity}")
    print("  Features:")
    for feat, present in match.features.items():
        if present:
            print(f"    - {feat}")


if __name__ == "__main__":
    main(sys.argv)