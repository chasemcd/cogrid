# Overcooked

A cooperative multi-agent cooking environment based on
[Carroll et al. (2019)](https://arxiv.org/abs/1910.05789). Two agents share a
kitchen and must coordinate to pick up ingredients, cook soups in pots, plate
the finished dishes, and deliver them for reward.

Object behavior is declared via `Container`, `Recipe`, and `when()` class
attributes. The autowire system auto-generates all array-level interaction,
tick, and state code at init time.

See the full documentation at `docs/environments/overcooked.md`.