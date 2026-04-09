# OvercookedV2 Benchmarks — Implementation Plan

Six environments from Gessler et al. (2025). All live within the existing
`cogrid/envs/overcooked/` package.

## Key Finding from Paper/Code

Pots accept **any** ingredient unconditionally. There is no validation at
placement time. `num_ingredients` in JaxMARL only controls observation
encoding. Ingredients labeled `2` (broccoli) and `3` (mushroom) in layouts
are **active distractors** — agents CAN cook with them, producing junk
dishes. Correctness is checked only at delivery time: target recipe = +20,
anything else = -20 (when negative rewards enabled).

---

## What Each Environment Tests

| Category | Envs | R indicator | L button | Incorrect penalty |
|----------|------|:-----------:|:--------:|:-----------------:|
| Grounded Coordination | Simple, Ring | yes | yes (-5 cost) | -20 |
| Test-Time Protocol | Simple, Wide | yes | no | -20 |
| Demo Cook | Simple, Wide | yes | no | none |

All share: 2 agents, `observable_radius=2`, `max_steps=400`,
`target_recipes=["onion_soup","tomato_soup"]`, `resample_on_delivery=True`,
`cook_time=20`, `local_view` feature.

---

## Components to Build

### 1. New ingredients (`overcooked_grid_objects.py`)

```python
Broccoli, BroccoliStack = make_ingredient_and_stack(
    "broccoli", "b", (34, 139, 34), "broccoli_stack", "B",
)
Mushroom, MushroomStack = make_ingredient_and_stack(
    "mushroom", "m", (139, 90, 43), "mushroom_stack", "M",
)
```

### 2. Soup types for all combos (`overcooked_grid_objects.py`)

With 4 ingredients and capacity 3, there are C(4+2,3) = 20 combos.
`onion_soup` and `tomato_soup` already exist (2). Need 18 new soup types.

Add `make_soup(name, char, color)` helper and
`build_open_pot_recipes(ingredients, cook_time)` that:

1. Enumerates all `combinations_with_replacement(ingredients, 3)`
2. For each combo, derives a deterministic result name
   (e.g. `"soup_onion_onion_tomato"`)
3. Reuses existing types (`onion_soup`, `tomato_soup`) where they match
4. Creates new soup types via `make_soup()` for the rest
5. Returns a list of `Recipe` objects

Naming convention: `soup_{ing1}_{ing2}_{ing3}` with ingredients sorted
alphabetically. Pure combos map to existing names:
- `["onion", "onion", "onion"]` → `"onion_soup"`
- `["tomato", "tomato", "tomato"]` → `"tomato_soup"`

### 3. OpenPot (`overcooked_grid_objects.py`)

```python
@register_object_type("open_pot", scope="overcooked")
class OpenPot(GridObj):
    color = Colors.Grey
    char = "u"  # lowercase to coexist with standard Pot 'U'
    container = Container(capacity=3, pickup_requires="plate")
    recipes = V2_POT_RECIPES  # all 20 combos
```

### 4. RecipeIndicator (`overcooked_grid_objects.py`)

```python
@register_object_type("recipe_indicator", scope="overcooked")
class RecipeIndicator(GridObj):
    color = (100, 100, 255)
    char = "R"
    is_wall = True
```

Passive display. Tick writes target recipe ID into `object_state_map`
at R positions. Agents see it via `local_view` when within
`observable_radius`.

### 5. ButtonIndicator (`overcooked_grid_objects.py`)

```python
@register_object_type("button_indicator", scope="overcooked")
class ButtonIndicator(GridObj):
    color = (200, 100, 255)
    char = "L"
    is_wall = True
```

Interactive via Toggle. Custom interaction sets timer. Tick writes recipe
to `object_state_map` at L positions only while timer > 0.

### 6. OpenDeliveryZone (`overcooked_grid_objects.py`)

Separate from existing `delivery_zone` to avoid changing V1 behavior.

```python
@register_object_type("open_delivery_zone", scope="overcooked")
class OpenDeliveryZone(GridObj):
    color = Colors.Green
    char = "X"
    can_place_on = when(agent_holding=V2_SOUP_NAMES)  # all 20 soups
    consumes_on_place = True
```

V2 layouts use `X` instead of `@`. Update char mapping:
JaxMARL `X` → cogrid `X` (open_delivery_zone).

### 7. Target recipe state + tick (`config.py`)

`build_target_recipe_tick(config)` returns a tick that:

1. Decrements `button_timer` for active buttons
2. Writes target recipe ID to `object_state_map` at `R` positions (always)
3. Writes target recipe ID to `object_state_map` at `L` positions
   only when `button_timer > 0`; clears when timer = 0
4. Detects delivery (delivery zone cell cleared) and resamples target
   recipe if `resample_on_delivery=True`

`build_target_recipe_extra_state(config)` returns init dict:
- `overcooked.target_recipe`: `int32` scalar — index into `target_recipes`
- `overcooked.button_timer`: `(n_buttons,) int32`
- `overcooked.button_positions`: `(n_buttons, 2) int32`
- `overcooked.indicator_positions`: `(n_indicators, 2) int32`

Positions are scanned from the parsed grid at init time.

### 8. Button interaction (`config.py`)

```python
def build_branch_activate_button(activation_time=10):
    def branch_activate_button(ctx):
        is_toggle = ctx.action == ctx.action_id.toggle
        button_id = ctx.type_ids.get("button_indicator", -1)
        is_button = ctx.facing_type == button_id
        hands_empty = ctx.held_item == -1
        button_idx, has_match = find_facing_instance(
            ctx.button_positions, ctx.facing_row, ctx.facing_col,
        )
        not_active = ctx.button_timer[button_idx] == 0
        should_apply = (ctx.can_interact & is_toggle & is_button
                        & hands_empty & has_match & not_active)
        new_timer = set_at(ctx.button_timer, button_idx, activation_time)
        return should_apply, {"button_timer": new_timer}
    return branch_activate_button
```

Wired via config `"interactions": [build_branch_activate_button(10)]`.

### 9. TargetRecipeDeliveryReward (`rewards.py`)

Extends `DeliveryReward`. After base class computes delivery mask:

1. Read `target_recipe` index from extra state
2. Map to result type name via `target_recipes[idx]`
3. Compare delivered soup type against target
4. Correct: `+coefficient` (common reward)
5. Incorrect + `penalize_incorrect=True`: `-coefficient`
6. Incorrect + `penalize_incorrect=False`: `0`

### 10. ButtonActivationCost (`rewards.py`)

Detects button_timer going from 0 → activation_time by diffing
prev/current state. Returns `-cost` to all agents (common reward).

---

## Layouts (JaxMARL → cogrid)

Character mapping:
```
W→C  A→+  X→X(open_delivery_zone)  B(plate)→=  P→u(open_pot)
R→R  L→L  0→O  1→T  2→B(broccoli_stack)  3→M(mushroom_stack)
```

| Layout | Grid |
|--------|------|
| grounded_coord_simple | `CCBCCCCC` / `C  C=  O` / `R +Lu+ X` / `C  C=  T` / `CCBCCCCC` |
| grounded_coord_ring | 9×9, see scratch file |
| test_time_simple | Same as gc_simple but `L` replaced with `C` (no button) |
| test_time_wide | 6×7, see scratch file |
| demo_cook_simple | 11×5, see scratch file |
| demo_cook_wide | 11×6, see scratch file |

---

## File Changes

| File | Change |
|------|--------|
| `overcooked_grid_objects.py` | `make_soup()`, `build_open_pot_recipes()`, OpenPot, RecipeIndicator, ButtonIndicator, OpenDeliveryZone, broccoli + mushroom ingredients |
| `config.py` | `build_target_recipe_tick()`, `build_target_recipe_extra_state()`, `build_branch_activate_button()` |
| `rewards.py` | `TargetRecipeDeliveryReward`, `ButtonActivationCost` |
| `features.py` | Layout index updates for 6 new layouts |
| `envs/__init__.py` | 6 layouts + 6 configs + 6 `registry.register()` calls |

---

## Implementation Order

```
1. Ingredients + soups + build_open_pot_recipes  (overcooked_grid_objects.py)
2. OpenPot, RecipeIndicator, ButtonIndicator,    (overcooked_grid_objects.py)
   OpenDeliveryZone
3. Layouts                                     (envs/__init__.py)
4. Target recipe extra state + tick            (config.py)
5. Button interaction branch                   (config.py)
6. Rewards                                     (rewards.py)
7. Configs + registration                      (envs/__init__.py)
8. Layout indices                              (features.py)
9. Tests
10. Docs
```

Steps 1-2 can be done together. Steps 3 and 4-6 can proceed in parallel
after 1-2 complete.
