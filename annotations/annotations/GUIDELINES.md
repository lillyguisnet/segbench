# segbench answer-key guidelines

Edit this file in the app (**guidelines**). Write each new decision with the
image id it came from. The task list and reasons: `docs/task-ideas.md`.

## For every task

- **One mask per object**, every object in scope, however small or far.
  No crowd masks.
- **Visible pixels only.** Do not guess the hidden part of an object. An
  object cut in two by something in front of it is ONE mask with two parts.
- **Check every mask at full zoom**, especially its edges, whether it came
  from SAM or by hand. A SAM suggestion is a starting point, not an answer.
- Exactly one **Object** label per mask; dishes also get one **Dish state**.
- Image done = every object in scope has an accepted, labelled mask and no
  suggestion is left.

## Per task

1. `log_ends_closeup_1`: every visible **log end** (the cut face, bark ring
   included), also the small and half-hidden ones at the edges. Logs seen
   from the side only (no cut face) are not in scope.
   **Use SAM clicks here (one click per log end), not a SAM text concept:**
   with ~70 objects at 4000×2252 a text concept runs out of GPU memory
   (measured 2026-10-09); a click takes ~1 s.
2. `field_with_cows`: every **cow**, including the distant ones and the
   brown one. Calves count as cows.
3. `fig_plant`: every **fig leaf** (large lobed leaves). Not the spider
   plant (long thin leaves) or the purple plant. Petiole (leaf stalk) excluded.
4. `autumn_tree_orange_1`: one **tree crown** mask: the main tree's foliage
   and branches. Not the background trees. The red share is computed inside
   it by a colour rule, so do not try to separate red from yellow by hand.
5. `farm_road_1`: one **road** mask: the gravel surface, from the bottom of
   the picture to where it disappears. Not the grass verges.
6. `kitchen_counter_dishes`: every **dish** (plate, bowl, mug, pot, pan,
   baking tray, lid, cutting board, food container), with **dirty** (on the
   counter or in the sink, or food visible), **clean** (in the drying rack,
   no food visible) or **unsure** (the clues disagree; not scored). Not
   cutlery or knives, bottles, appliances, cloths. Stacked bowls: one mask
   per bowl, visible part only.

## Bias check (README, "How the answer key is made")

For ~10 objects (5 cows, 5 dishes) draw a **second** mask entirely by hand
(poly tool, SAM off), labelled the same, and write "hand-only" in the mask
note. These are used only to measure whether SAM-assisted masks favour SAM.
