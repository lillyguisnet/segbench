# Benchmark photos

The first 18 photos were taken by our family on 2026-10-08, between 18:41
and 18:45, with two phones (Android and iPhone). They are the original files
from the phones, at full resolution. Eight more were added on 2026-10-14
(second table below); they are smaller copies, not phone originals.

Before they were added here, all hidden data inside the files was removed
(GPS location, phone model and serial number, capture time). Only the
image orientation and the colour profile were kept. The pixels are exactly
the camera's. If you add a photo, strip it the same way first:

```bash
exiftool -overwrite_original -all= -tagsfromfile @ -icc_profile -orientation PHOTO.jpg
```

**Licence:** [Creative Commons Attribution 4.0 International](LICENSE-CC-BY-4.0.txt)
(CC BY 4.0). You may use, share and change these photos for any purpose,
including commercial use, as long as you credit them, for example:

> Photos: segbench authors, CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

The code of this project is under Apache 2.0 (see `../LICENSE`); that
licence does not cover the photos.

| file | what it shows | phone | planned task |
|---|---|---|---|
| `fig_plant.jpg` | fig plant among other house plants | Android | fig tree: each leaf |
| `kitchen_counter_dishes.jpg` | kitchen counter with dirty dishes and pots | Android | stack of dishes: each dish |
| `fallen_leaves_on_grass.jpg` | red and yellow leaves scattered on grass | Android | |
| `autumn_tree_orange_1.jpg` | orange autumn tree against the sky | Android | autumn leaves: reddish clumps |
| `autumn_tree_orange_2.jpg` | orange autumn tree against the sky | iPhone | autumn leaves: reddish clumps |
| `maple_leaves_dusk.jpg` | green and yellow maple leaves, low light | iPhone | |
| `log_pile_barn_1.jpg` | log pile in front of an old barn | iPhone | log pile: each log |
| `log_pile_barn_2.jpg` | log pile in front of an old barn | iPhone | log pile: each log |
| `log_pile_barn_3.jpg` | log pile in front of an old barn | iPhone | log pile: each log |
| `log_pile_barn_4.jpg` | log pile in front of an old barn | Android | log pile: each log |
| `log_ends_closeup_1.jpg` | close-up of the cut log ends | Android | log pile: each log end |
| `log_ends_closeup_2.jpg` | close-up of the cut log ends | Android | log pile: each log end |
| `farm_road_1.jpg` | gravel road, barn, yellow tree (portrait) | Android | |
| `farm_road_2.jpg` | gravel road, barn, yellow tree (landscape) | Android | |
| `red_house_driveway.jpg` | red brick house and car at the end of a driveway | Android | |
| `field_with_cows.jpg` | pasture with distant cows | Android | |
| `sunset_field_1.jpg` | sunset over a field | iPhone | |
| `sunset_field_2.jpg` | sunset over a field | iPhone | |

### Added 2026-10-14

Eight photos received in the project conversation of 2026-10-14, already
resized (0.5 to 1.2 megapixels, longest side 1015 to 1439 pixels; the
originals above are about 9 megapixels) and already without any capture
data: no date, phone or location, only a Google sRGB colour profile, so
they came through Google Photos. The strip command above was run anyway;
the pixels are unchanged. Not in any task, run or answer key yet.

| file | what it shows | size | planned task |
|---|---|---|---|
| `leafcutter_ants_branch.jpg` | leaf-cutter ants carrying leaf pieces along a branch; a blurred person far left | 1439x810 | |
| `butterfly_display_case.jpg` | pinned butterflies and beetles arranged symmetrically in a case | 743x1319 | |
| `living_plant_wall.jpg` | indoor living wall of pothos, ferns and purple-leaved plants | 727x1030 | |
| `blue_mineral_specimen.jpg` | blue, green and brown mineral specimen, close-up | 1439x810 | |
| `mounted_birds_display.jpg` | museum case of mounted birds in flight (geese, swan, hawks, owl) | 743x1319 | |
| `oyster_mushrooms_hands.jpg` | a cluster of oyster mushrooms held in two hands over dead leaves | 1015x1015 | |
| `strawberry_coulis_jars.jpg` | jars of strawberry coulis and other preserves on a wooden table | 1015x457 | |
| `cows_under_pine_tree.jpg` | black cows and a calf under a pine tree, more cows far away in the pasture; one far cow, lying down, is white with a bit of red | 1439x810 | the white cow with a bit of red (only one; the black cows are distractors) |

