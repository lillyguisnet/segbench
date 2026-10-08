# Benchmark photos

All photos were taken by our family on 2026-10-08, between 18:41 and 18:45,
with two phones (Android and iPhone). They are the original files from the
phones, at full resolution.

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
