# Consensus review

A standalone, local-first editor for the six-task consensus. No server API,
model calls, credentials, or writes to recorn's answer keys.

## Build and open

From the project root:

```sh
uv run scripts/build_consensus_review.py
# Or choose a different saved consensus:
uv run scripts/build_consensus_review.py results/consensus-pilot-1.json
```

Open this folder with Chattering's artifact tool, or serve the project
locally and open `apps/consensus-review/index.html`. The builder snapshots the
consensus into `seed.js` and copies the full-resolution photos into `images/`.
Those generated assets are ignored by Git; the source photos and consensus
already live in the repository. Original images must be stored upright.

## Fast review

- **Next uncertain (U)** jumps to disagreement cases, zoomed in.
- **Next unreviewed (N)** visits the remaining visible proposals.
- **Enter** keeps an object and advances. **Delete** rejects it and advances.
- Drag to move a dot; **A** adds a missed object; **V** selects; **X** removes.
- For dishes, the **Dirty / Clean / Unsure** buttons stay visible directly
  above the photo. Select a dot to enable them; **D / C / S** are the keyboard
  shortcuts. Each visible dish dot displays **D / C / ?** for its current
  label (model suggestions are still unreviewed). Labels save immediately.
  Keep the object to confirm it; changing its label alone doesn't accept it.
- Wheel zooms around the cursor; Space + drag pans; **F** fits the photo.
- Two fingers pan/zoom on touchscreens; a dot can be dragged with one finger.
- Hold **H** to inspect the photo without overlays.
- **Ctrl/Cmd Z** undoes; Shift Z or Ctrl/Cmd Y redoes. Histories are per photo,
  bounded to 60 changes and live only for the current page session.
- All source proposals begin **unreviewed**, including high-agreement ones.
  Only human-confirmed or human-added points appear green.
- Low-support proposals are initially hidden, not deleted. Reveal them with
  the checkbox; still inspect the entire photo for objects no model found.

Tree and road have paint/erase brushes. These edit the original, coarse
consensus grid (300 pixels on its longest side), **not** a full-resolution
mask. Export includes explicit grid dimensions. Red foliage area is a share
of the whole photo, not the crown. For final fine-edged masks, use recorn.

## Saving and transfer

Edits autosave to browser localStorage, keyed by the source file's SHA-256.
This is browser/origin-specific, not a server save or cross-device sync.
If storage is blocked or full, the status warns and JSON export still works.
Concurrent edits in another tab stop autosave rather than silently replacing
newer work. Export that tab's work before reloading to resolve the conflict.

**Export corrections** downloads JSON with:

- source snapshot and its hash, original image hashes and dimensions;
- original-pixel coordinates, stable candidate IDs and human-added IDs;
- kept / rejected / pending / unsure decisions, dish labels and review notes;
- coarse region masks as row-major binary strings with explicit grid sizes;
- full-photo-check flags and an action log, including undo/redo/import.

Use **Import** to restore that file in another browser/device. Import validates
the schema, source snapshot, image dimensions/hashes, point coordinates,
statuses, labels, original candidate IDs and region dimensions before replacing
anything. Files from different consensus versions are refused; there is no
silent rebase of human corrections onto new model suggestions.

A complete-photo-check flag does **not** automatically accept remaining dots.
Corrected points are annotation hints, not masks, and the export is not a
recorn import file or a benchmark answer key. Nothing updates the provisional
rankings automatically. The source consensus remains unchanged.

Model agreement indices are heuristic, **not calibrated correctness
probabilities**. Review some high-agreement areas and all missed-object areas,
not just the disagreement queue. Maintain the independent hand-annotation
checks described in the main README to detect anchoring bias.

## Tests

```sh
uv run scripts/test_consensus_review.py URL /path/to/chromium
```

Use the actual artifact preview URL when testing. The test uses separate,
fresh browser profiles and exercises rendering, dragging, adding, rejecting,
labels, keyboard acceptance, painting, undo/redo, reload persistence,
export/import, invalid-import rejection, mobile layout, pinch gestures,
blocked storage and concurrent-tab conflicts. It never edits the user's draft.
