# Answer keys (masks), made in recorn

One mask per object for the six tasks, made in Lilly's annotation app
[recorn](https://github.com/lillyguisnet/recorn) (private repo; branch
`local-remote` adds the "run on the GPU host" mode used here), with
SAM 3.1 suggestions. What to draw: `annotations/GUIDELINES.md`. Why masks,
and the safeguards against SAM bias: the main README.

## Open it

**https://lambda.tail69222b.ts.net:18771** from any device on our
Tailscale network (tailnet only, not public). On lambda itself:
http://127.0.0.1:8771.

## What runs where (all on lambda)

| piece | how | undo |
|---|---|---|
| app | `systemctl --user status segbench-recorn` (transient; gone at reboot) | `systemctl --user stop segbench-recorn` |
| start it again | `systemd-run --user --unit=segbench-recorn --setenv=CORN_ROOT=$HOME/Projects/segbench/annotations --working-directory=$HOME/Projects/recorn -p Restart=on-failure bash -lc 'exec uv run recorn serve --host 127.0.0.1 --port 8771'` | |
| tailnet address | `tailscale serve --bg --https=18771 http://127.0.0.1:8771` | `tailscale serve --https=18771 off` |
| SAM 3.1 | `sam31.service`, moved to GPU 0 until reboot (`docs/gpu-setup.md`) | see there |

## Files here

- `config.json`: recorn settings (`"host": "local"`, the SAM text concepts
  used for batch suggestions).
- `annotations/`: **the answer key** (one JSON per photo, masks as COCO
  RLE, with where each mask came from and how many pixels were edited by
  hand), plus `labels.json` and `GUIDELINES.md`.
- `proposals/`: SAM's raw suggestions.
- `data/segbench/manifest.json`: the six photos (canonical PNGs are
  rebuilt by `scripts/prepare_annotation_set.py`, not committed).
- `backup/`, `backup-history/`: recorn's automatic copies (not committed).

The app's **commit** button commits `annotations/` and `proposals/` in
this repository and pushes it (public), as Lilly.

## Known limits (2026-10-09)

- **Logs: click, don't use a text concept.** A SAM text concept on the log
  photo (~70 objects, 4000×2252) runs out of GPU memory; a click per log
  end works (~1 s each).
- **"fig leaf" finds nothing; "leaf" finds 31** (all three plants). Use
  "leaf", then keep only the fig leaves.
- **Other jobs can use GPU 0** while it is free of the website's 9B (a
  "figurer" test was on it on 2026-10-09). Less memory for SAM if so.
