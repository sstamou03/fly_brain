# Fly Bar

A fruit fly walks into a bar. We give it a drink and simulate its entire brain.

The simulator runs the **male CNS v1.0 connectome** as a leaky integrate-and-fire network:
166,700 neurons, every synapse between them. A drink is turned into a taste vector, fed into
the real gustatory receptor neurons of the legs, labellum and pharynx, and the whole brain is
simulated for a second. If the feeding motor neuron **MN9** fires, the fly extends its
proboscis — it likes the drink.

The web app replays what happened: the taste channels, the circuit from tongue to motor
neuron, and the 3D brain lighting up.

## The brain map

The connectome is the complete wiring diagram of an adult male *Drosophila melanogaster*,
released in 2026 by **Google Research and HHMI Janelia** (with the FlyEM / male-CNS
consortium). Google's team did the machine-learning reconstruction that traced every neuron
and synapse out of the electron-microscopy volume. The data is public at
[male-cns.janelia.org](https://male-cns.janelia.org/download).

We did not train anything — the wiring is theirs, we only run current through it.

## Run it locally

You need [Node.js](https://nodejs.org) 20+.

```bash
cd app
npm install
npm run dev
```

Open http://localhost:5173.

The app ships with the simulation results already exported to `app/public/data`, so you do
**not** need Python, the connectome, or a GPU just to play with it.

### Or with Docker

```bash
docker compose up --build
```

Open http://localhost:8080.

## Layout

```
brain/    the model    sim.py (LIF simulator), taste.py (drinks -> taste neurons), build_brain.py
runs/     experiments  bar.py, hunger.py, calibrate.py, ...
export/   data for the app
app/      React + three.js front end
tests/    sanity checks on toy networks
media/    screenshots
```

## Rebuilding the simulation (optional)

Only needed if you want to change the drinks or the model. Requires Python 3.11+ with
`numpy pandas torch pyarrow`, and the male-CNS v1.0 files from
[male-cns.janelia.org/download](https://male-cns.janelia.org/download) placed in `data/`.

Run everything from the repo root:

```bash
python -m brain.build_brain        # connectome -> brain.npz (~200 MB, a few minutes)
python -m runs.bar                 # serve every drink, rank them
python -m runs.hunger              # how hungry the fly must be to accept each cocktail
python -m export.export_ui         # record brain replays -> ui/data.json
python -m export.export_brain_cloud  # neuron positions -> web_data/
python -m export.export_web        # assemble everything -> app/public/data
```

Sanity check of the simulator: `python -m tests.test_sim`.

The neuron model and constants follow Shiu et al., *Nature* 2024.
