# Bundled assets

Everything here is served locally; nothing is fetched from third-party hosts at runtime.

| File | Source | Licence | Notes |
| --- | --- | --- | --- |
| `textures/earth-bmng-july-4k.jpg` (~1.0 MB) | NASA Earth Observatory, *Blue Marble: Next Generation* with topography and bathymetry, July 2004 (`world.topo.bathy.200407.3x5400x2700.jpg`), Reto Stöckli, NASA GSFC | Public domain (NASA imagery, no copyright) | Resized from 5400×2700 to 4096×2048 (JPEG q84). Used in the landing's 3D globe only as a land/ocean mask and relief, shaded in greys. It is base imagery, not weather data. |
| `data/initial_state.json`, `data/static_fields.json` | Exported from this project's own training data | Project data | The real Day-0 tensors behind every field in the story. |

Fonts are bundled through npm (`@fontsource-variable/inter`, `@fontsource-variable/jetbrains-mono`, `@fontsource/instrument-serif`), all under the SIL Open Font License 1.1. Coastlines come from Natural Earth (public domain) via `world-atlas` and are inlined in `src/assets/geo/`.
