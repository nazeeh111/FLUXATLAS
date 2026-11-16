# Current observatory capture

[observatory-current.png](observatory-current.png) is an actual browser capture of the **0.1.1.dev0** interface, recorded on 2026-11-16 at a 1280×900 CSS-pixel viewport. It shows the default retained archive case `test-120`, with total-energy fields and a shared logarithmic color scale. It is not a fabricated interface or a fresh OpenMC execution.

The reference is the previously retained independent OpenMC calculation with seed **3012841** and **40,000 particle histories**, using educational synthetic two-group materials. The paired learned field is the saved hybrid surrogate prediction for that case. The displayed 877.54 ms reference and 0.61 ms prediction timings belong to the retained archive; they are not measurements made while taking this screenshot. The field units are volume-mean scalar flux in cm⁻² per source particle, not dose.

The screenshot records the current readability/accessibility changes and product integration. Scientific arrays, numeric model weights, saved references, calibration and study results retain their existing identities. No new transport calculation, model training or calibration was performed for this image. The original [observatory.png](observatory.png), launch film and earlier media are preserved.

## Byte identities

- Capture SHA256: `a741876375abaeabb905f3efe6fdd250ab417057c3bad5bb05f8f4a1131c6835`.
- [Archive](../web/atlas.json) SHA256: `61fb3a81f319351f3d576280e23649edd6bb5ff90455373ab2767b532646fd9f`.
- [Numeric model weights](../data/model.npz) SHA256: `a5af41901998e85cff727de7d6ee051a8026d3558f69c22f0894a43d398f58cd`.
- [Interface script](../web/app.js) SHA256: `eaa9a9f41b793b415c6ab0c3ee38e5699e9dda33df5ceebf42b6f2e9f4d4c3ac`.
- [Interface HTML](../web/index.html) SHA256: `75aa9a634f6c75bbd85c7a32fcf9f3ccfd6f11fd842038acad6eed95b18a4223`.
- [Interface CSS](../web/style.css) SHA256: `2b6ddfa9e594e21af3eacaf8965a544a839174966d6c720c567a9d054ee5286e`.

These hashes identify the captured bytes and retained inputs; they do not authenticate scientific conclusions or replace the [scope and limits](../README.md#scope-and-limits) and [technical report](../docs/technical-report.md).
