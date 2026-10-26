# FLUXATLAS launch film

The film is a 20-second, 1920×1080 local HTML composition. `brag.mp4` is the final delivery, `brag.jpg` is its settled comparison poster, and `composition/index.html` is the editable source. No narration or music is included: the quiet instrument-film treatment is intentional.

The visual evidence is exported from the completed `web/atlas.json` archive by `export-evidence.py`. The reference/model case is chosen by proximity to the median held-out relative L₂ error. `composition/assets/provenance.json` records its ID, input configuration, reference seed/history count, numerical scale limits and exact archive SHA-256. Cells and acquisition points are shown without smoothing. The final scene contains a genuine browser capture of the observatory, rather than a fabricated mockup.

To refresh the composition after a numerical study and current-source screenshot have been verified:

```sh
python media/build-composition.py path/to/current-observatory.png
```

Render `media/composition` using a compatible Hyperframes runtime with Chrome and FFmpeg. The validated build uses the installed pinned Hyperframes 0.8.33 local runner, with standard quality, 30 fps and one rendering worker. A renderer is optional for normal FLUXATLAS operation. No paid API or cloud rendering is required.

The strongest settled comparison frame is selected as the poster, then baked into the first video frame so the bare MP4 presents the same accurate thumbnail. See `verification.json` for measured output properties and retained checks. The raw renderer output and pre-fix visual-check history are retained in task-local work storage. The released folder contains the verified current composition and media.

Animation-library attribution is in [THIRD-PARTY.md](THIRD-PARTY.md). The visualization scripts, generated scientific graphics and composition are original project material. Numerical evidence may change when the study changes; refresh the exports instead of editing numbers into the film.
