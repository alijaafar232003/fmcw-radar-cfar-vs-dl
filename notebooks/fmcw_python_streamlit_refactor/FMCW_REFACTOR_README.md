# FMCW Radar Range-Doppler Refactor

## Files

- `fmcw_radar_pipeline.py`: core functions + runnable demo.
- `streamlit_app.py`: Streamlit UI that imports the core functions.
- `requirements_fmcw.txt`: minimal dependencies.

## Run the Python demo

```bash
pip install -r requirements_fmcw.txt
python fmcw_radar_pipeline.py
```

Generated outputs:

```text
outputs/demo_rd_map.png
outputs/example_frame/rd_map.npy
outputs/example_frame/rd_map_preview.png
outputs/example_frame/labels.json
```

## Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Preserved notebook logic

The refactor keeps the same core approach:

1. Generate complex analytic TX chirp.
2. Generate RX signal using target delay `tau = 2R(t)/c`.
3. Dechirp with `tx * conj(rx)`.
4. Apply 2D Blackman window.
5. Range FFT along fast-time axis.
6. Keep positive range bins.
7. Doppler FFT along chirp axis.
8. `fftshift` Doppler axis only.
9. Output RD map as rows = range bins, columns = velocity bins.
