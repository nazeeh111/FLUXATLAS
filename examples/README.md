# Independent reference example

`reference/018aaf1e873bfacdc6f5/` contains a genuinely executed 20,000-history OpenMC fixed-source calculation, its original synthetic library/statepoint, normalized field and metadata. The model XML library path was changed from the original absolute task path to `mgxs.h5` for portability; physics inputs are unchanged. Run `openmc` from that directory to execute a new calculation. Use `fluxatlas example` to regenerate inputs and output with a documented seed.
