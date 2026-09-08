# Real photographs for the sim-to-real test

Drop photos here, then run:

```bash
.venv/bin/python model/predict_real.py --images real_photos/turntable --turntable-step 30
.venv/bin/python model/predict_real.py --images real_photos/plain
.venv/bin/python model/predict_real.py --images real_photos/cluttered
```

## Why the protocol is specific

Absolute 6-DoF pose cannot be labelled by hand -- that is the whole reason this project
generates synthetic data. So the real-photo test gets its numbers two ways, and both
depend on how the photos are captured.

### `turntable/` -- 12 photos, the quantitative test

Put the mug on a plate. Mark 12 positions 30 deg apart around the plate's rim. **Rotate the
plate, not the camera**, one mark per photo. Keep the camera fixed on a stack of books.

Absolute pose stays unlabellable, but the *relative* 30 deg between consecutive frames is
known exactly -- and comparing predicted relative rotations against it is a real error
measurement. On synthetic renders the model scores **3.5 deg median** on this exact metric,
so that is the number the real photos are measured against.

Start with the handle pointing directly to the camera's right, mug upright. That matches
the model's canonical frame: `+Z` up through the cup axis from base to rim, `+X` out
through the handle.

### `plain/` and `cluttered/` -- 5 photos each, the controlled comparison

The same mug, same framing, shot twice: once on a blank sheet of paper or a bare wall,
once on a normally messy desk.

`tools/domain_shift_eval.py` predicts background clutter is the dominant failure mode
(+83.9 deg, versus +1.0 deg for lighting colour). Shooting both conditions tests that
prediction on real data. A gap between the two sets confirms it; no gap refutes it. Either
outcome is a result -- which is why both sets matter more than either alone.

## Framing (all photos)

`tools/controlled_tests.py` found accuracy depends on apparent size, degrading outside
`s/z` of roughly 0.26-0.39. In practice:

- mug spans **about a third of the frame** -- not filling it, not a speck
- mug roughly **centred** (the model centre-crops to square)
- camera **at or slightly above** mug height
- **0.5-1 m** away, normal room lighting
- hold still; defocus blur costs up to +36 deg

Frame it much larger or smaller and you measure the framing penalty rather than the
sim-to-real gap.

## Expect it to fail

The model saw untextured procedural mugs on flat backgrounds. Your mug differs in shape
*and* setting, so shape mismatch and domain gap arrive together and cannot be fully
separated. The honest result is a number plus that caveat -- not a claim of transfer.
