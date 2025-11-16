import numpy as np
import traceback

def try_reshape_and_predict(model, arr):
    """
    Try multiple common reshape permutations to match model.input_shape for Conv1D style models.
    arr: numpy array of mfcc or preprocessed feature (e.g. shape (40,173) or (173,40) or (1,40,173,1))
    Returns: (probs, used_shape) on success, or raises ValueError with diagnostics.
    """
    # Normalize to numpy
    x = np.array(arr)
    diagnostics = []
    # Print model expectation
    expected = model.input_shape  # tuple like (None, steps, features) or (None, steps, features, ...)
    diagnostics.append(f"model.input_shape: {expected}")
    diagnostics.append(f"original array shape: {x.shape}, ndim={x.ndim}")

    # Generate candidate transforms
    candidates = []

    # If arr is 2D: (n_mfcc, frames) or (frames, n_mfcc)
    if x.ndim == 2:
        candidates.append(np.expand_dims(x.T, axis=0))  # (1, frames, n_mfcc)
        candidates.append(np.expand_dims(x, axis=0))    # (1, n_mfcc, frames)
        candidates.append(np.expand_dims(x[..., np.newaxis], axis=0))       # (1, n_mfcc, frames, 1)
        candidates.append(np.expand_dims(x.T[..., np.newaxis], axis=0))     # (1, frames, n_mfcc, 1)

    # If arr is 3D already
    elif x.ndim == 3:
        candidates.append(x)                            # as-is
        candidates.append(x.squeeze())                  # squeezed
        # try swapping last two axes: (1, a, b) -> (1, b, a)
        candidates.append(np.transpose(x, (0, 2, 1)) if x.shape[0] == 1 else np.transpose(x, (0, 2, 1)))
        # try adding channel axis at end if missing
        candidates.append(x[..., np.newaxis])

    # If arr is 4D (you likely have an extra axis)
    elif x.ndim == 4:
        candidates.append(np.squeeze(x, axis=-1))
        candidates.append(np.squeeze(x, axis=0))
        candidates.append(np.squeeze(x))
        candidates.append(np.transpose(np.squeeze(x), (0,2,1)) if x.shape[0] == 1 else np.transpose(np.squeeze(x), (0,2,1)))

    # always include a few more generic attempts
    try:
        # If it's 2D but not considered above (just in case)
        candidates.append(np.expand_dims(x, axis=0))
    except Exception as e:
        diagnostics.append(f"expand_dims failed: {e}")

    # Deduplicate by shape to avoid redundant tries
    shaped = {}
    for cand in candidates:
        shaped[cand.shape] = cand
    final_candidates = list(shaped.values())

    # try predicting with each candidate
    for cand in final_candidates:
        diagnostics.append(f"trying candidate shape: {cand.shape}")
        try:
            preds = model.predict(cand)
            diagnostics.append(f"SUCCESS with shape {cand.shape}")
            return preds[0], cand.shape, diagnostics
        except Exception as e:
            diagnostics.append(f"FAILED for shape {cand.shape}: {type(e).__name__}: {e}")
            # keep trying

    # If none worked, raise with full diagnostics
    raise ValueError("All candidate shapes failed. Diagnostics:\n" + "\n".join(diagnostics))

# Usage in your code (replace how you call predict)
try:
    probs, used_shape, diagnostics = try_reshape_and_predict(model, mfcc)  # mfcc = your extracted mfcc arr
    print("Used input shape for model:", used_shape)
    # decode probs to label etc...
except Exception as e:
    print("Predict failed. See diagnostics below:")
    print(e)
    # optionally, print stack trace for more details:
    traceback.print_exc()


