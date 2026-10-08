"""Load the published numeric pilot with the independently installed DVIDIA package.

No downloads, dynamic remote code, pickle, or hardware control occur here.
"""
from pathlib import Path


def load_visual(root, lane='visual'):
    if lane not in ('visual', 'native'):
        raise ValueError('Choose visual or native.')
    from dvidia_training.footage_train import load_visual_model
    return load_visual_model(Path(root) / lane / 'training' / 'model.json')


def predict_next_rgb(root, rgb, lane='visual'):
    """One 16×16 RGB frame in [0,1] -> one normalized next-frame estimate."""
    import numpy as np
    model, arrays = load_visual(root, lane)
    size = model['training']['resolution']
    frame = np.asarray(rgb, dtype=np.float32)
    if frame.shape != (size, size, 3) or not np.isfinite(frame).all() or (frame < 0).any() or (frame > 1).any():
        raise ValueError('Expected one finite normalized RGB frame at the model resolution.')
    latent = (frame.reshape(-1) - arrays['mean']) @ arrays['basis'].T
    following = np.append(latent, 1.) @ arrays['transition']
    return np.clip(following @ arrays['basis'] + arrays['mean'], 0., 1.).reshape(size, size, 3)


def load_movement(root):
    """Return the validated simulation movement head; no scene is executed."""
    import json
    from dvidia_training.arm_distill import MovementHead
    path = Path(root) / 'native' / 'movement' / 'movement_head.json'
    return MovementHead(json.loads(path.read_text()))
