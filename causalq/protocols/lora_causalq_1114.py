"""Study registry; only Phase39 backbone-gate is released."""
STUDY = 'lora_causalq_1114'
VARIANTS = ('U0', 'UA', 'C0', 'CA', 'B1', 'B2', 'B3')
BACKBONES = ('dinov3_vitl16', 'dinov2_vitl14_hpi_p16')
MATRIX = tuple(f'{b}_{v}_L{l}_S0' for b in BACKBONES for v in VARIANTS for l in (0, 1))
PHASE = 39
FORMAL_TRAINING_AUTHORIZED = False


def require_action(action):
    if action != 'backbone-gate':
        raise ValueError('Phase39 releases backbone-gate only; no training, data or target access')
