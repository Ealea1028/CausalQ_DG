"""Small CPU-only identities and atomic metadata writers."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import torch


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tree_digest(value):
    # Same canonical algorithm as the historical source engine; no tools import.
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(f'tensor:{x.dtype}:{tuple(x.shape)}:'.encode())
            h.update(x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b'dict:')
            for key in sorted(x, key=lambda k: (type(k).__name__, repr(k))):
                visit(key); visit(x[key])
        elif isinstance(x, (tuple, list)):
            h.update(f'{type(x).__name__}:{len(x)}:'.encode())
            for item in x:
                visit(item)
        else:
            h.update(f'{type(x).__name__}:{x!r};'.encode())
    visit(value)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    stage.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    os.replace(stage, path)
