"""Explicit, zero-update migration of the accepted Phase42 original donors."""
import json
from pathlib import Path

from causalq.runtime.cohort_runner import capture_rng
from causalq.runtime.identity import file_sha, tree_digest
from causalq.runtime.lora_l1_formal import DIRECTORY, VARIANTS
from causalq.runtime.lora_training import compact_state, load_state_file, rng_digest

AUDIT = DIRECTORY/'phase42_vgpu48_audit_acceptance.json'
RECOVERY = DIRECTORY/'phase42_vgpu48_recovery_acceptance.json'


def accepted():
    audit = json.loads(AUDIT.read_text())
    recovery = json.loads(RECOVERY.read_text())
    steps = {v: audit['donors'][v]['step'] for v in VARIANTS}
    if (not recovery['accepted'] or not recovery['original_donors_unchanged']
            or recovery['engineering_states_eligible_for_formal'] is not False
            or recovery['formal_updates'] != 0 or recovery['engineering_updates'] != 4
            or recovery['model_max_abs'] != 0 or recovery['optimizer_max_abs'] != 0
            or not recovery['all_rng_equal'] or steps != recovery['donor_committed_steps']
            or sum(steps.values()) != 3238):
        raise ValueError('Accepted recovery evidence changed')
    return audit, recovery


def check_runtime(environment):
    _, recovery = accepted()
    if environment != recovery['runtime']:
        raise ValueError('Accepted48GB runtime changed: '+repr(
            sorted(k for k in set(environment) | set(recovery['runtime'])
                   if environment.get(k) != recovery['runtime'].get(k))))


def provenance(variant):
    audit, recovery = accepted()
    row = audit['donors'][variant]
    return {'source_root': audit['source_root'], 'source_producer': audit['source_producer'],
            'source_step': row['step'], 'source_cursor': row['step']*8,
            'source_payload_sha256': row['expected_sha256'], 'source_payload_bytes': row['bytes'],
            'source_identity': row['identity'], 'audit_sha256': file_sha(AUDIT),
            'recovery_acceptance_sha256': file_sha(RECOVERY), 'gate_producer': recovery['gate_producer'],
            'migration_optimizer_updates': 0, 'engineering_states_used': False,
            'old_to_new_gpu_bitwise_equivalence_claimed': False}


def restore_or_migrate(trainer, destination, identity, variant):
    """Restore same-release progress or copy the exact donor state; never advance."""
    from tools.check_lora_vgpu48_resume import verify_donor

    audit, _ = accepted()
    row = audit['donors'][variant]
    # Verify the immutable original even when resuming the migrated run.
    source = Path(audit['source_root'])
    folder = verify_donor(source, variant, row)
    if (trainer.initial_head_sha256 != row['identity']['initial_head_sha256']
            or trainer.initial_adapter_sha256 != row['identity']['initial_adapter_sha256']
            or identity.get('continuation_from') != provenance(variant)):
        raise ValueError('Migration initialization/provenance changed')
    step = trainer.restore(destination, identity)
    if step:
        if step < row['step']: raise ValueError('Migrated progress precedes its original donor')
        return step, {'action': 'same_sha_restore', 'step': step, 'optimizer_updates': 0}
    step = trainer.restore(source/variant/'checkpoints', row['identity'])
    if step != row['step']: raise ValueError('Unexpected newer donor; return a new audit')
    original = load_state_file(folder/'state.pt')
    checks = {'model_equal': tree_digest(compact_state(trainer.model)) == tree_digest(original['model']),
              'optimizer_equal': tree_digest(trainer.optimizer.state_dict()) == tree_digest(original['optimizer']),
              'rng_equal': rng_digest(capture_rng()) == rng_digest(original['rng'])}
    if not all(checks.values()): raise ValueError('Migration changed model/AdamW/RNG')
    return step, dict(checks, action='original_donor_migration', step=step,
                      source_cursor=step*8, optimizer_updates=0,
                      source_payload_sha256=row['expected_sha256'])


def effective_progress(committed):
    """Count original progress exactly once, including donors not yet imported."""
    audit, _ = accepted()
    result = {}
    for variant in VARIANTS:
        step = committed[variant]
        donor = audit['donors'][variant]['step']
        if step and step < donor: raise ValueError('Migrated cursor regressed')
        result[variant] = step or donor
    return result
