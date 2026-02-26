# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Patient registry and manifest utilities.

Provides read/write operations on patient consent registries
and record-level manifest CSV files.
"""

import csv
from pathlib import Path


def read_manifest(manifest_path):
    """Read manifest or registry from CSV. Returns list of dicts."""
    records = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def write_manifest(records, output_path, fieldnames=None):
    """Write manifest or registry records to CSV."""
    output_path = Path(output_path)
    if not fieldnames and records:
        fieldnames = list(records[0].keys())

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Written: {len(records)} records -> {output_path}")
    return output_path


def revoke_consent(records, patient_id):
    """Set a patient's consent to revoked. Returns updated list."""
    found = False
    for r in records:
        if r['patient_id'] == patient_id:
            r['consent_status'] = 'revoked'
            found = True
    if not found:
        raise ValueError(f"Patient {patient_id} not found in registry")
    print(f"Consent revoked for {patient_id}")
    return records


def get_patient_ids(records):
    """Get sorted list of unique patient IDs."""
    return sorted(set(r['patient_id'] for r in records))


def get_active_patients(records):
    """Get sorted list of unique patient IDs with active consent."""
    return sorted(set(r['patient_id'] for r in records if r['consent_status'] == 'active'))
