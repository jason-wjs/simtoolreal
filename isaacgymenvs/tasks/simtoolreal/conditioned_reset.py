"""Finite, same-tool reset bank for the explicitly conditioned Gym variant."""
import hashlib
from pathlib import Path
import torch


class ConditionedResetBank:
    def __init__(self, path, device, asset_files):
        path = Path(path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        data = torch.load(path, map_location=device)
        assert len(asset_files) == len(data['asset_sha256']) == 1200
        assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in asset_files] == data['asset_sha256']
        self.object = data['object_root']
        self.table = data['table_root']
        self.dof = data['dof_state']
        self.rows = data['rows_by_tool']
        self.counts = data['counts_by_tool']
        self.lower, self.upper = data['lower'], data['upper']
        assert self.object.shape == self.table.shape == (24576, 13)
        assert self.dof.shape == (24576, 29, 2)
        assert self.rows.shape == (1200, 21) and self.counts.shape == (1200,)
        assert bool((self.counts > 0).all()) and bool((self.counts <= 21).all())
        assert all(bool(torch.isfinite(v).all()) for v in (self.object, self.table, self.dof))
        assert not bool(self.object[:, 7:].any()) and not bool(self.table[:, 7:].any())
        valid = torch.arange(21, device=device)[None, :] < self.counts[:, None]
        assert bool(((self.rows[valid] >= 0) & (self.rows[valid] < 24576)).all())
        assert bool(((self.rows % 1200 == torch.arange(1200, device=device)[:, None]) | ~valid).all())
        assert bool(((self.dof[:, :, 0] >= self.lower) & (self.dof[:, :, 0] <= self.upper)).all())

    def sample(self, env_ids):
        tools = env_ids % 1200
        columns = (torch.rand(len(env_ids), device=env_ids.device) * self.counts[tools]).long()
        return self.rows[tools, columns]
