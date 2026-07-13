# Combat AI wire protocol

The schemas in this directory are the public boundary between the Paper arena,
Mineflayer rollout workers, the Python trainer, and the EaglerForge adapter.

All angles are radians. Positions and velocities are expressed in an egocentric
coordinate frame unless a field explicitly says otherwise. Actions represent
ordinary player controls; they never contain a target entity ID or placement
coordinate.

Protocol messages carry a `schema_version` field. A component must reject an
unknown major schema version instead of guessing.
