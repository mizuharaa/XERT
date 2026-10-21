# Local adaptation

Upstream `g1.xml`, referenced meshes, license and README retain their original
bytes, with an immutable commit and SHA-256 hashes in `SOURCE.json`.

`scripts/prepare_embodiments.py` derives `g1_upper.xml`: it removes the pelvis's
free joint and both leg body subtrees, removes the corresponding leg actuators
and the full-body keyframe, and adds named sites at the wrist-yaw body origins.
The pelvis is consequently fixed to the world; the waist and both arms remain
actuated. These wrist-frame sites are not fingertip tracking measurements.

This simulation-only adaptation has no balancing or locomotion controller.
