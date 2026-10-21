# Local adaptation

Upstream `right_hand.xml`, referenced meshes, license and README retain their
original bytes, with an immutable commit and SHA-256 hashes in `SOURCE.json`.

`scripts/prepare_embodiments.py` derives `right_hand_xert.xml` by adding named
sites at the compiled tip-geometry center in each distal finger body. All
original joints, actuators and geometry remain; the palm is fixed to the world.
Site positions are simulator geometry definitions, not measured human keypoints.
