# Motion capture

- `cmu_35_01_walk.bvh` -- subject 35, trial 01 ("walk")
- `cmu_16_35_jog.bvh` -- subject 16, trial 35 ("run/jog")

Both from the CMU Graphics Lab Motion Capture Database, 120 fps, in the BVH
conversion by Bruce Hahne, taken from https://github.com/una-dinosauria/cmu-mocap.

The data used in this project was obtained from mocap.cs.cmu.edu. The database
was created with funding from NSF EIA-0196217. CMU's terms allow the motion
data to be copied, modified and redistributed without permission.

`tools/export_indiana_game.py` retargets one gait cycle of each onto the
Indiana Jones rig as the "Walk" and "Run" clips.
