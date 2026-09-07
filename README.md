# robot_control

This directory owns canonical lower-level robot contracts and the Real2Sim
artifact pipeline. `sim2real` remains responsible for policy execution and
task orchestration; `hdgp` remains responsible for robot assets and RL.

This Git branch targets Ubuntu 22.04 and ROS 2 Humble only. It contains
validated OpenArm and Tesollo Humble driver snapshots directly under
`ros_ws/src`. Jazzy is maintained on a separate long-lived branch; do not
merge the two distribution branches wholesale.

The first complete profile is `openarm_tesollo`. RH56F1 and the simple gripper
currently provide static component contracts only.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test,hdf5]'
robotctl r2s preflight
robotctl r2s collect --dry-run
```

Install ROS dependencies and build the imported drivers with:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths ros_ws/src --ignore-src -r -y
./ros_ws/build.sh
source ros_ws/install/setup.bash
```

The wrapper rejects non-Humble environments and keeps all generated products
inside `ros_ws/{build,install,log}`.

No command is published unless `--execute` is explicit. A ROS adapter must
also be installed for execution; the core CLI deliberately fails without one.

Calibration JSON v1 is read-only compatibility input. All newly exported
bundles are schema v2, checksum protected, and tied to a profile and asset
manifest hash.

## Handing a calibration to hdgp

`hdgp` reads schema v1 and one scalar per actuator group, so `r2s export` can
write that form beside the bundle:

```bash
robotctl r2s export --bundle bundle.json --validation verdict.json \
    --output exported.json --hdgp real2sim_actuator.json
OPENARM_REAL2SIM_ACTUATOR_CALIBRATION=$PWD/real2sim_actuator.json ./train.sh ...
```

Two things about that conversion are worth knowing before trusting a run.
`get_actuator_params` answers a group name it does not recognise with the
env's own default and reports nothing, so the group each profile group lands
in is declared as `hdgp_group` in the profile rather than guessed; a measured
group without one is refused. And collapsing a group's joints to one scalar is
refused when they disagree by more than `--hdgp-max-spread` of their mean —
the arm's real gains run kp 70 / 60 / 10, where the average describes no joint
in the group. Groups with no measurement are left out, so the env keeps its own
gain; the export names them.

## Teaching poses and trajectories by hand

`robotctl teach` lets an operator move an arm by hand and have it keep the
pose, name poses, and record and replay motions. The motor gains are fixed at
bringup, so compliance is made in software: while a hand is pushing, the
trajectory controller's target is re-commanded to where the arm is, with
gravity feedforward published beside it (see `src/robot_control/teaching.py`).

```bash
./ros_ws/load_effort_controllers.sh right        # the feedforward path
robotctl teach hold   --group openarm_right_arm --gravity 1.1 --payload 0.95,0,0,0.05 --execute
#   space = lock/unlock, s = save the pose as taught_N, q = quit
robotctl teach hold   --group openarm_left_arm --group openarm_left_gripper --gravity 1.0 --execute
#   an arm and its gripper together: the gripper is opened and closed by hand too
robotctl teach hold   --group openarm_right_arm --group tesollo_hand --gravity 1.1 \
    --urdf ~/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf --payload 0.95,-0.0045,-0.0172,0.2215 \
    --soft-p tesollo_hand=1.0 --execute
#   the hand's own loop makes it stiff to push: --soft-p lowers its PID p for the session
robotctl teach record --group openarm_right_arm --gravity 1.1 --output demo.npz --execute
robotctl teach replay --group openarm_right_arm --input demo.npz --repeat 3 --execute
robotctl teach save   --group openarm_right_arm --name pour_start --gravity 1.1
robotctl teach goto   --group openarm_right_arm --name pour_start --execute
robotctl teach list
```

`--group` repeats: every group given is driven in its own lane, and
recordings, poses and replays cover them all. Lanes on different joint-state
topics get their own backend and are merged onto the first lane's clock. The
Tesollo hand is one lane, `tesollo_hand`, joined at runtime from the profile's
four phalanx groups (a partial stream to one controller would undo the other
lanes' hold).
Gravity feedforward goes only to groups with an effort controller. A push is a
deflection past the standing droop (`--push-rad`, default 0.02 rad, or 5% of a
joint's range where that is smaller, as it is for a gripper's stroke); the arm latches once it has rested `--still-sec`, and a push that never
rests is cut off after `--max-follow-sec` as gravity-model drift. A replay is
authorized whole before the arm moves and refused, not slowed, if the
recording asks for more than the profile allows. Every session releases the
feedforward torque when it ends unless `--keep-gravity` is passed, so the arm
sags by its droop; poses saved from inside a session carry the scale that was
holding them and `goto` puts it back on.
