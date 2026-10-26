# Flying a Camber vehicle

Three ways to fly what Camber designed, from least to most setup. All three
run the same simulator: MuJoCo, with the vehicle's mass, inertia and rotor
allocation taken straight from its spec (DECISIONS SIM-01).

## 0. Fly it yourself, in the app

Step 5 of the app, "Fly", with **You** as the pilot. It flies in the window,
in real time, and you have the sticks.

| | |
|---|---|
| `W` `S` | pitch — forward, back |
| `A` `D` | roll — left, right |
| `Q` `E` | yaw — turn left, right |
| `Space` `Z` | throttle up, down |
| `R` | reset to where you started |
| `G` | arm / disarm |
| `1` `2` `3` | chase, FPV, orbit camera |
| `Esc` | stop flying |

A gamepad works as well, and better: plug it in, move a stick, and the HUD
says `gamepad`. Mode 2, as sold — left stick throttle and yaw, right stick
pitch and roll.

**Three modes**, the ones a real radio has:

- **Assist** — the sticks ask for a *speed*, and centring them stops the
  vehicle and holds it there. Tilt is limited to 35 degrees. Start here.
- **Angle** — the sticks ask for a bank angle, up to 35 degrees, and the
  throttle is yours. Let go and it levels itself.
- **Acro** — the sticks ask for *rates*. Nothing levels it, and it is the only
  mode that can fly inverted. What FPV racers use.

Throttle is 0 to 1 with **hover at mid-stick** in every mode, on every
vehicle. Without that curve, hover on an 11:1 racer sits at 8.5% of the stick.

The same view watches any pilot fly: pick "Gate race", or one of your own
Python pilots, and press *Fly it here* to watch it in real time, with lap
times, instead of waiting for a recording.

The physics is the same simulator used everywhere else in Camber -- same
integrator, same motors, same aerodynamics -- only paced to the wall clock.
The HUD says `real time` and how much computing headroom is spare; if the
machine cannot keep up it says that instead, rather than quietly running slow.

## 1. From the command line

```console
$ camber sim examples/racer5.toml --viewer                 # gate course, interactive
$ camber sim examples/racer5.toml --record lap.mp4          # same, to video
$ camber sim examples/racer5.toml --pilot hover --viewer
```

In the viewer: drag with the left mouse to orbit, right to pan, scroll to
zoom; double-click the airframe, then Ctrl+right-drag to shove it and watch
the controller catch it. Space pauses.

## 2. Program it in Python

A pilot is a class with `update(state, sim)` returning a motor command in
[0, 1] per rotor. Put it in a file and pass the file:

```console
$ camber sim examples/racer5.toml --pilot examples/pilots/figure_eight.py --viewer
$ camber sim examples/racer5.toml --pilot examples/pilots/flip.py --viewer
```

`state` is in REP-103 axes -- x forward, y left, z up -- and has `position`,
`velocity`, `quaternion` (w, x, y, z), `rates` (gyro, body), `specific_force`
(accelerometer, body), `thrust` (per rotor, N), and `euler`.

You choose how much of the stack to write yourself:

| subclass / call | you provide | you get for free |
|---|---|---|
| `Controller.update` | motor commands | nothing |
| `Cascade.rates(state, rates, thrust, dt)` | body rates + collective | rate loop, mixer |
| `Cascade.attitude(state, quat, thrust, dt)` | attitude + collective | attitude and rate loops, mixer |
| set `self.target / velocity / accel / yaw` on a `Cascade` | a trajectory | everything |

`figure_eight.py` is the trajectory end; `flip.py` flies body rates directly.
The mixer inverts the vehicle's own allocation matrix, so an unusual layout --
a hexacopter, tilted rotors, an offset CG -- mixes correctly with no table to
write.

## 3. Program it over ROS 2

This runs from a clone of the repository and needs [pixi](https://pixi.sh). The
first run downloads ROS 2 Jazzy, about 2.5 GB, into `ros/.pixi`: its own
environment, so nothing is installed system-wide.

```console
$ pixi run -m ros/pixi.toml fly                 # bridge + example pilot
```

That builds the workspace, starts `camber_sim` with the viewer, and starts
`gate_pilot`, an ordinary ROS 2 node that knows nothing about MuJoCo. To run
your own node against the bridge instead:

```console
$ cd ros && pixi shell
$ colcon build --symlink-install --base-paths src && source install/setup.bash
$ ros2 run camber_ros bridge --ros-args -p spec:=../examples/racer5.toml
$ ros2 topic pub /camber/cmd/pose geometry_msgs/PoseStamped \
    "{pose: {position: {x: 3.0, y: 0.0, z: 2.0}, orientation: {w: 1.0}}}"
```

| topic | type | direction |
|---|---|---|
| `/camber/odom` | `nav_msgs/Odometry` | out, 250 Hz, world frame |
| `/camber/imu` | `sensor_msgs/Imu` | out |
| `/camber/thrust` | `std_msgs/Float64MultiArray` | out, N per rotor |
| `/clock`, `/tf` | | out; run your nodes with `use_sim_time:=true` |
| `/camber/cmd/motors` | `std_msgs/Float64MultiArray` | in: [0, 1] per rotor |
| `/camber/cmd/wrench` | `geometry_msgs/Wrench` | in: `force.z` collective N, `torque` body N m |
| `/camber/cmd/pose` | `geometry_msgs/PoseStamped` | in: hold a position and yaw |
| `/camber/cmd/setpoint` | `trajectory_msgs/MultiDOFJointTrajectoryPoint` | in: position + velocity + acceleration |

Whichever command arrived last is flown. Race on `setpoint` -- a pose alone can
only say "go here and stop". If the command stream goes quiet for 0.5 s the
bridge holds position. Only one simulation per ROS domain: the bridge will say
so if it sees another `/clock`.

## Measuring the aerodynamics

```console
$ camber aero examples/racer5.toml --out aero/racer5
$ camber sim examples/racer5.toml --aero aero/racer5/aero.json --viewer
```

`camber aero` does not use a fixed test matrix. It works out the conditions this
vehicle actually flies in -- for a racer, tilts out to 65 degrees nose down at
the speeds those tilts produce; for an endurance quad, gentle tilts at cruise;
for a fixed-wing body, an angle-of-attack sweep -- and runs each through
OpenFOAM with the rotors as actuator discs at the thrust that condition needs
(DECISIONS AERO-01). The result is a table of airframe force and moment
coefficients that replaces the simulator's drag estimate.

## What is and is not modelled

Modelled: rigid-body dynamics and contact, rotor spin-up lag, thrust as speed
squared, rotor reaction torque, airframe drag and moment from CFD or estimate.

Not modelled, and not small at racing speed: rotor behaviour in forward flight
-- H-force, blade flapping, and the change in thrust as the rotor translates --
and the prop wash's effect on the airframe changing with throttle between the
measured trim points. Sensor noise and battery sag are not modelled either.
