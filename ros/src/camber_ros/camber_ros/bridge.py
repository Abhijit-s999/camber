"""camber_sim: the designed vehicle in MuJoCo, as a ROS 2 node.

Publishes (world frame is REP-103: x forward, y left, z up -- MuJoCo's own):
    /clock                  rosgraph_msgs/Clock      simulation time
    /camber/odom            nav_msgs/Odometry        centre of mass, world
    /camber/imu             sensor_msgs/Imu          body rates, specific force
    /camber/thrust          std_msgs/Float64MultiArray   per rotor, N
    /tf                     world -> base_link

Accepts commands at three levels; whichever arrived most recently is flown:
    /camber/cmd/motors      std_msgs/Float64MultiArray   [0, 1] per rotor
    /camber/cmd/wrench      geometry_msgs/Wrench         force.z = collective
                                                          thrust N, torque =
                                                          body moments N m
    /camber/cmd/pose        geometry_msgs/PoseStamped    hold position + yaw
    /camber/cmd/setpoint    trajectory_msgs/MultiDOFJointTrajectoryPoint
                                                          position + yaw with
                                                          velocity and
                                                          acceleration feed-
                                                          forward: the one to
                                                          race with

A command stream that goes quiet for `timeout` seconds is treated as lost: the
bridge falls back to holding the last position rather than flying the last
command forever. A pilot node that crashes should not take the vehicle with it.

Run with `use_sim_time:=true` on your own nodes so their clocks follow /clock.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Wrench
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

from camber.sim import Simulation
from camber.sim.course import Course
from camber.sim.flight import Cascade, Mixer
from camber.spec import io


def _stamp(t: float):
    from builtin_interfaces.msg import Time
    s = int(t)
    return Time(sec=s, nanosec=int((t - s) * 1e9))


class Bridge(Node):
    def __init__(self) -> None:
        super().__init__("camber_sim")
        spec = self.declare_parameter("spec", "examples/racer5.toml").value
        course = self.declare_parameter("course", "oval").value
        self.publish_hz = float(self.declare_parameter("publish_hz", 250.0).value)
        self.realtime = bool(self.declare_parameter("realtime", True).value)
        viewer = bool(self.declare_parameter("viewer", True).value)
        self.timeout = float(self.declare_parameter("timeout", 0.5).value)

        self.vehicle = io.load(spec)
        self.course = Course.oval() if course == "oval" else None
        self.sim = Simulation(self.vehicle, course=self.course)
        self.state = self.sim.reset()
        self.mixer = Mixer(self.sim.allocation, self.sim.max_thrust)
        self.hold = Cascade()
        self.hold.reset(self.sim)
        self.hold.target = self.state.position + np.array([0.0, 0.0, 1.0])
        self.mode, self.command, self.last_cmd = "pose", None, None

        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_odom = self.create_publisher(Odometry, "/camber/odom", 10)
        self.pub_imu = self.create_publisher(Imu, "/camber/imu", 10)
        self.pub_thrust = self.create_publisher(Float64MultiArray, "/camber/thrust", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Float64MultiArray, "/camber/cmd/motors",
                                 self._on_motors, 10)
        self.create_subscription(Wrench, "/camber/cmd/wrench", self._on_wrench, 10)
        self.create_subscription(PoseStamped, "/camber/cmd/pose", self._on_pose, 10)
        self.create_subscription(MultiDOFJointTrajectoryPoint,
                                 "/camber/cmd/setpoint", self._on_setpoint, 10)

        self.viewer = None
        if viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
        # Two simulations on one ROS domain both publish /clock, and every
        # sim-time node then sees time jump back and forth -- timers fire at
        # the wrong rate and the odometry interleaves two unrelated vehicles,
        # with no error anywhere. It happened in testing, from processes that
        # outlived their launch. Look once, shortly after starting, and say so.
        self.create_timer(1.0, self._check_alone)
        n = len(self.vehicle.thrusters)
        self.get_logger().info(
            f"flying {self.vehicle.name}: {self.sim.mass * 1000:.0f} g, {n} rotors, "
            f"dt {self.sim.dt * 1000:.1f} ms; waiting for commands")

    def _check_alone(self) -> None:
        if getattr(self, "_checked", False):
            return
        self._checked = True
        others = self.count_publishers("/clock") - 1
        if others > 0:
            self.get_logger().error(
                f"{others} other node(s) are publishing /clock on this domain: "
                f"another simulation is running, and sim time will be garbage. "
                f"Stop it, or set a different ROS_DOMAIN_ID.")

    # --- commands --------------------------------------------------------
    def _fresh(self) -> None:
        self.last_cmd = self.sim.t

    def _on_motors(self, msg: Float64MultiArray) -> None:
        u = np.asarray(msg.data, dtype=float)
        if u.shape != self.sim.max_thrust.shape:
            self.get_logger().warn(f"motor command has {u.size} values, "
                                   f"vehicle has {self.sim.max_thrust.size} rotors")
            return
        self.mode, self.command = "motors", np.clip(u, 0.0, 1.0)
        self._fresh()

    def _on_wrench(self, msg: Wrench) -> None:
        self.mode = "wrench"
        self.command = (msg.force.z, np.array([msg.torque.x, msg.torque.y,
                                               msg.torque.z]))
        self._fresh()

    def _on_pose(self, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        self.hold.target = np.array([p.x, p.y, p.z])
        self.hold.velocity = np.zeros(3)
        self.hold.yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                         1 - 2 * (q.y * q.y + q.z * q.z)))
        self.mode = "pose"
        self._fresh()

    def _on_setpoint(self, msg: MultiDOFJointTrajectoryPoint) -> None:
        """A pose alone can only say "go here and stop". Racing needs "pass
        through here at this velocity", which is what the feed-forward terms
        carry; without them a carrot-chasing pilot settles at about 2 m/s."""
        if not msg.transforms:
            return
        tr = msg.transforms[0]
        self.hold.target = np.array([tr.translation.x, tr.translation.y,
                                     tr.translation.z])
        q = tr.rotation
        self.hold.yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                         1 - 2 * (q.y * q.y + q.z * q.z)))
        v = msg.velocities[0].linear if msg.velocities else None
        a = msg.accelerations[0].linear if msg.accelerations else None
        self.hold.velocity = np.array([v.x, v.y, v.z]) if v else np.zeros(3)
        self.hold.accel = np.array([a.x, a.y, a.z]) if a else np.zeros(3)
        self.mode = "pose"
        self._fresh()

    def _control(self) -> np.ndarray:
        stale = self.last_cmd is not None and self.sim.t - self.last_cmd > self.timeout
        moving = (np.linalg.norm(self.hold.velocity) > 0
                  or np.linalg.norm(self.hold.accel) > 0)
        if stale and (self.mode != "pose" or moving):
            self.get_logger().warn("command stream lost; holding position")
            self.hold.target = self.state.position.copy()
            self.hold.velocity = np.zeros(3)
            self.hold.accel = np.zeros(3)
            self.mode = "pose"
            self.last_cmd = None
        if self.mode == "motors":
            return self.command
        if self.mode == "wrench":
            fz, moment = self.command
            return self.mixer.commands(fz, moment)
        return self.hold.update(self.state, self.sim)

    # --- publishing ------------------------------------------------------
    def _publish(self) -> None:
        s, t = self.state, _stamp(self.sim.t)
        self.pub_clock.publish(Clock(clock=t))
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = t, "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, \
            odom.pose.pose.position.z = map(float, s.position)
        w, x, y, z = map(float, s.quaternion)
        o = odom.pose.pose.orientation
        o.w, o.x, o.y, o.z = w, x, y, z
        v_body = s.rotation.T @ s.velocity           # twist is in child frame
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, \
            odom.twist.twist.linear.z = map(float, v_body)
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, \
            odom.twist.twist.angular.z = map(float, s.rates)
        self.pub_odom.publish(odom)

        imu = Imu()
        imu.header.stamp, imu.header.frame_id = t, "base_link"
        imu.orientation = odom.pose.pose.orientation
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = \
            map(float, s.rates)
        imu.linear_acceleration.x, imu.linear_acceleration.y, \
            imu.linear_acceleration.z = map(float, s.specific_force)
        self.pub_imu.publish(imu)
        self.pub_thrust.publish(Float64MultiArray(data=[float(x) for x in s.thrust]))

        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = t, "world", "base_link"
        tf.transform.translation.x, tf.transform.translation.y, \
            tf.transform.translation.z = map(float, s.position)
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform(tf)

    def run(self) -> None:
        per_publish = max(1, int(round(1.0 / (self.publish_hz * self.sim.dt))))
        wall0 = time.perf_counter()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            for _ in range(per_publish):
                self.state = self.sim.step(self._control())
            self._publish()
            if self.viewer is not None:
                if not self.viewer.is_running():
                    break
                self.viewer.sync()
            if self.realtime:
                lag = self.sim.t - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)


def main() -> None:
    rclpy.init()
    node = Bridge()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node.viewer is not None:
            node.viewer.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
