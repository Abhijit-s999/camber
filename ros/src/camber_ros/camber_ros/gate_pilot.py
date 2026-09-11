"""gate_pilot: an example of programming the vehicle over ROS 2.

Written the way a user would write one -- its own node, knowing nothing about
MuJoCo, reading /camber/odom and publishing /camber/cmd/pose. It flies the oval
gate course by publishing a setpoint just past the next gate with a velocity
through it -- position alone would only say "go here and stop".

Replace `setpoint()` with your own logic, or publish /camber/cmd/wrench to fly
the moments yourself, or /camber/cmd/motors to fly the props.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Transform, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

from camber.sim.course import Course


class GatePilot(Node):
    def __init__(self) -> None:
        super().__init__("gate_pilot")
        self.speed = float(self.declare_parameter("speed", 8.0).value)
        self.laps = int(self.declare_parameter("laps", 2).value)
        self.course = Course.oval()
        self.gates = [g.centre for g in self.course.gates]
        self.i, self.lap, self.p = 0, 0, None
        self.pub = self.create_publisher(MultiDOFJointTrajectoryPoint,
                                         "/camber/cmd/setpoint", 10)
        self.create_subscription(Odometry, "/camber/odom", self._odom, 10)
        self.create_timer(0.02, self._tick)                    # 50 Hz

    def _odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.position
        self.p = np.array([q.x, q.y, q.z])

    def setpoint(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Where to be, how fast to be going, and which way to face."""
        gate = self.course.gates[self.i]
        if gate.crossed(self.p):
            self.i = (self.i + 1) % len(self.course.gates)
            if self.i == 0:
                self.lap += 1
                self.get_logger().info(f"lap {self.lap} complete")
            gate = self.course.gates[self.i]
        d = gate.centre - self.p
        dist = float(np.linalg.norm(d))
        heading = d / max(dist, 1e-6)
        blend = np.clip(1.0 - dist / 4.0, 0.0, 1.0)      # square up at the gate
        direction = (1 - blend) * heading + blend * gate.normal
        direction /= np.linalg.norm(direction)
        return (gate.centre + 0.5 * gate.normal, self.speed * direction,
                float(np.arctan2(direction[1], direction[0])))

    def _tick(self) -> None:
        if self.p is None:
            return
        if self.lap >= self.laps:
            where, vel, yaw = self.course.finish, np.zeros(3), 0.0
        else:
            where, vel, yaw = self.setpoint()
        sp = MultiDOFJointTrajectoryPoint()
        tr = Transform()
        tr.translation.x, tr.translation.y, tr.translation.z = map(float, where)
        tr.rotation.w, tr.rotation.z = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.linear.z = map(float, vel)
        sp.transforms, sp.velocities = [tr], [tw]
        self.pub.publish(sp)


def main() -> None:
    rclpy.init()
    node = GatePilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
