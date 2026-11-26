
#!/usr/bin/env python3

import rclpy
import numpy as np

from rclpy.node import Node
from grid_map_msgs.msg import GridMap


class SdfGridChecker(Node):

    def __init__(self):
        super().__init__("sdf_grid_checker")

        self.subscription = self.create_subscription(
            GridMap,
            "/sdf_grid_map",
            self.callback,
            10,
        )

        self.received = False

    def callback(self, msg):

        if self.received:
            return

        self.received = True

        print("\n========== GridMap Info ==========")

        print(f"Frame       : {msg.header.frame_id}")
        print(f"Resolution  : {msg.info.resolution}")
        print(f"Length X    : {msg.info.length_x}")
        print(f"Length Y    : {msg.info.length_y}")
        print(f"Center X    : {msg.info.pose.position.x}")
        print(f"Center Y    : {msg.info.pose.position.y}")

        for layer_name, array_msg in zip(
            msg.layers,
            msg.data,
        ):

            values = np.asarray(
                array_msg.data,
                dtype=np.float64,
            )

            valid = np.isfinite(values)
            valid_values = values[valid]

            print(f"\n========== {layer_name} ==========")

            print(f"Total cells : {len(values)}")
            print(f"Valid cells : {len(valid_values)}")

            if len(valid_values) == 0:
                print("No valid values")
                continue

            print(f"Min         : {np.min(valid_values):.6f}")
            print(f"Max         : {np.max(valid_values):.6f}")
            print(f"Mean        : {np.mean(valid_values):.6f}")
            print(f"Std         : {np.std(valid_values):.6f}")

            print(
                f"Unique      : "
                f"{len(np.unique(valid_values))}"
            )

            print(
                f"Zero count  : "
                f"{np.count_nonzero(valid_values == 0)}"
            )

        print("\n==================================")

        rclpy.shutdown()


def main():

    rclpy.init()

    node = SdfGridChecker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
