rm -rf build/ install/ log/
colcon build --symlink-install # --cmake-args -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python3" -DPython3_FIND_VIRTUALENV=ONLYecho "$CONDA_PREFIX"

# sed -i '1c\#!/home/robolab/miniconda3/envs/oren_env/bin/python' ~/oren/install/oren_ros/lib/oren_ros/sdf_query_node 
# sed -i '1c\#!/home/robolab/miniconda3/envs/oren_env/bin/python' ~/oren/install/oren_ros/lib/oren_ros/mapping_node

source install/setup.bash
