#!/bin/bash
set -e
# Path to PX4's Iris SDF Jinja template
PX4_JINJA_PATH="/home/px4user/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf.jinja"
# Path to IRIS airframe file for persistent parameters
AIRFRAME_PATH="/home/px4user/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/10015_gazebo-classic_iris"
# Define the sensor XML to insert inside base_link
SENSOR_XML='
      <!-- Forward-looking camera pitched up 30 degrees -->
      <sensor name="forward_camera" type="camera">
        <pose>0.1 0 0.1 0 -0.5236 0</pose> <!-- Position: 0.1m forward, 0.1m up; Orientation: 0 roll, -30 deg pitch (upward tilt), 0 yaw -->
        <camera>
          <horizontal_fov>1.3962634</horizontal_fov> <!-- ~80 degrees FOV -->
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.01</near>
            <far>100</far>
          </clip>
          <distortion>
            <k1>0.0</k1>
            <k2>0.0</k2>
            <k3>0.0</k3>
            <p1>0.0</p1>
            <p2>0.0</p2>
          </distortion>
        </camera>
        <always_on>true</always_on>
        <update_rate>30.0</update_rate>
        <plugin name="forward_camera_plugin" filename="libgazebo_ros_camera.so">
          <robotNamespace>/</robotNamespace>
          <cameraName>forward_camera</cameraName>
          <imageTopicName>image_raw</imageTopicName>
          <cameraInfoTopicName>camera_info</cameraInfoTopicName>
          <frameName>forward_camera_optical</frameName>
          <hackBaseline>0.0</hackBaseline>
          <distortionK1>0.0</distortionK1>
          <distortionK2>0.0</distortionK2>
          <distortionK3>0.0</distortionK3>
          <distortionT1>0.0</distortionT1>
          <distortionT2>0.0</distortionT2>
        </plugin>
      </sensor>
'
# Define the frame XML to insert after base_link's </link>
FRAME_XML='
    <!-- Optical frame for the camera -->
    <link name="forward_camera_optical">
      <pose>0.1 0 0.1 0 -0.5236 0</pose>
      <inertial>
        <mass>0.01</mass> <!-- Small mass for camera; adjust if needed -->
        <inertia>
          <ixx>0.0001</ixx> <!-- Symmetric inertia for stability -->
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.0001</iyy>
          <iyz>0.0</iyz>
          <izz>0.0001</izz>
        </inertia>
      </inertial>
    </link>
    <joint name="forward_camera_joint" type="fixed">
      <parent>base_link</parent>
      <child>forward_camera_optical</child>
    </joint>
'
# Define the payload XML to insert after the frame
PAYLOAD_XML='
    <!-- 2.5kg payload for Chimera CX10 simulation -->
    <link name="payload">
      <pose>0.15 0 -0.05 0 0 0</pose> <!-- Positioned 0.2m below base_link center -->
      <inertial>
        <mass>2.5</mass>
        <inertia>
          <ixx>0.0208</ixx>
          <iyy>0.0208</iyy>
          <izz>0.0375</izz>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyz>0.0</iyz>
        </inertia>
      </inertial>
      <collision name="payload_collision">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>0.3 0.3 0.1</size> <!-- Approximate box shape for payload -->
          </box>
        </geometry>
      </collision>
      <visual name="payload_visual">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>0.3 0.3 0.1</size>
          </box>
        </geometry>
      </visual>
    </link>
    <joint name="payload_joint" type="fixed">
      <parent>base_link</parent>
      <child>payload</child>
    </joint>
'
# Use awk to insert sensor inside base_link before </link>, and frame + payload after </link>
awk -v sensor="${SENSOR_XML}" -v frame="${FRAME_XML}" -v payload="${PAYLOAD_XML}" '
/<link name='"'"'base_link'"'"'>/ { in_base = 1 }
/<\/link>/ && in_base {
  printf "%s", sensor;
  print $0;
  printf "%s%s", frame, payload;
  in_base = 0;
  next
}
{ print }
' "$PX4_JINJA_PATH" > temp.sdf.jinja
echo "Updated Jinja template written to temp.sdf.jinja."
# Optionally replace the original file (uncomment if needed, after verifying temp.sdf.jinja)
mv temp.sdf.jinja "$PX4_JINJA_PATH"
# Scale motor poses for ~452mm wheelbase (original ~318mm diagonal, scale ~1.42)
sed -i 's/<pose>0.225 0.225 -0.063 0 0 0<\/pose>/<pose>0.3195 0.3195 -0.063 0 0 0<\/pose>/g' "$PX4_JINJA_PATH"
sed -i 's/<pose>0.225 -0.225 -0.063 0 0 -1.570796<\/pose>/<pose>0.3195 -0.3195 -0.063 0 0 -1.570796<\/pose>/g' "$PX4_JINJA_PATH"
sed -i 's/<pose>-0.225 -0.225 -0.063 0 0 3.141593<\/pose>/<pose>-0.3195 -0.3195 -0.063 0 0 3.141593<\/pose>/g' "$PX4_JINJA_PATH"
sed -i 's/<pose>-0.225 0.225 -0.063 0 0 1.570796<\/pose>/<pose>-0.3195 0.3195 -0.063 0 0 1.570796<\/pose>/g' "$PX4_JINJA_PATH"
# Update prop radius to 0.127m (10" diameter props)
sed -i 's/<radius>0.127<\/radius>/<radius>0.127<\/radius>/g' "$PX4_JINJA_PATH"  # Already matches, no change needed
# Scale thrust_factor ~3x for stronger motors (original 0.166 -> 0.5)
sed -i 's/<thrust_factor>0.166<\/thrust_factor>/<thrust_factor>0.5<\/thrust_factor>/g' "$PX4_JINJA_PATH"
# Scale torque_factor ~3x (original 0.002 -> 0.006)
sed -i 's/<torque_factor>0.002<\/torque_factor>/<torque_factor>0.006<\/torque_factor>/g' "$PX4_JINJA_PATH"
# Update base_link mass to ~0.826kg (CX10 dry weight)
sed -i 's/<mass>0.2<\/mass>/<mass>0.826<\/mass>/g' "$PX4_JINJA_PATH"
# Scale base_link inertia ~8x (scale^2 * mass_ratio ≈ 2*4=8)
sed -i 's/<ixx>0.006<\/ixx>/<ixx>0.048<\/ixx>/g' "$PX4_JINJA_PATH"
sed -i 's/<iyy>0.006<\/iyy>/<iyy>0.048<\/iyy>/g' "$PX4_JINJA_PATH"
sed -i 's/<izz>0.01<\/izz>/<izz>0.08<\/izz>/g' "$PX4_JINJA_PATH"
# Append persistent parameter changes to airframe file for closer dynamics
echo "param set COM_DISARM_LAND 0" >> "$AIRFRAME_PATH"
echo "param set MPC_TKO_SPEED 5.0" >> "$AIRFRAME_PATH"
echo "param set NAV_RCL_ACT 0" >> "$AIRFRAME_PATH"
echo "param set COM_RC_IN_MODE 4" >> "$AIRFRAME_PATH"
echo "param set MPC_XY_VEL_MAX 38.9" >> "$AIRFRAME_PATH"  # ~140km/h max
echo "param set BAT_N_CELLS 6" >> "$AIRFRAME_PATH"
echo "param set MPC_THR_HOVER 0.6" >> "$AIRFRAME_PATH"
echo "param set MPC_THR_MAX 1.0" >> "$AIRFRAME_PATH"
echo "param set MPC_THR_MIN 0.15" >> "$AIRFRAME_PATH"
echo "param set MC_PITCHRATE_P 0.08" >> "$AIRFRAME_PATH"
echo "param set MC_ROLLRATE_P 0.08" >> "$AIRFRAME_PATH"
echo "param set MC_YAWRATE_P 0.05" >> "$AIRFRAME_PATH"  # Lower for larger frame
echo " Successfully Patched !"
