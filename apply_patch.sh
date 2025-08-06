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
      <pose>0.15 0 -0.05 0 0 0</pose> <!-- Positioned 0.05m below base_link center -->
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
            <size>0.15 0.15 0.08</size> <!-- Approximate box shape for payload -->
          </box>
        </geometry>
      </collision>
      <visual name="payload_visual">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>0.15 0.15 0.08</size>
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
# Scale motor poses for ~452mm wheelbase (original arm ~0.255m, scale ~1.25)
sed -i 's/<pose>0.13 -0.22 0.023 0 0 0<\/pose>/<pose>0.1625 -0.275 0.023 0 0 0<\/pose>/g' "$PX4_JINJA_PATH"
echo "Verifying rotor_0 pose:"
grep "<pose>0.1625 -0.275 0.023 0 0 0<\/pose>" "$PX4_JINJA_PATH"
sed -i 's/<pose>-0.13 0.2 0.023 0 0 0<\/pose>/<pose>-0.1625 0.25 0.023 0 0 0<\/pose>/g' "$PX4_JINJA_PATH"
echo "Verifying rotor_1 pose:"
grep "<pose>-0.1625 0.25 0.023 0 0 0<\/pose>" "$PX4_JINJA_PATH"
sed -i 's/<pose>0.13 0.22 0.023 0 0 0<\/pose>/<pose>0.1625 0.275 0.023 0 0 0<\/pose>/g' "$PX4_JINJA_PATH"
echo "Verifying rotor_2 pose:"
grep "<pose>0.1625 0.275 0.023 0 0 0<\/pose>" "$PX4_JINJA_PATH"
sed -i 's/<pose>-0.13 -0.2 0.023 0 0 0<\/pose>/<pose>-0.1625 -0.25 0.023 0 0 0<\/pose>/g' "$PX4_JINJA_PATH"
echo "Verifying rotor_3 pose:"
grep "<pose>-0.1625 -0.25 0.023 0 0 0<\/pose>" "$PX4_JINJA_PATH"
# Update prop radius to 0.127m (10" diameter props, radius 0.127m)
sed -i 's/<radius>0.128<\/radius>/<radius>0.127<\/radius>/g' "$PX4_JINJA_PATH"
echo "Verifying prop radius:"
grep "<radius>0.127<\/radius>" "$PX4_JINJA_PATH"
# Scale thrust_factor (motorConstant) ~10x for stronger motors (original 5.84e-06 -> 5.84e-05)
sed -i 's/5.84e-06/5.84e-05/g' "$PX4_JINJA_PATH"
echo "Verifying motorConstant:"
grep "5.84e-05" "$PX4_JINJA_PATH"
# Scale torque_factor (momentConstant) ~10x (original 0.06 -> 0.6)
sed -i 's/0.06/0.6/g' "$PX4_JINJA_PATH"
echo "Verifying momentConstant:"
grep "0.6" "$PX4_JINJA_PATH" | grep momentConstant
# Update maxRotVelocity to 900 (lower KV)
sed -i 's/1100/900/g' "$PX4_JINJA_PATH"
echo "Verifying maxRotVelocity:"
grep "900" "$PX4_JINJA_PATH" | grep maxRotVelocity
# Update base_link mass to ~0.826kg (CX10 dry weight)
sed -i 's/<mass>1.5<\/mass>/<mass>0.826<\/mass>/g' "$PX4_JINJA_PATH"
echo "Verifying base_link mass:"
grep "<mass>0.826<\/mass>" "$PX4_JINJA_PATH"
# Scale base_link inertia ~1.7x (mass_ratio * scale^2 ≈ 0.55 * 1.56 ≈ 0.86, but ~1.7 for stability)
sed -i 's/<ixx>0.029125<\/ixx>/<ixx>0.05<\/ixx>/g' "$PX4_JINJA_PATH"
sed -i 's/<iyy>0.029125<\/iyy>/<iyy>0.05<\/iyy>/g' "$PX4_JINJA_PATH"
sed -i 's/<izz>0.055225<\/izz>/<izz>0.09<\/izz>/g' "$PX4_JINJA_PATH"
echo "Verifying base_link inertia:"
grep "<ixx>0.05<\/ixx>" "$PX4_JINJA_PATH"
grep "<iyy>0.05<\/iyy>" "$PX4_JINJA_PATH"
grep "<izz>0.09<\/izz>" "$PX4_JINJA_PATH"
# Scale base_link collision box ~1.25x
sed -i 's/<size>0.47 0.47 0.11<\/size>/<size>0.5875 0.5875 0.11<\/size>/g' "$PX4_JINJA_PATH"
echo "Verifying base_link collision size:"
grep "<size>0.5875 0.5875 0.11<\/size>" "$PX4_JINJA_PATH"
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
echo "param set MPC_MAN_TILT_MAX 20" >> "$AIRFRAME_PATH"  # Cine stability
echo "param set MC_YAW_P 0.05" >> "$AIRFRAME_PATH"
echo "param set MC_YAWRATE_MAX 90" >> "$AIRFRAME_PATH"
echo "param set MPC_Z_VEL_MAX_UP 3.0" >> "$AIRFRAME_PATH"  # Loaded climb
echo "param set MPC_Z_VEL_MAX_DN 2.0" >> "$AIRFRAME_PATH"
echo "param set MPC_XY_CRUISE 10.0" >> "$AIRFRAME_PATH"  # ~36km/h cruise
echo "param set MC_AIRMODE 1" >> "$AIRFRAME_PATH"  # Cinewhoop style
# Verify airframe parameter additions
echo "Verifying airframe parameters:"
tail -n 19 "$AIRFRAME_PATH"  # Shows added params
echo " Successfully Patched !"
