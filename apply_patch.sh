#!/bin/bash
set -e

# Path to PX4's Iris SDF Jinja template
PX4_JINJA_PATH="/home/px4user/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf.jinja"

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
        <plugin name="forward_camera_plugin" filename="libgazebo_camera_plugin.so">
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
    </link>
    <joint name="forward_camera_joint" type="fixed">
      <parent>base_link</parent>
      <child>forward_camera_optical</child>
    </joint>
'

# Use awk to insert sensor inside base_link before </link>, and frame after </link>
awk -v sensor="${SENSOR_XML}" -v frame="${FRAME_XML}" '
/<link name='"'"'base_link'"'"'>/ { in_base = 1 }
/<\/link>/ && in_base { 
  printf "%s", sensor; 
  print $0; 
  printf "%s", frame; 
  in_base = 0; 
  next 
}
{ print }
' "$PX4_JINJA_PATH" > temp.sdf.jinja
echo "Updated Jinja template written to temp.sdf.jinja. 
# Optionally replace the original file (uncomment if needed, after verifying temp.sdf.jinja)
mv temp.sdf.jinja "$PX4_JINJA_PATH"

echo " Successfully Patched !"
