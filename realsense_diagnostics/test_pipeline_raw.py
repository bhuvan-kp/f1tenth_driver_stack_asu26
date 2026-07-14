import time
import pyrealsense2 as rs

counts = {'color': 0, 'infra1': 0, 'infra2': 0, 'accel': 0, 'gyro': 0}

def callback(frame):
    if frame.is_frameset():
        for f in frame.as_frameset():
            _tally(f)
    else:
        _tally(frame)

def _tally(f):
    if f.is_video_frame():
        profile = f.get_profile()
        st = profile.stream_type()
        if st == rs.stream.color:
            counts['color'] += 1
        elif st == rs.stream.infrared:
            idx = profile.stream_index()
            counts['infra1' if idx == 1 else 'infra2'] += 1
    elif f.is_motion_frame():
        st = f.get_profile().stream_type()
        counts['gyro' if st == rs.stream.gyro else 'accel'] += 1

print("Enumerating devices...")
ctx = rs.context()
devices = ctx.query_devices()
print(f"Found {len(devices)} device(s)")
for d in devices:
    print(" -", d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))

print("Starting pipeline (color+infra1+infra2 640x480@30, accel 200, gyro 400)...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)

profile = pipeline.start(config, callback)
print("Pipeline started. Device:", profile.get_device().get_info(rs.camera_info.name))

print("Sleeping 5 seconds while callback runs in the background...")
time.sleep(5)

pipeline.stop()
print("Pipeline stopped.")
print("Frame counts over 5 seconds:", counts)

if sum(counts.values()) == 0:
    print("\n>>> ZERO frames received at the librealsense layer. This is NOT a ROS issue.")
else:
    print("\n>>> Frames ARE arriving at the librealsense layer. The problem is elsewhere (ROS/DDS side).")
