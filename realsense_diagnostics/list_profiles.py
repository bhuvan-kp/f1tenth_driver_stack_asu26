import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("No device found")
else:
    for dev in devices:
        print("Device:", dev.get_info(rs.camera_info.name))
        for sensor in dev.query_sensors():
            print(" Sensor:", sensor.get_info(rs.camera_info.name))
            for p in sensor.get_stream_profiles():
                vp = p.as_video_stream_profile() if p.is_video_stream_profile() else None
                mp = p.as_motion_stream_profile() if p.is_motion_stream_profile() else None
                if vp:
                    print(f"   VIDEO  {p.stream_name():10s} idx={p.stream_index()} {vp.width()}x{vp.height()} {p.format()} @ {p.fps()}fps")
                elif mp:
                    print(f"   MOTION {p.stream_name():10s} @ {p.fps()}fps  format={p.format()}")
