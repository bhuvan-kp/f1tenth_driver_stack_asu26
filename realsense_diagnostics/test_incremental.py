import time
import pyrealsense2 as rs


def run_test(name, configure_fn, seconds=4):
    print(f"\n=== {name} ===")
    counts = {}

    def callback(frame):
        if frame.is_frameset():
            for f in frame.as_frameset():
                _tally(f, counts)
        else:
            _tally(frame, counts)

    pipeline = rs.pipeline()
    config = rs.config()
    configure_fn(config)

    try:
        profile = pipeline.start(config, callback)
    except Exception as e:
        print(f"  pipeline.start() FAILED: {e}")
        return

    time.sleep(seconds)
    pipeline.stop()
    print(f"  counts over {seconds}s: {counts if counts else '(nothing at all)'}")


def _tally(f, counts):
    if f.is_video_frame():
        profile = f.get_profile()
        st = profile.stream_type()
        key = str(st)
        if st == rs.stream.infrared:
            key = f"infra{profile.stream_index()}"
        counts[key] = counts.get(key, 0) + 1
    elif f.is_motion_frame():
        st = f.get_profile().stream_type()
        key = 'gyro' if st == rs.stream.gyro else 'accel'
        counts[key] = counts.get(key, 0) + 1


run_test(
    "Motion only (accel 200 + gyro 400)",
    lambda c: (
        c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200),
        c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400),
    ),
)

run_test(
    "Infra1 only, 640x480@30",
    lambda c: c.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30),
)

run_test(
    "Color only, 640x480@30 bgr8",
    lambda c: c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30),
)

run_test(
    "Infra1 + Infra2, 640x480@30",
    lambda c: (
        c.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30),
        c.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30),
    ),
)

run_test(
    "Infra1 only, low-res 480x270@15",
    lambda c: c.enable_stream(rs.stream.infrared, 1, 480, 270, rs.format.y8, 15),
)

print("\nDone. Compare which configs actually produced nonzero counts.")
