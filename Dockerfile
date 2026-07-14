FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# All apt dependencies from f1tenth_driver_stack, ros2_joy_driver, AND the
# RealSense build, combined and deduplicated. Installed once at build time
# instead of on every container start.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ackermann-msgs \
    ros-humble-diagnostic-updater \
    ros-humble-serial-driver \
    ros-humble-asio-cmake-module \
    ros-humble-joy \
    ros-humble-joy-teleop \
    ros-humble-nav-msgs \
    ros-humble-tf-transformations \
    python3-scipy \
    python3-colcon-common-extensions \
    libsdl2-dev \
    git cmake build-essential pkg-config \
    libssl-dev libusb-1.0-0-dev libudev-dev \
    libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    python3-pip python3-dev pybind11-dev \
    ros-humble-cv-bridge ros-humble-sensor-msgs ros-humble-image-transport \
    udev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch safetensors
RUN pip3 install --no-cache-dir --force-reinstall "setuptools==59.6.0"

# ---------------------------------------------------------------------------
# Build librealsense2 from source, using the NATIVE Linux kernel backend.
# This requires the Jetson HOST kernel to already have the RealSense-patched
# uvcvideo/hid_sensor_hub/iio kernel modules installed (one-time host setup,
# see: jetsonhacks/jetson-orin-librealsense) -- NOT something this Dockerfile
# can do, since kernel modules load into the host kernel, not the container.
# ---------------------------------------------------------------------------
ARG LIBREALSENSE_VERSION=v2.55.1
RUN git clone --recurse-submodules --shallow-submodules --depth 1 --branch ${LIBREALSENSE_VERSION} \
    https://github.com/IntelRealSense/librealsense.git /tmp/librealsense \
    && mkdir -p /tmp/librealsense/build \
    && cd /tmp/librealsense/build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DFORCE_RSUSB_BACKEND=false \
        -DBUILD_PYTHON_BINDINGS=true \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DBUILD_EXAMPLES=false \
        -DBUILD_GRAPHICAL_EXAMPLES=false \
        -DBUILD_WITH_CUDA=false \
    2>&1 | tee cmake_configure.log \
    && ! grep -qi "pybind11.*not found" cmake_configure.log \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# Belt-and-braces cleanup: remove any incomplete pyrealsense2 stub package,
# find wherever the real compiled extension landed, and copy it flat into
# /usr/local/lib so PYTHONPATH picks it up unambiguously. Fails the BUILD
# immediately if the bindings are broken, instead of failing at runtime.
RUN rm -rf /usr/lib/python3/dist-packages/pyrealsense2 \
    && rm -rf $(python3 -c "import sysconfig; print(sysconfig.get_path('platlib'))")/pyrealsense2 \
    && find / -xdev -name "pyrealsense2*.so*" -exec cp -v {} /usr/local/lib/ \; \
    && ldconfig \
    && rm -rf /tmp/librealsense \
    && PYTHONPATH=/usr/local/lib python3 -c "import pyrealsense2 as rs; print('pyrealsense2 OK:', rs.__file__); print(rs.pipeline)"

WORKDIR /f1tenth_ws

# NOTE: package source is intentionally NOT copied here. It is bind-mounted
# from ./src at container run time (see docker-compose.yml), so you can
# edit any package's source on the host and just re-run colcon build
# inside the container without rebuilding the image.

ENV SDL_JOYSTICK_DEVICE=/dev/input/js0
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ENV PYTHONPATH=/usr/local/lib:${PYTHONPATH}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash", "-c", "colcon build --symlink-install && source install/setup.bash && ros2 launch f1tenth_bringup bringup_launch.py"]
