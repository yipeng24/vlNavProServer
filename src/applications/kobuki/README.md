# Kobuki ROS2 package

This repository and the [dependencies repository](https://github.com/AIResearchLab/kobuki_dependencies.git) focuses on merging all the public forks available on the original [Kobuki Base](https://github.com/kobuki-base) repositories and extending the work towards an complete ROS2 package for kobuki based turtlebot robots. Additionally a Dockerfile and prebuilt containers for kobuki system has been provided that supports both ARM64 and AMD64 systems. 

## Docker Usage

### Docker Compose based use

Add the following snippet under `services` to any compose.yaml file to add this container.

```bash
  kobuki:
    image: ghcr.io/airesearchlab/kobuki:humble
    command: ros2 launch kobuki kobuki.launch.py
    restart: unless-stopped
    privileged: true
    network_mode: host
    volumes:
      - /dev:/dev
```

### Setup for pulling container from ghcr.io and running

Clone this reposiotory

```bash
git clone https://github.com/AIResearchLab/kobuki.git
```

Pull the Docker image and start compose (No need to run `docker compose build`)
```bash
cd kobuki/docker
docker compose pull
docker compose up
```

### Setup for building the container on device running

Clone this reposiotory

```bash
git clone https://github.com/AIResearchLab/kobuki.git
```

Build the Docker image and start compose
```bash
cd kobuki/docker
docker compose -f compose-build.yaml build
docker compose -f compose-build.yaml up
```

## Direct Installation


1. Install the following binary dependencies

    ```bash
    sudo apt install ros-humble-angles ros-humble-diagnostics ros-humble-joint-state-publisher ros-humble-ros-testing
    ```

2. Optional Dependencies for using kobuki_keyop [Issue #21](https://github.com/ros2/teleop_twist_keyboard/issues/21)
    ```bash
    sudo apt install xterm
    ```

3. Optional Dependencies for using kobuki_joyop 
    ```bash
    sudo apt install ros-humble-teleop-twist-joy ros-humble-joy
    ```

3. Clone main repository and dependencies
    ```bash
    git clone --recurse-submodules https://github.com/AIResearchLab/kobuki.git
    git clone --recurse-submodules https://github.com/AIResearchLab/kobuki_dependencies.git
    ```

4. Build the system with following command. 
    ```bash
    colcon build
    ```

5. Set udev rule for kobuki using following commands and unplug and replug the usb cable.
    ```bash
    wget https://raw.githubusercontent.com/kobuki-base/kobuki_ftdi/devel/60-kobuki.rules
    sudo cp 60-kobuki.rules /etc/udev/rules.d

    sudo service udev reload
    sudo service udev restart
    ```

## Non-ROS usage

### Check connectivity

Run following commands in the workspace.
```bash
source ./install/setup.bash
kobuki-version-info
```

Ouput should be like,
```bash
Version Info:
Hardware Version: 1.0.4
Firmware Version: 1.2.0
Software Version: 1.1.0
Unique Device ID: 97713968-842422349-1361404194
```

### Kobuki Teleop keyboard

Run following commands in the workspace and follow onscreen instructions
```bash
source ./install/setup.bash
kobuki-simple-keyop
```

## ROS usage

### Kobuki system with navigation stack (joysitck backup) start

```bash
source ./install/setup.bash
ros2 launch kobuki kobuki_navigation.launch.py
```

### Kobuki system with joystick start

```bash
source ./install/setup.bash
ros2 launch kobuki kobuki_joystick.launch.py
```

### Kobuki KeyOp control

Run following commands in the workspace to start kobuki system.
```bash
source ./install/setup.bash
ros2 launch kobuki_node kobuki_node.launch.py
```

Run following commands in the workspace to use kobuki keyboard control.
```bash
source ./install/setup.bash
ros2 run kobuki_keyop kobuki_keyop_node
```
