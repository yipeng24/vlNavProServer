Quik start

`ros2 launch rosbridge_server rosbridge_websocket_launch.xml `

`python3 -m http.server 8080`

`ros2 launch tasks bringup.launch.py`

`ros2 launch tasks nav2_waypoint.launch.py`

`ros2 launch container container.launch.py`

`ros2 run container bridge_node`

`ros2 run container waypoint_client`
