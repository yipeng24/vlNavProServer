# cmd_vel mux

## About

This repository is an extention of the original [repository](https://github.com/kobuki-base/cmd_vel_mux) that combines all active public forks.

A ROS 2 package for selecting among a number of incoming geometry_msg/msg/Twist messages, choosing the highest
priority one to republish on the output topic.  It will automatically dislodge streams that are lower
priority or that stop publishing for any reason.  The stream currently in use is published on the
"active" topic.

<br>

## Merged forks

|Version | Fork                                                                                          | Merged | Tested |
|:---:	 |---	                                                                                            |:---:   |:---:	  |
|0.1.0 	 | [indro-robotics/cmd_vel_mux](https://github.com/indro-robotics/cmd_vel_mux)  	                |&#9745; |&#9745; |
|0.1.0 	 | [indro-robotics/cmd_vel_mux](https://github.com/indro-robotics/cmd_vel_mux)  	                |&#9745; |&#9745; |
|0.1.0 	 | [slamcore/cmd_vel_mux](https://github.com/slamcore/cmd_vel_mux)  	                            |&#9745; |&#9745; |


## Published Topics

|Topic      | Type                      | Description |
|:---	      |:---	                      |:---         |
|`/active`	 | `std_msgs/msg/String`     | A latched topic.  Publishes the "name" field of the currently active `geometry_msgs/msg/Twist` stream (see Parameters below), or "idle" if nothing is being a published.|
|`/mux/output/cmd_vel` | `geometry_msgs/msg/Twist` | The current `geometry_msgs/msg/Twist` message.  The values from the current highest-priority `geometry_msgs/msg/Twist` are republished here without change.|


## Subscribed Topics
* `/any` (`geometry_msgs/msg/Twist`) - For each named input (see Parameters below), a topic of type `geometry_msgs/msg/Twist` is subscribed to based on the `topic` parameter.  The data from the highest-priority of these inputs is republished on the output `/cmd_vel`.


## Parameters

The cmd_vel_mux node uses a dictionary to hold the subscribers, with `name` being the key of each element. Following parameters corresponds to each of the input `geometry_msgs/msg/Twist` topics. 

|Parameter    | Type          | Description | Example |
|:---	        |:---	          |:---         | :---    |
|`name`       | string        | This is what will be published on the `active` topic when the currently active publisher changes. | `default_input` |
|`topic`      | string        | This is what will be subscribed to.| `mux/input/default` |
|`timeout`    | double        | If no data is received on the input topic for this amount of time, the input will be automatically disabled.| `0.1` |
|`priority`   | integer       | The higher the number, the higher the priority.  Higher priority topics will dislodge lower priority topics and start publishing to the output topic automatically.| `0` |
|`short_desc` | string        | Short Description of the input and this is informational only..| `The default cmd_vel, controllers unaware that we are multiplexing cmd_vel should come here` |

See config/cmd_vel_mux_params.yaml for an example of how to set these up.

## Changing values
The dictionary translates to parameters in the form of `subscribers.` + name + `topic/timeout/priority/short_desc`.
It can be changed on the fly using `ros2 param set` command, followed with the parameter in the form mentioned above and the value.

## Adding subscriber
To add a new subscriber, call the service `/cmd_vel_mux/set_parameters_atomically` with a `rcl_interfaces/srv/SetParametersAtomically` message with a parameters value equal to an array of 4 `rcl_interfaces/msg/Parameter` messages, containing the new subscriber topic, timeout, priority and short_desc. All 4 need to be present for a new subscriber to be added.

**Ex.1** for testing:
```
ros2 service call /cmd_vel_mux/set_parameters_atomically rcl_interfaces/srv/SetParametersAtomically "{parameters: [{name : subscribers.new_input.topic, value: {type: 4, string_value: /input/new_input}}, {name : subscribers.new_input.priority, value: {type: 2, integer_value: 20}}, {name : subscribers.new_input.timeout, value: {type: 3, double_value: 0.2}}, {name : subscribers.new_input.short_desc, value: {type: 4, string_value: 'test input'}}]}"
```
sets *new_input* subscriber listening at topic `/input/new_input`

## Deleting subscriber
To delete a subscriber, call the service `/cmd_vel_mux/set_parameters_atomically` with a `rcl_interfaces/srv/SetParametersAtomically` message with a parameters value equal to an array of 4 `rcl_interfaces/msg/Parameter` messages, containing the new subscriber topic,timeout,priority and short_desc with the `type` key inside the `rcl/parameter/ParameterValue` being set to 0 (PARAMETER_NOT_SET).

**Ex.2** for testing:
```
ros2 service call /cmd_vel_mux/set_parameters_atomically rcl_interfaces/srv/SetParametersAtomically "{parameters: [{name : subscribers.new_input.topic, value: {type: 0}}, {name : subscribers.new_input.priority, value: {type: 0}}, {name : subscribers.new_input.timeout, value: {type: 0}}, {name : subscribers.new_input.short_desc, value: {type: 0}}]}"
```
 deletes *new_input* subscriber added at ex.1
