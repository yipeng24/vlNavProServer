#---------------------------------------------------------------------------------------------------------------------------
#----
#----   Start base image
#----
#---------------------------------------------------------------------------------------------------------------------------

FROM ros:humble-ros-base-jammy as base

## Parameters
ENV WORKSPACE_ROOT=/kobuki

#############################################################################################################################
#####
#####   Install Dependencies
#####
#############################################################################################################################

WORKDIR /

RUN apt-get update -y
RUN apt-get install -y --no-install-recommends  ros-dev-tools \
                                                ros-$ROS_DISTRO-angles \
                                                ros-$ROS_DISTRO-diagnostics \
                                                ros-$ROS_DISTRO-joint-state-publisher \
                                                ros-$ROS_DISTRO-ros-testing \
                                                ros-$ROS_DISTRO-action-msgs \
                                                ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
                                                udev

RUN apt-get clean

#############################################################################################################################
#####
#####   Install Kobuki packages
#####
#############################################################################################################################

WORKDIR ${WORKSPACE_ROOT}/src

RUN git clone --recurse-submodules https://github.com/AIResearchLab/kobuki.git
RUN git clone --recurse-submodules https://github.com/AIResearchLab/kobuki_dependencies.git 

RUN rm /etc/ros/rosdep/sources.list.d/20-default.list

RUN rosdep init && rosdep update && rosdep install --from-paths ${WORKSPACE_ROOT}/src -y --ignore-src


#############################################################################################################################
#####
#####   Build Kobuki packages
#####
#############################################################################################################################

WORKDIR ${WORKSPACE_ROOT}

RUN . /opt/ros/humble/setup.sh && colcon build
WORKDIR /


#############################################################################################################################
#####
#####   Remove workspace source and build files that are not relevent to running the system
#####
#############################################################################################################################

RUN rm -rf ${WORKSPACE_ROOT}/src
RUN rm -rf ${WORKSPACE_ROOT}/log
RUN rm -rf ${WORKSPACE_ROOT}/build
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /tmp/*

RUN apt-get clean


#---------------------------------------------------------------------------------------------------------------------------
#----
#----   Start final release image
#----
#---------------------------------------------------------------------------------------------------------------------------


FROM ros:humble-ros-base-jammy as final

## Parameters
ENV WORKSPACE_ROOT=/kobuki
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

WORKDIR /

COPY --from=base / /

RUN wget https://raw.githubusercontent.com/AIResearchLab/kobuki_ftdi/devel/60-kobuki.rules && \
    cp 60-kobuki.rules /etc/udev/rules.d

COPY docker/workspace_entrypoint.sh /workspace_entrypoint.sh

RUN chmod +x /workspace_entrypoint.sh

ENTRYPOINT [ "/workspace_entrypoint.sh" ]

WORKDIR ${WORKSPACE_ROOT}
