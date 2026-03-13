cmake_minimum_required(VERSION 3.8)
project(kobuki_nav)

# 找到 ament_cmake
find_package(ament_cmake REQUIRED)

# 安装 launch 和 config 目录
install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# 结束
ament_package()
