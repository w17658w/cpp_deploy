# test/CMakeLists.txt — 硬件通信测试程序编译
#
# 在 deploy/ 目录下：
#   cmake -B build_test -DCMAKE_BUILD_TYPE=Release
#   cmake --build build_test -j
#
# 产物在 build_test/test_*/
#   test_imu        — WT901 IMU 串口测试
#   test_actuator  — Arduino Mega2560 + 12 路舵机测试
#   test_gpio       — 12 路触地 GPIO 测试

set(TEST_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# 公共源码（被多个测试程序复用）
set(COMMON_IO
    ${TEST_SRC_DIR}/io/serial.cpp
)

# ---------- test_imu: 只依赖 serial + wt901 ----------
add_executable(test_imu
    test_imu.cpp
    ${COMMON_IO}
    ${TEST_SRC_DIR}/perception/wt901.cpp
)
target_include_directories(test_imu PRIVATE
    ${TEST_SRC_DIR}/io
    ${TEST_SRC_DIR}/perception
)

# ---------- test_actuator: 依赖 serial + protocol + actuator ----------
add_executable(test_actuator
    test_actuator.cpp
    ${COMMON_IO}
    ${TEST_SRC_DIR}/io/protocol.cpp
    ${TEST_SRC_DIR}/control/actuator.cpp
)
target_include_directories(test_actuator PRIVATE
    ${TEST_SRC_DIR}/io
    ${TEST_SRC_DIR}/control
)

# ---------- test_gpio: 依赖 gpio（libgpiod） ----------
add_executable(test_gpio
    test_gpio.cpp
    ${TEST_SRC_DIR}/perception/gpio.cpp
)
target_include_directories(test_gpio PRIVATE
    ${TEST_SRC_DIR}/perception
)

# Linux: libgpiod + pthread
if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(test_imu PRIVATE Threads::Threads)
    target_link_libraries(test_actuator PRIVATE Threads::Threads)

    find_library(GPIOD_LIBRARY gpiod)
    if(GPIOD_LIBRARY)
        target_link_libraries(test_gpio PRIVATE ${GPIOD_LIBRARY} Threads::Threads)
        target_compile_definitions(test_gpio PRIVATE HAVE_GPIOD)
    else()
        message(WARNING "未找到 libgpiod，test_gpio 将编译为 stub（read 恒返回 0）")
    endif()
endif()
