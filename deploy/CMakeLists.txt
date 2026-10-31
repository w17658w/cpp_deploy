cmake_minimum_required(VERSION 3.10)
project(robot12_deploy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(robot12_deploy
    src/app/main.cpp
    src/app/config.cpp
    src/perception/wt901.cpp
    src/perception/gpio.cpp
    src/control/state.cpp
    src/control/mlp.cpp
    src/control/actuator.cpp
    src/io/serial.cpp
    src/io/protocol.cpp
)

target_include_directories(robot12_deploy PRIVATE
    src
    src/app
    src/perception
    src/control
    src/io
)

# std::thread 需要 pthread（Linux）
if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(robot12_deploy PRIVATE Threads::Threads)

    # libgpiod（触地 GPIO 直读）；找不到则编译为 stub（read 恒返回 0）
    find_library(GPIOD_LIBRARY gpiod)
    if(GPIOD_LIBRARY)
        target_link_libraries(robot12_deploy PRIVATE ${GPIOD_LIBRARY})
        add_compile_definitions(HAVE_GPIOD)
    else()
        message(WARNING "未找到 libgpiod，触地 GPIO 读取将编译为 stub（需 apt install libgpiod-dev）")
    endif()
endif()

# Windows 开启多字节字符集，避免 CreateFileA 编码问题
if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()
