cmake_minimum_required(VERSION 3.14)
project(ipdiskann LANGUAGES CXX)

# 查找 pybind11（使用的是系统或 CMake 提供的版本）
find_package(pybind11 REQUIRED)

# 查找 Python 安装目录（用于获取 site-packages 路径）
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 添加模块
pybind11_add_module(ipdiskann MODULE
        index_wrapper.cpp
        pybind_module.cpp
)

# 设置共享库的属性
set_target_properties(ipdiskann PROPERTIES
        PREFIX ""                     # 不要加 lib 前缀
        OUTPUT_NAME "ipdiskann"      # 输出名
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
)

# 链接 C++ 库
target_link_libraries(ipdiskann PRIVATE diskann)

install(TARGETS ipdiskann
        LIBRARY DESTINATION ${Python3_SITEARCH}   # Unix: .so
        RUNTIME DESTINATION ${Python3_SITEARCH}   # Windows: .pyd
        ARCHIVE DESTINATION ${Python3_SITEARCH}   # fallback
)
