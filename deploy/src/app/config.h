// config.h — 极简 key:value 配置解析器（零依赖）
//
// 只支持 YAML 的最小子集：一行一个 "key: value"，# 开头为注释。
// 值类型支持：字符串（可带双引号）、数字（int/float）、bool（true/false）。
// 存储为 std::map<string,string>，读取时用 get<T>() 做类型转换。
#pragma once

#include <map>
#include <string>
#include <vector>
#include <stdexcept>

class Config {
public:
    // 从文件加载；失败抛 std::runtime_error
    void load(const std::string& path);

    // 是否存在某键
    bool has(const std::string& key) const;

    // 类型化取值（缺省时抛异常，便于尽早暴露配置错误）
    std::string getString(const std::string& key) const;
    int         getInt(const std::string& key) const;
    double      getDouble(const std::string& key) const;
    bool        getBool(const std::string& key) const;
    // 逗号分隔的整数列表，如 "68,69,70,71" -> {68,69,70,71}
    std::vector<int> getIntList(const std::string& key) const;
    // 逗号分隔的浮点列表，如 "0.463648,0,0" -> {0.463648,0,0}
    std::vector<double> getDoubleList(const std::string& key) const;

private:
    std::map<std::string, std::string> kv_;

    const std::string& raw(const std::string& key) const;
    static std::string trim(const std::string& s);
};
