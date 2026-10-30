// config.cpp — 极简 key:value 配置解析器实现
#include "config.h"

#include <fstream>
#include <sstream>
#include <cctype>

std::string Config::trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

void Config::load(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("无法打开配置文件: " + path);

    kv_.clear();
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;      // 空行 / 注释

        size_t colon = t.find(':');
        if (colon == std::string::npos)
            throw std::runtime_error("配置格式错误(第 " + std::to_string(lineno) + " 行，缺冒号): " + line);

        std::string key = trim(t.substr(0, colon));
        std::string val = trim(t.substr(colon + 1));
        // 去掉字符串值两端的双引号
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);

        if (key.empty())
            throw std::runtime_error("配置格式错误(第 " + std::to_string(lineno) + " 行，键为空): " + line);

        kv_[key] = val;
    }
}

bool Config::has(const std::string& key) const {
    return kv_.find(key) != kv_.end();
}

const std::string& Config::raw(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end())
        throw std::runtime_error("配置缺少键: " + key);
    return it->second;
}

std::string Config::getString(const std::string& key) const { return raw(key); }

int Config::getInt(const std::string& key) const {
    return std::stoi(raw(key));
}

double Config::getDouble(const std::string& key) const {
    return std::stod(raw(key));
}

bool Config::getBool(const std::string& key) const {
    std::string v = raw(key);
    for (auto& c : v) c = (char)std::tolower((unsigned char)c);
    if (v == "true" || v == "1" || v == "yes" || v == "on") return true;
    if (v == "false" || v == "0" || v == "no" || v == "off") return false;
    throw std::runtime_error("配置键 " + key + " 不是合法 bool 值: " + raw(key));
}

std::vector<int> Config::getIntList(const std::string& key) const {
    std::string v = raw(key);
    std::vector<int> out;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ',')) {
        out.push_back(std::stoi(trim(item)));
    }
    return out;
}
