// mlp.h — SAC Actor 前向推理（43 -> 256 -> 256 -> 24，取前 12 维均值）
//
// 权重布局（与 export_weights.py 导出顺序一致，全 float32）:
//   W1[hidden*obs] b1[hidden] W2[hidden*hidden] b2[hidden] W3[out*hidden] b3[out] scale[act] bias[act]
// 评估动作: action[k] = tanh(mean[k]) * scale[k] + bias[k]   ∈ [action_low, action_high]
#pragma once

#include <string>
#include <vector>

class Mlp {
public:
    // 设置维度（通常 obs=43, hidden=256, action=12）
    void init(int obs_dim, int hidden_dim, int action_dim);

    // 从 .bin 加载权重；失败返回 false
    bool load(const std::string& path);

    // 前向：obs[obs_dim] -> action[action_dim]
    void forward(const float* obs, float* action);

    int obsDim() const    { return obs_; }
    int actionDim() const { return act_; }

private:
    int obs_ = 0, hid_ = 0, out_ = 0, act_ = 0;

    std::vector<float> w1_, b1_, w2_, b2_, w3_, b3_, scale_, bias_;

    // 前向中间缓冲（预分配，避免每步动态分配）
    std::vector<float> h1_, h2_, o_;
};
