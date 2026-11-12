// mlp.cpp — MLP 前向推理实现
#include "mlp.h"

#include <cstdio>
#include <cmath>

void Mlp::init(int obs_dim, int hidden_dim, int action_dim) {
    obs_ = obs_dim;
    hid_ = hidden_dim;
    act_ = action_dim;
    out_ = action_dim * 2;   // 24 = mean(12) + log_std(12)

    w1_.resize((size_t)hid_ * obs_);
    b1_.resize(hid_);
    w2_.resize((size_t)hid_ * hid_);
    b2_.resize(hid_);
    w3_.resize((size_t)out_ * hid_);
    b3_.resize(out_);
    scale_.resize(act_);
    bias_.resize(act_);

    h1_.resize(hid_);
    h2_.resize(hid_);
    o_.resize(out_);
}

static bool readFloats(FILE* f, std::vector<float>& v, size_t n) {
    v.resize(n);
    if (n == 0) return true;
    return fread(v.data(), sizeof(float), n, f) == n;
}

bool Mlp::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    bool ok = true;
    ok = ok && readFloats(f, w1_, (size_t)hid_ * obs_);
    ok = ok && readFloats(f, b1_, hid_);
    ok = ok && readFloats(f, w2_, (size_t)hid_ * hid_);
    ok = ok && readFloats(f, b2_, hid_);
    ok = ok && readFloats(f, w3_, (size_t)out_ * hid_);
    ok = ok && readFloats(f, b3_, out_);
    ok = ok && readFloats(f, scale_, act_);
    ok = ok && readFloats(f, bias_, act_);

    // 校验文件恰好读完（末尾无多余字节）
    if (ok) {
        int c = fgetc(f);
        if (c != EOF) ok = false;
    }
    fclose(f);
    return ok;
}

void Mlp::forward(const float* x, float* action) {
    // 层1: h1 = ReLU(W1 x + b1)
    for (int j = 0; j < hid_; ++j) {
        float s = b1_[j];
        const float* w = &w1_[(size_t)j * obs_];
        for (int i = 0; i < obs_; ++i) s += w[i] * x[i];
        h1_[j] = s > 0.0f ? s : 0.0f;
    }
    // 层2: h2 = ReLU(W2 h1 + b2)
    for (int j = 0; j < hid_; ++j) {
        float s = b2_[j];
        const float* w = &w2_[(size_t)j * hid_];
        for (int i = 0; i < hid_; ++i) s += w[i] * h1_[i];
        h2_[j] = s > 0.0f ? s : 0.0f;
    }
    // 层3: o = W3 h2 + b3（24 维）
    for (int k = 0; k < out_; ++k) {
        float s = b3_[k];
        const float* w = &w3_[(size_t)k * hid_];
        for (int i = 0; i < hid_; ++i) s += w[i] * h2_[i];
        o_[k] = s;
    }
    // 动作 = tanh(mean) * scale + bias（取前 act_ 维作为均值）
    for (int k = 0; k < act_; ++k) {
        action[k] = std::tanh(o_[k]) * scale_[k] + bias_[k];
    }
}
