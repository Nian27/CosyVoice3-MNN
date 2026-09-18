#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
namespace vdt {
struct AddTok { std::string s; int id; bool special; };
struct Tokenizer {
    std::vector<std::string> id2tok;
    std::unordered_map<std::string,int> tok2id;
    std::unordered_map<uint64_t,int> mergeNew;   // (a<<32|b) -> merged id
    std::vector<AddTok> added;
    std::unordered_map<std::string,int> spec2id;
    std::unordered_map<unsigned char,std::string> b2u;
    std::unordered_map<std::string,unsigned char> u2b;
    bool load(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    int idOf(const std::string& s) const;
    std::vector<std::string> preTok(const std::string& t) const;
    std::vector<int> bpe(const std::string& raw) const;
};
struct TextTables {
    // 大表必须 mmap: 622MB 整块读入会把 App 内存压爆(实测 VmHWM 2.08GB 后挂死)
    const uint16_t* emb = nullptr;
    const uint16_t* fc1 = nullptr;
    const uint16_t* fc2 = nullptr;
    // ★ 该 MLP 是 bias=True 的: 漏掉 bias 会让 projection cos 掉到 0.83 (实测)
    std::vector<float> b1, b2;
    bool load(const std::string& dir);
    void project(const float* in, float* out) const;
    void embedProject(int id, float* out) const;
    void embedRaw(int id, float* out) const;
};
// 拼装: instructIds + asstIds -> [maxTok,2048] fp32, 返回 token 数 (<0 失败)
int buildPrompt(const Tokenizer& tk, const TextTables& tb, const std::vector<int>& instructIds,
                const std::vector<int>& asstIds, int langId, const float* codecTable, int codecVocab,
                float* out, int maxTok);
}
