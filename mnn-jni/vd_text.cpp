// vd_text.cpp - 设备端分词器 (GPT2 byte-level BPE, Qwen2 pretok) + VoiceDesign prompt 拼装
// 拼装逻辑与 PC 端逐位对齐 (见 scripts/export/vd_recon_prompt.py, exact_equal=True)
#include "vd_text.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace vdt {

// ---------- UTF-8 ----------
static int u8dec(const char* s, uint32_t& cp) {
    const unsigned char* p = (const unsigned char*)s;
    if (p[0] < 0x80) { cp = p[0]; return 1; }
    if ((p[0] & 0xE0) == 0xC0) { cp = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F); return 2; }
    if ((p[0] & 0xF0) == 0xE0) { cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); return 3; }
    if ((p[0] & 0xF8) == 0xF0) { cp = ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F); return 4; }
    cp = p[0]; return 1;
}
// Unicode 类别近似: \p{L} / \p{N} / \p{Zs}  (覆盖中英日常文本; 见 SKILL 验证记录)
static bool isLet(uint32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (c >= 0x00C0 && c <= 0x024F) return true;                 // Latin ext
    if (c >= 0x0370 && c <= 0x03FF) return true;                 // Greek
    if (c >= 0x0400 && c <= 0x04FF) return true;                 // Cyrillic
    if (c >= 0x3040 && c <= 0x30FF) return true;                 // kana
    if (c >= 0x3400 && c <= 0x4DBF) return true;                 // CJK ext A
    if (c >= 0x4E00 && c <= 0x9FFF) return true;                 // CJK
    if (c >= 0xAC00 && c <= 0xD7AF) return true;                 // Hangul
    if (c >= 0xF900 && c <= 0xFAFF) return true;                 // CJK compat
    if (c >= 0x20000 && c <= 0x2FA1F) return true;               // CJK ext B+
    if (c >= 0xFF21 && c <= 0xFF3A) return true;                 // fullwidth A-Z
    if (c >= 0xFF41 && c <= 0xFF5A) return true;                 // fullwidth a-z
    return false;
}
static bool isNum(uint32_t c) {
    if (c < 0x80) return c >= '0' && c <= '9';
    if (c >= 0x0660 && c <= 0x0669) return true;
    if (c >= 0xFF10 && c <= 0xFF19) return true;                 // fullwidth digits
    return false;
}
static bool isSp(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C || c == 0x85 || c == 0xA0 || c == 0x3000;
}
static bool isUpperA(char c) { return c >= 'A' && c <= 'Z'; }

// ---------- GPT-2 byte<->unicode ----------
static void buildB2U(std::unordered_map<unsigned char,std::string>& b2u, std::unordered_map<std::string,unsigned char>& u2b) {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    std::vector<int> cs = bs; int n = 0;
    for (int b = 0; b < 256; b++) if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n); n++; }
    for (size_t i = 0; i < bs.size(); i++) {
        uint32_t cp = (uint32_t)cs[i];
        char buf[8]; int L = 0;
        if (cp < 0x80) buf[L++] = (char)cp;
        else if (cp < 0x800) { buf[L++] = (char)(0xC0 | (cp >> 6)); buf[L++] = (char)(0x80 | (cp & 0x3F)); }
        else { buf[L++] = (char)(0xE0 | (cp >> 12)); buf[L++] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[L++] = (char)(0x80 | (cp & 0x3F)); }
        buf[L] = 0;
        b2u[(unsigned char)bs[i]] = std::string(buf, L);
        u2b[std::string(buf, L)] = (unsigned char)bs[i];
    }
}

// ---------- load ----------
static uint32_t rdU32(FILE* f) { uint32_t v = 0; if (fread(&v, 4, 1, f) != 1) v = 0; return v; }
static std::string rdStr(FILE* f) {
    unsigned char lo = 0, hi = 0;
    if (fread(&lo, 1, 1, f) != 1) return std::string();
    if (fread(&hi, 1, 1, f) != 1) return std::string();
    size_t n = (size_t)lo | ((size_t)hi << 8);
    std::string s(n, 0);
    if (n && fread(&s[0], 1, n, f) != n) s.clear();
    return s;
}
bool Tokenizer::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char mg[4]; if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "KOTB", 4) != 0) { fclose(f); return false; }
    rdU32(f);
    uint32_t nv = rdU32(f), nm = rdU32(f), na = rdU32(f);
    id2tok.resize(nv);
    for (uint32_t i = 0; i < nv; i++) { id2tok[i] = rdStr(f); tok2id[id2tok[i]] = (int)i; }
    for (uint32_t i = 0; i < nm; i++) {
        uint32_t a = rdU32(f), b = rdU32(f), m = rdU32(f);
        if (a != 0xFFFFFFFFu && b != 0xFFFFFFFFu && m != 0xFFFFFFFFu) mergeNew[(uint64_t)a << 32 | b] = (int)m;
    }
    for (uint32_t i = 0; i < na; i++) {
        uint32_t id = rdU32(f); unsigned char sp = 0;
        if (fread(&sp, 1, 1, f) != 1) break;
        std::string s = rdStr(f);
        added.push_back({s, (int)id, sp != 0});
        if (sp) { spec2id[s] = (int)id; }
    }
    fclose(f);
    std::sort(added.begin(), added.end(), [](const AddTok& x, const AddTok& y) { return x.s.size() > y.s.size(); });
    buildB2U(b2u, u2b);
    return true;
}

// ---------- BPE on one pre-token ----------
std::vector<int> Tokenizer::bpe(const std::string& raw) const {
    // bytes -> alphabet string
    std::string s;
    for (unsigned char c : raw) s += b2u.at(c);
    // 切成 alphabet 字符单元
    std::vector<std::string> sym;
    for (size_t i = 0; i < s.size();) { uint32_t cp; int L = u8dec(s.c_str() + i, cp); sym.push_back(s.substr(i, L)); i += L; }
    if (sym.empty()) return {};
    // 逐轮合并 rank 最小者
    while (sym.size() > 1) {
        int best = -1; int bestNew = -1;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            auto ia = tok2id.find(sym[i]), ib = tok2id.find(sym[i + 1]);
            if (ia == tok2id.end() || ib == tok2id.end()) continue;
            auto it = mergeNew.find((uint64_t)(uint32_t)ia->second << 32 | (uint32_t)ib->second);
            if (it == mergeNew.end()) continue;
            if (best < 0 || it->second < bestNew) { best = (int)i; bestNew = it->second; }
        }
        if (best < 0) break;
        sym[best] += sym[best + 1];
        sym.erase(sym.begin() + best + 1);
    }
    std::vector<int> ids;
    for (auto& t : sym) { auto it = tok2id.find(t); if (it != tok2id.end()) ids.push_back(it->second); }
    return ids;
}

// ---------- 预分词 (Qwen2/GPT-4 正则的等价扫描) ----------
std::vector<std::string> Tokenizer::preTok(const std::string& t) const {
    std::vector<std::string> out;
    size_t i = 0, n = t.size();
    std::vector<uint32_t> cps; std::vector<size_t> off;
    for (size_t k = 0; k < n;) { uint32_t cp; int L = u8dec(t.c_str() + k, cp); cps.push_back(cp); off.push_back(k); k += L; }
    off.push_back(n);
    size_t j = 0;
    auto push = [&](size_t a, size_t b) { if (b > a) out.push_back(t.substr(off[a], off[b] - off[a])); };
    while (j < cps.size()) {
        uint32_t c = cps[j];
        size_t start = j;
        // 1) 's 't 're 've 'm 'll 'd
        if (c == '\'') {
            std::string rest = t.substr(off[j], std::min<size_t>(4, n - off[j]));
            std::string low; for (char ch : rest) low += (char)tolower((unsigned char)ch);
            for (const char* cand : {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}) {
                size_t L = strlen(cand);
                if (low.size() >= L && low.compare(0, L, cand) == 0) {
                    // 're/'ve/'ll 需两个字母
                    size_t adv = L;
                    size_t k = j; while (adv-- && k < cps.size()) k++;
                    push(j, k); j = k; goto next;
                }
            }
        }
        // 2) [^\r\n\p{L}\p{N}]? \p{L}+
        {
            size_t k = j;
            if (!isLet(cps[k]) && !isNum(cps[k]) && cps[k] != '\r' && cps[k] != '\n') k++;
            size_t a = k;
            while (k < cps.size() && isLet(cps[k])) k++;
            if (k > a) { push(j, k); j = k; goto next; }
        }
        // 3) \p{N}
        if (isNum(c)) { push(j, j + 1); j++; goto next; }
        // 4) ' '? [^\s\p{L}\p{N}]+ [\r\n]*
        {
            size_t k = j;
            if (cps[k] == ' ') k++;
            size_t a = k;
            while (k < cps.size() && !isSp(cps[k]) && !isLet(cps[k]) && !isNum(cps[k])) k++;
            if (k > a) { while (k < cps.size() && (cps[k] == '\r' || cps[k] == '\n')) k++; push(j, k); j = k; goto next; }
        }
        // 5) \s* [\r\n]+
        {
            size_t k = j;
            while (k < cps.size() && isSp(cps[k]) && cps[k] != '\r' && cps[k] != '\n') k++;
            size_t a = k;
            while (k < cps.size() && (cps[k] == '\r' || cps[k] == '\n')) k++;
            if (k > a) { push(j, k); j = k; goto next; }
        }
        // 6/7) \s+
        if (isSp(c)) { size_t k = j; while (k < cps.size() && isSp(cps[k])) k++; push(j, k); j = k; goto next; }
        // 兜底: 单字符
        push(j, j + 1); j++;
        next:;
        (void)start;
    }
    return out;
}

// ---------- encode ----------
std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    size_t i = 0;
    while (i < text.size()) {
        // 先匹配 special added token (最长优先)
        int hitId = -1; size_t hitLen = 0;
        for (auto& a : added) {
            if (!a.special) continue;
            if (a.s.size() <= hitLen) continue;
            if (text.compare(i, a.s.size(), a.s) == 0) { hitId = a.id; hitLen = a.s.size(); }
        }
        if (hitId >= 0) { ids.push_back(hitId); i += hitLen; continue; }
        // 找下一个 special 起点, 其间做常规 BPE
        size_t nxt = text.size();
        for (auto& a : added) {
            if (!a.special) continue;
            size_t p = text.find(a.s, i);
            if (p != std::string::npos && p < nxt) nxt = p;
        }
        std::string span = text.substr(i, nxt - i);
        for (auto& pt : preTok(span)) { auto v = bpe(pt); ids.insert(ids.end(), v.begin(), v.end()); }
        i = nxt;
    }
    return ids;
}

int Tokenizer::idOf(const std::string& s) const { auto it = spec2id.find(s); return it == spec2id.end() ? -1 : it->second; }

// ---------- 文本表 ----------
static const void* mmapRO(const std::string& p, size_t bytes) {
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    void* q = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    return (q == MAP_FAILED) ? nullptr : q;
}
static std::vector<float> readF32(const std::string& p, size_t n) {
    std::vector<float> v(n);
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { v.clear(); return v; }
    if (fread(v.data(), 4, n, f) != n) v.clear();
    fclose(f);
    return v;
}
bool TextTables::load(const std::string& dir) {
    emb = (const uint16_t*)mmapRO(dir + "text_embedding.fp16", (size_t)151936 * 2048 * 2);
    fc1 = (const uint16_t*)mmapRO(dir + "text_proj_fc1.fp16", (size_t)2048 * 2048 * 2);
    fc2 = (const uint16_t*)mmapRO(dir + "text_proj_fc2.fp16", (size_t)2048 * 2048 * 2);
    b1 = readF32(dir + "text_proj_fc1_bias.f32", 2048);
    b2 = readF32(dir + "text_proj_fc2_bias.f32", 2048);
    return emb && fc1 && fc2 && b1.size() == 2048 && b2.size() == 2048;
}
static inline float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF; float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
}
// text_projection: fc2( SiLU( fc1(x) ) )
void TextTables::project(const float* in, float* out) const {
    static thread_local std::vector<float> mid;
    mid.resize(2048);
    for (int o = 0; o < 2048; o++) {
        const uint16_t* w = &fc1[(size_t)o * 2048];
        float acc = b1[o];
        for (int i = 0; i < 2048; i++) acc += h2f(w[i]) * in[i];
        mid[o] = acc / (1.0f + expf(-acc));
    }
    for (int o = 0; o < 2048; o++) {
        const uint16_t* w = &fc2[(size_t)o * 2048];
        float acc = b2[o];
        for (int i = 0; i < 2048; i++) acc += h2f(w[i]) * mid[i];
        out[o] = acc;
    }
}
void TextTables::embedProject(int id, float* out) const {
    static thread_local std::vector<float> tmp(2048);
    const uint16_t* e = &emb[(size_t)id * 2048];
    for (int i = 0; i < 2048; i++) tmp[i] = h2f(e[i]);
    project(tmp.data(), out);
}
void TextTables::embedRaw(int id, float* out) const {
    const uint16_t* e = &emb[(size_t)id * 2048];
    for (int i = 0; i < 2048; i++) out[i] = h2f(e[i]);
}

// ---------- prompt 拼装 (与 PC 端逐位对齐) ----------
// ids 布局: instruct_ids | asst_ids
int buildPrompt(const Tokenizer& tk, const TextTables& tb, const std::vector<int>& instructIds,
                const std::vector<int>& asstIds, int langId, const float* codecTable, int codecVocab,
                float* out, int maxTok) {
    // 注意: 真实 token 名不带 | (实测 tokenizer_config: 151671 <tts_pad>, 151672 <tts_text_bos>, 151673 <tts_text_eod>)
    const int BOS = tk.idOf("<tts_text_bos>"), EOS = tk.idOf("<tts_text_eod>"), PAD = tk.idOf("<tts_pad>");
    const int CPAD = 2148, CBOS = 2149, CTHINK = 2154, CTHINKB = 2156, CTHINKE = 2157;
    if (BOS < 0 || EOS < 0 || PAD < 0) return -1;
    static thread_local std::vector<float> bos(2048), eos(2048), pad(2048), tmp(2048), ce(2048);
    tb.embedProject(BOS, bos.data()); tb.embedProject(EOS, eos.data()); tb.embedProject(PAD, pad.data());
    auto put = [&](int idx, const float* v) { memcpy(out + (size_t)idx * 2048, v, 2048 * 4); };
    auto codec = [&](int id, float* dst) {
        if (id >= 0 && id < codecVocab) memcpy(dst, codecTable + (size_t)id * 2048, 2048 * 4);
        else memset(dst, 0, 2048 * 4);
    };
    int w = 0;
    if (w + (int)instructIds.size() > maxTok) return -1;
    for (int id : instructIds) { tb.embedProject(id, tmp.data()); put(w++, tmp.data()); }
    // codec 条件 (6)
    float codec6[6 * 2048];
    codec(CTHINK, codec6 + 0 * 2048); codec(CTHINKB, codec6 + 1 * 2048); codec(langId, codec6 + 2 * 2048);
    codec(CTHINKE, codec6 + 3 * 2048); codec(CPAD, codec6 + 4 * 2048); codec(CBOS, codec6 + 5 * 2048);
    // role (asst[0..2])
    for (int k = 0; k < 3; k++) { tb.embedProject(asstIds[k], tmp.data()); put(w++, tmp.data()); }
    // main (5) = cat([pad]*4, bos) + codec[:5]
    for (int k = 0; k < 4; k++) { for (int i = 0; i < 2048; i++) tmp[i] = pad[i] + codec6[k * 2048 + i]; put(w++, tmp.data()); }
    for (int i = 0; i < 2048; i++) tmp[i] = bos[i] + codec6[4 * 2048 + i]; put(w++, tmp.data());
    // body (N+1) = cat((te(asst[3:-5]), eos)) + ce([pad]*(N+1))
    int N = (int)asstIds.size() - 8;
    for (int k = 0; k < N; k++) {
        tb.embedProject(asstIds[3 + k], tmp.data()); codec(CPAD, ce.data());
        for (int i = 0; i < 2048; i++) tmp[i] += ce[i]; put(w++, tmp.data());
    }
    { codec(CPAD, ce.data()); for (int i = 0; i < 2048; i++) tmp[i] = eos[i] + ce[i]; put(w++, tmp.data()); }
    // tail (1) = pad + ce(bos)
    { codec(CBOS, ce.data()); for (int i = 0; i < 2048; i++) tmp[i] = pad[i] + ce[i]; put(w++, tmp.data()); }
    return w;
}

} // namespace vdt

// ---------- 自测 CLI ----------
#ifndef VDT_NO_MAIN
static std::string unesc(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size() && s[i+1] == 'n') { o += '\n'; i++; }
        else o += s[i];
    }
    return o;
}
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL_DIR CASES_TXT\n", argv[0]); return 2; }
    std::string d = argv[1]; if (d.back() != '/') d += '/';
    vdt::Tokenizer tk;
    if (!tk.load(d + "tokenizer.bin")) { printf("TOK_LOAD_FAIL\n"); return 3; }
    FILE* f = fopen(argv[2], "r");
    if (!f) { printf("CASES_OPEN_FAIL\n"); return 3; }
    char line[65536];
    int ci = 0;
    while (fgets(line, sizeof(line), f)) {
        std::string L(line);
        while (!L.empty() && (L.back() == '\n' || L.back() == '\r')) L.pop_back();
        if (L.empty()) continue;
        std::vector<std::string> col; size_t p = 0;
        while (true) { size_t q = L.find('\t', p); if (q == std::string::npos) { col.push_back(L.substr(p)); break; } col.push_back(L.substr(p, q - p)); p = q + 1; }
        if (col.size() < 4) continue;
        std::string ins = unesc(col[0]), txt = unesc(col[1]);
        auto ii = tk.encode("<|im_start|>user\n" + ins + "<|im_end|>\n");
        auto ai = tk.encode("<|im_start|>assistant\n" + txt + "<|im_end|>\n<|im_start|>assistant\n");
        printf("CASE %d\nGOT_A ", ci); for (int v : ii) printf("%d,", v); printf("\nEXP_A %s\n", col[2].c_str());
        printf("GOT_B "); for (int v : ai) printf("%d,", v); printf("\nEXP_B %s\n", col[3].c_str());
        ci++;
    }
    fclose(f);
    printf("TOKDONE %d\n", ci);
    return 0;
}
#endif
