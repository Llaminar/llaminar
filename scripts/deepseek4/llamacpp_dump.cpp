// Oracle dumper: run a llama.cpp model on a fixed prompt (prefill + greedy decode)
// and write selected graph tensors + logits as .npy files for parity work.
//
// usage: dsv4_dump -m model.gguf -o outdir -p "prompt" [-n decode_steps] [-r regex] [-t threads]
//   tokens.txt written with the prompt token ids and greedily generated ids.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <vector>
#include <sys/stat.h>

struct dump_state {
    std::string outdir;
    std::regex re;
    int step = 0;
};

static void write_npy(const std::string & path, const std::vector<float> & data, const std::vector<int64_t> & shape) {
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        hdr += std::to_string(shape[i]);
        hdr += (shape.size() == 1 || i + 1 < shape.size()) ? ", " : "";
    }
    hdr += "), }";
    size_t total = 10 + hdr.size() + 1;
    size_t pad = (64 - total % 64) % 64;
    hdr.append(pad, ' ');
    hdr += '\n';
    std::ofstream f(path, std::ios::binary);
    const char magic[] = "\x93NUMPY\x01\x00";
    f.write(magic, 8);
    uint16_t hl = (uint16_t) hdr.size();
    f.write((const char *) &hl, 2);
    f.write(hdr.data(), hdr.size());
    f.write((const char *) data.data(), data.size() * sizeof(float));
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (dump_state *) ud;
    const bool want = std::regex_match(std::string(t->name), st->re);
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<float> out(n);
    // gather in logical order (handles non-contiguous views)
    const auto * tr = ggml_get_type_traits(t->type);
    int64_t idx = 0;
    std::vector<float> row(t->ne[0]);
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
        const uint8_t * p = raw.data() + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        if (t->type == GGML_TYPE_F32) {
            for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) out[idx++] = *(const float *) (p + i0 * t->nb[0]);
        } else if (t->type == GGML_TYPE_I32) {
            for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) out[idx++] = (float) *(const int32_t *) (p + i0 * t->nb[0]);
        } else if (tr->to_float) {
            tr->to_float(p, row.data(), t->ne[0]);
            for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) out[idx++] = row[i0];
        } else {
            return true;
        }
    }
    std::vector<int64_t> shape;
    int nd = ggml_n_dims(t);
    for (int d = nd - 1; d >= 0; --d) shape.push_back(t->ne[d]);
    std::string name = t->name;
    for (auto & c : name) if (c == '/' || c == ' ' || c == '(' || c == ')') c = '_';
    write_npy(st->outdir + "/step" + std::to_string(st->step) + "__" + name + ".npy", out, shape);
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, outdir = "dump", prompt = "Hello", re = ".*";
    int n_decode = 4, threads = 6;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model_path = next();
        else if (a == "-o") outdir = next();
        else if (a == "-p") prompt = next();
        else if (a == "-f") { std::ifstream f(next()); prompt.assign(std::istreambuf_iterator<char>(f), {}); }
        else if (a == "-n") n_decode = std::stoi(next());
        else if (a == "-r") re = next();
        else if (a == "-t") threads = std::stoi(next());
    }
    mkdir(outdir.c_str(), 0755);
    llama_backend_init();
    dump_state st{outdir, std::regex(re)};
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(prompt.size() + 16);
    int nt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(),
                            llama_vocab_get_add_bos(vocab), true);
    if (nt < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(nt);
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 4096;
    cp.n_ubatch = 4096;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::ofstream tf(outdir + "/tokens.txt");
    for (auto t : toks) tf << t << " ";
    tf << "\n";
    llama_batch batch = llama_batch_get_one(toks.data(), (int) toks.size());
    std::vector<llama_token> gen;
    for (int s = 0; s <= n_decode; ++s) {
        st.step = s;
        if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed at step %d\n", s); return 1; }
        const float * logits = llama_get_logits_ith(ctx, -1);
        std::vector<float> l(logits, logits + n_vocab);
        write_npy(outdir + "/step" + std::to_string(s) + "__logits_last.npy", l, {n_vocab});
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (l[i] > l[best]) best = i;
        gen.push_back(best);
        batch = llama_batch_get_one(&gen.back(), 1);
    }
    for (auto t : gen) tf << t << " ";
    tf << "\n";
    fprintf(stderr, "prompt tokens %d, generated:", nt);
    for (auto t : gen) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
