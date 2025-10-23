/**
 * @file rope_dump.cpp
 * @brief Minimal utility to apply llaminar::attn::apply_rope to synthetic Q/K and dump results as JSON.
 */
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include "operators/common/AttentionPrimitives.h"

static void print_json(const std::vector<float> &q_in, const std::vector<float> &k_in,
                       const std::vector<float> &q_rot, const std::vector<float> &k_rot,
                       int seq_len, int q_heads, int k_heads, int head_dim, int n_past, float freq_base)
{
    std::cout << "{\n";
    std::cout << "  \"seq_len\": " << seq_len << ",\n";
    std::cout << "  \"q_heads\": " << q_heads << ",\n";
    std::cout << "  \"k_heads\": " << k_heads << ",\n";
    std::cout << "  \"head_dim\": " << head_dim << ",\n";
    std::cout << "  \"n_past\": " << n_past << ",\n";
    std::cout << "  \"freq_base\": " << freq_base << ",\n";
    auto dump_array = [&](const char *name, const std::vector<float> &v)
    {
        std::cout << "  \"" << name << "\": [";
        for (size_t i = 0; i < v.size(); ++i)
        {
            if (i)
                std::cout << ',';
            std::cout << v[i];
        }
        std::cout << "]";
    };
    dump_array("q_input", q_in);
    std::cout << ",\n";
    dump_array("k_input", k_in);
    std::cout << ",\n";
    dump_array("q_rot", q_rot);
    std::cout << ",\n";
    dump_array("k_rot", k_rot);
    std::cout << "\n}" << std::endl;
}

int main(int argc, char **argv)
{
    int seq_len = 3;
    int q_heads = 4;
    int k_heads = 2;
    int head_dim = 8;
    int n_past = 0;
    float freq_base = 10000.f;
    unsigned seed = 42;
    std::string input_json;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto need = [&](const std::string &name)
        { if(i+1>=argc){ std::cerr<<"Missing value for "<<name<<"\n"; std::exit(1);} return std::string(argv[++i]); };
        if (a == "--seq")
            seq_len = std::stoi(need(a));
        else if (a == "--q-heads")
            q_heads = std::stoi(need(a));
        else if (a == "--k-heads")
            k_heads = std::stoi(need(a));
        else if (a == "--head-dim")
            head_dim = std::stoi(need(a));
        else if (a == "--n-past")
            n_past = std::stoi(need(a));
        else if (a == "--freq-base")
            freq_base = std::stof(need(a));
        else if (a == "--seed")
            seed = (unsigned)std::stoul(need(a));
        else if (a == "--input-json")
            input_json = need(a);
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    size_t q_size = (size_t)seq_len * q_heads * head_dim;
    size_t k_size = (size_t)seq_len * k_heads * head_dim;
    std::vector<float> q_in(q_size), k_in(k_size);
    if (!input_json.empty())
    {
        std::ifstream fin(input_json);
        if (!fin)
        {
            std::cerr << "Failed to open input JSON: " << input_json << "\n";
            return 1;
        }
        std::stringstream buffer;
        buffer << fin.rdbuf();
        std::string text = buffer.str();
        // Very lightweight JSON extraction (not robust): look for arrays named q_input or q / k_input or k
        auto extract = [&](const std::string &key, std::vector<float> &out)
        {
            size_t pos = text.find("\"" + key + "\"");
            if (pos == std::string::npos)
                return false;
            pos = text.find('[', pos);
            if (pos == std::string::npos)
                return false;
            size_t end = text.find(']', pos);
            if (end == std::string::npos)
                return false;
            std::string arr = text.substr(pos + 1, end - pos - 1);
            std::stringstream ss(arr);
            out.clear();
            out.reserve(1024);
            std::string num;
            while (std::getline(ss, num, ','))
            {
                if (num.empty())
                    continue;
                out.push_back(std::stof(num));
            }
            return true;
        };
        bool got_q = extract("q_input", q_in) || extract("q", q_in);
        bool got_k = extract("k_input", k_in) || extract("k", k_in);
        if (!got_q || q_in.size() != q_size)
        {
            std::cerr << "Input JSON missing q_input/q or size mismatch (expected " << q_size << ")\n";
            return 1;
        }
        if (!got_k || k_in.size() != k_size)
        {
            std::cerr << "Input JSON missing k_input/k or size mismatch (expected " << k_size << ")\n";
            return 1;
        }
    }
    else
    {
        for (auto &x : q_in)
            x = dist(rng);
        for (auto &x : k_in)
            x = dist(rng);
    }
    std::vector<float> q_rot = q_in; // copy then rotate in-place
    std::vector<float> k_rot = k_in;
    llaminar::attn::apply_rope(q_rot.data(), k_rot.data(), seq_len, head_dim, q_heads, k_heads, n_past, freq_base);
    print_json(q_in, k_in, q_rot, k_rot, seq_len, q_heads, k_heads, head_dim, n_past, freq_base);
    return 0;
}
