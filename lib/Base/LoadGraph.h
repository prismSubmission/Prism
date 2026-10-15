
#pragma once
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cctype>
#include <boost/functional/hash.hpp>
#include <stdexcept>
#include <nlohmann/json.hpp>

inline void saveMultipleMatchesToBin(const std::string &out_path, const std::vector<std::unordered_map<int, int>> &full_matches)
{
    if (full_matches.empty())
    {
        throw std::runtime_error("full_matches is empty.");
    }

    // 推断 vertex_count：使用第一个 full_match 的大小
    int32_t vertex_count = static_cast<int32_t>(full_matches[0].size());
    int64_t embedding_count = static_cast<int64_t>(full_matches.size());

    // 默认 match_order 是 0 到 vertex_count-1
    std::vector<int> match_order(vertex_count);
    for (int i = 0; i < vertex_count; ++i)
    {
        match_order[i] = i;
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out)
        throw std::runtime_error("Failed to open output file: " + out_path);

    // 写入 header
    out.write(reinterpret_cast<const char *>(&vertex_count), sizeof(int32_t));
    out.write(reinterpret_cast<const char *>(&embedding_count), sizeof(int64_t));
    out.write(reinterpret_cast<const char *>(match_order.data()), sizeof(int32_t) * vertex_count);

    // 写入所有 embedding
    for (const auto &match : full_matches)
    {
        if (static_cast<int32_t>(match.size()) != vertex_count)
        {
            throw std::runtime_error("Mismatch in match size with vertex_count.");
        }

        std::vector<int> embedding(vertex_count);
        for (int i = 0; i < vertex_count; ++i)
        {
            auto it = match.find(i);
            if (it == match.end())
            {
                throw std::runtime_error("Missing match for query vertex ID: " + std::to_string(i));
            }
            embedding[i] = it->second;
        }

        out.write(reinterpret_cast<const char *>(embedding.data()), sizeof(int32_t) * vertex_count);
    }

    out.close();
}

inline void load_model_assignment(const std::string &csv_path, std::unordered_map<int, int> &vertex_model_map, //<vertex_id, model_code>
                                  std::vector<double> &model_mask)
{
    std::ifstream infile(csv_path);
    if (!infile.is_open())
    {
        throw std::runtime_error("Cannot open CSV: " + csv_path);
    }

    std::string line;
    std::getline(infile, line); // 跳过表头

    int max_vid = -1;

    while (std::getline(infile, line))
    {
        std::istringstream ss(line);
        std::string node_type, vertex_id_str, model_str;
        std::getline(ss, node_type, ',');
        std::getline(ss, vertex_id_str, ',');
        std::getline(ss, model_str, ',');
        while (!model_str.empty() && std::isspace(static_cast<unsigned char>(model_str.back())))
        {
            model_str.pop_back();
        }

        int vid = std::stoi(vertex_id_str);
        int model_code;

        if (model_str == "A")
            model_code = 0; // 情感分析
        else if (model_str == "B")
            model_code = 1; // 评论分类
        else if (model_str == "C")
            model_code = 2; // 图片分类
        else
            continue; // 空字符串或非法值

        vertex_model_map[vid] = model_code;
        if (vid > max_vid)
            max_vid = vid;
    }

    // 初始化 mask 向量，大小为最大 vertex_id + 1，默认 0
    model_mask.assign(max_vid + 1, 0.0);

    // 设置有模型的顶点位置为 0.5
    for (const auto &pair : vertex_model_map)
    {
        model_mask[pair.first] = 0.5;
    }
    return;
}

inline void load_model_assignment_parler(const std::string &json_path, std::unordered_map<int, int> &vertex_model_map, // <vertex_id, model_code>
                                         std::vector<double> &model_mask, std::string query_graph_name, int vertex_count)
{
    using json = nlohmann::json;

    // === 1. 读取 JSON 文件 ===
    std::ifstream ifs(json_path);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Failed to open json file: " + json_path);
    }

    json j;
    try
    {
        ifs >> j;
    }
    catch (const nlohmann::json::parse_error &e)
    {
        std::cerr << "JSON 解析错误: " << e.what() << std::endl;
        std::cerr << "错误位置: byte " << e.byte << std::endl;

        // 读取前几个字符看看
        ifs.seekg(0);
        std::string first100;
        std::getline(ifs, first100);
        std::cerr << "文件开头: " << first100.substr(0, 100) << std::endl;
    }

    // === 2. 找到对应的 query_graph_name ===
    if (!j.contains(query_graph_name))
    {
        // 没找到直接返回（或你也可以选择 throw）
        return;
    }

    const json &assignment = j[query_graph_name];

    vertex_model_map.clear();

    // === 3. 反转映射：model -> [vertex]  ==> vertex -> model ===
    int max_vertex_id = -1;

    for (auto it = assignment.begin(); it != assignment.end(); ++it)
    {
        // key 是 model_code（字符串形式的 int）
        int model_code = std::stoi(it.key());

        // value 是一个 vertex_id 列表
        const json &vertex_list = it.value();
        for (const auto &v : vertex_list)
        {
            int vertex_id = v.get<int>();
            if (model_code == 3 || model_code == 4 || model_code == 5)
            {
                vertex_model_map[vertex_id] = 3;
            }
            else if (model_code == 2)
            {
                vertex_model_map[vertex_id] = 4;
            }
            if (vertex_id > max_vertex_id)
            {
                max_vertex_id = vertex_id;
            }
        }
    }
    model_mask.assign(vertex_count, 0.0);

    // === 5. 给出现过的 vertex_id 赋 0.5 ===
    for (const auto &kv : vertex_model_map)
    {
        int vertex_id = kv.first;
        model_mask[vertex_id] = 0.5;
    }
}
