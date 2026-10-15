// getOracle.cpp
#include "SubgraphMatching/CandidateSpace.h" // 类声明
#include "onnxruntime_cxx_api.h"
#include "tokenizers_cpp.h"
#include "SubgraphMatching/PatternGraph.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>
#include <string>
#include <array>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace tokenizers;

static std::string readFileToString(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Failed to open file: " + path);
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

static bool fileExists(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    return ifs.good();
}

static std::string resolveRepoRelativeFile(const std::string &relative_path)
{
    const std::array<std::string, 3> prefixes = {"", "../", "../../"};
    for (const auto &prefix : prefixes)
    {
        std::string candidate = prefix + relative_path;
        if (fileExists(candidate))
            return candidate;
    }

    // Keep the build-directory layout as the fallback used by the provided scripts.
    return "../" + relative_path;
}

static std::string modelFile(const std::string &model_name, const std::string &file_name)
{
    return resolveRepoRelativeFile("models/" + model_name + "/" + file_name);
}

bool saveToFile(const std::vector<std::string> &vec, const std::string &filename)
{
    std::ofstream outFile(filename);

    if (!outFile.is_open())
    {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    for (const auto &str : vec)
    {
        outFile << str << std::endl;
    }

    outFile.close();
    return true;
}

static void print_vec_float(const std::vector<float> &v)
{
    std::cout << "[";
    for (size_t i = 0; i < v.size(); ++i)
    {
        std::cout << v[i];
        if (i + 1 < v.size())
            std::cout << ", ";
    }
    std::cout << "]\n";
}

// 定义返回结果结构
struct ClassificationResult
{
    std::string label;
    float confidence;
};

namespace GraphLib
{
    namespace SubgraphMatching
    {
        bool CandidateSpace::getPicPathForVertexIds(const std::vector<int> &vertexIds, std::vector<std::string> &vertex_to_pic,
                                                    std::unordered_map<int, int> &vertex_to_index)
        {
            std::vector<std::string> &result = vertex_to_pic;

            // 1. 读取vertex_product.csv，建立asin到vertex_id的映射
            std::unordered_map<std::string, int> asinToVertexId;
            std::unordered_map<int, std::string> VertexIdToasin;
            std::ifstream vfile(vertex_product_path_);
            std::string line;

            std::getline(vfile, line); // 跳过标题行

            while (std::getline(vfile, line))
            {
                std::stringstream ss(line);
                std::string vertexIdStr, asin, parentAsin;

                std::getline(ss, vertexIdStr, ',');
                std::getline(ss, asin, ',');
                std::getline(ss, parentAsin, ',');

                int vertexId = std::stoi(vertexIdStr);
                asinToVertexId[asin] = vertexId;
                if (!parentAsin.empty())
                {
                    asinToVertexId[parentAsin] = vertexId;
                    VertexIdToasin[vertexId] = parentAsin;
                }
            }

            // 2. 为每个输入的vertex_id获取parentsin
            for (int i = 0; i < vertexIds.size(); ++i)
            {
                int vertexId = vertexIds[i];
                auto it = VertexIdToasin.find(vertexId);
                if (it != VertexIdToasin.end())
                {
                    vertex_to_pic.push_back(it->second);
                }
                else
                {
                    vertex_to_pic.push_back(""); // 没有找到对应文本
                }
                vertex_to_index[vertexId] = i;
            }

            std::string root_path = image_path_;
            if (root_path.empty())
            {
                root_path = resolveRepoRelativeFile("dataset/" + dataset_ + "/pic/images_kcore");
            }
            for (int i = 0; i < vertex_to_pic.size(); ++i)
            {
                std::string tmp_path = root_path + "/0_" + vertex_to_pic[i] + "_img1.jpg";
                std::ifstream infile(tmp_path);
                if (!infile.good())
                {
                    tmp_path = root_path + "/0_" + vertex_to_pic[i] + "_img2.jpg";
                }
                vertex_to_pic[i] = tmp_path;
            }

            return true;
        }

        int CandidateSpace::predictAll(std::string model_key, std::vector<std::string> &all_inputs, std::vector<float> &all_results)
        {
            size_t total_count = all_inputs.size();
            if (total_count == 0)
                return 0;

            // 针对 3090 的安全 Batch Size
            size_t BATCH_SIZE = 128;

            all_results.clear();
            all_results.reserve(total_count);

            cout << "[ONNX Score] Predicting " << total_count << " samples in batches of " << BATCH_SIZE << "." << endl;
            for (size_t start_idx = 0; start_idx < total_count; start_idx += BATCH_SIZE)
            {
                size_t end_idx = std::min(start_idx + BATCH_SIZE, total_count);

                std::vector<std::string> batch_inputs(all_inputs.begin() + start_idx, all_inputs.begin() + end_idx);
                // batch_inputs[0] = std::string("I really support Trump");
                // batch_inputs[1] = std::string("I really hate Trump");
                std::vector<float> batch_results = predictBatch_with_ONNX_score(model_key, batch_inputs);
                all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
            }
            return (int)all_results.size();
        }

        int CandidateSpace::predictAll_baseline(std::string model_key, std::vector<std::string> &all_inputs, std::vector<uint8_t> &all_results)
        {
            size_t total_count = all_inputs.size();
            if (total_count == 0)
                return 0;

            // 针对 3090 的安全 Batch Size
            size_t BATCH_SIZE = 128;

            all_results.clear();
            all_results.reserve(total_count);

            cout << "[ONNX Score] Predicting " << total_count << " samples in batches of " << BATCH_SIZE << "." << endl;
            for (size_t start_idx = 0; start_idx < total_count; start_idx += BATCH_SIZE)
            {
                size_t end_idx = std::min(start_idx + BATCH_SIZE, total_count);

                std::vector<std::string> batch_inputs(all_inputs.begin() + start_idx, all_inputs.begin() + end_idx);
                std::vector<uint8_t> batch_results = predictBatch_with_ONNX(model_key, batch_inputs);
                all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
            }
            return (int)all_results.size();
        }

        void CandidateSpace::buildCompactVectorMapping(std::vector<int> &needed_vertex_ids, std::vector<std::string> &vertex_to_text,
                                                       std::unordered_map<int, int> &vertex_to_index)
        {
            // CSV解析辅助函数 - 正确处理引号包裹的字段
            auto parseCSVLine = [](const std::string &line) -> std::vector<std::string>
            {
                std::vector<std::string> fields;
                std::string field;
                bool in_quotes = false;

                for (size_t i = 0; i < line.length(); i++)
                {
                    char c = line[i];

                    if (c == '"')
                    {
                        // 处理转义的双引号：两个连续的双引号表示一个双引号字符
                        if (i + 1 < line.length() && line[i + 1] == '"')
                        {
                            field += '"';
                            i++; // 跳过下一个双引号
                        }
                        else
                        {
                            in_quotes = !in_quotes;
                        }
                    }
                    else if (c == ',' && !in_quotes)
                    {
                        // 不在引号内的逗号表示字段结束
                        fields.push_back(field);
                        field.clear();
                    }
                    else
                    {
                        field += c;
                    }
                }

                // 添加最后一个字段
                fields.push_back(field);
                return fields;
            };

            // 1. 加载 vertex_review.csv，只加载需要的 vertex_id
            std::unordered_map<int, int> vertex_to_row;
            {
                std::ifstream file(vertex_review_path_);
                if (!file.is_open())
                {
                    throw std::runtime_error("无法打开文件: " + vertex_review_path_);
                }

                std::string line;
                std::getline(file, line); // 跳过标题行 "vertex_id,row_id"

                std::unordered_set<int> needed_set(needed_vertex_ids.begin(), needed_vertex_ids.end());

                int line_num = 1;
                while (std::getline(file, line))
                {
                    line_num++;

                    // 使用CSV解析函数
                    std::vector<std::string> fields = parseCSVLine(line);
                    if (fields.size() < 2)
                    {
                        // 字段不足，跳过这一行
                        continue;
                    }

                    try
                    {
                        int vertex_id = std::stoi(fields[0]);

                        // 检查是否是需要的 vertex_id
                        if (needed_set.find(vertex_id) != needed_set.end())
                        {
                            int row_id = std::stoi(fields[1]);
                            vertex_to_row[vertex_id] = row_id;
                        }
                    }
                    catch (const std::exception &e)
                    {
                        // 转换失败，跳过这一行
                        continue;
                    }
                }
            }

            // 2. 初始化数据结构
            vertex_to_text.resize(vertex_to_row.size(), "");
            vertex_to_index.reserve(vertex_to_row.size());

            // 3. 分配索引
            size_t current_index = 0;
            for (const auto &[vertex_id, row_id] : vertex_to_row)
            {
                vertex_to_index[vertex_id] = current_index++;
            }

            // 4. 建立反向映射：row_id → [vertex_ids]
            std::unordered_map<int, std::vector<int>> row_to_vertices;
            for (const auto &[vertex_id, row_id] : vertex_to_row)
            {
                row_to_vertices[row_id].push_back(vertex_id);
            }

            // 5. 加载 reviews_kcore.csv，只处理相关的行
            {
                std::ifstream file(reviews_kcore_path_);
                if (!file.is_open())
                {
                    throw std::runtime_error("无法打开文件: " + reviews_kcore_path_);
                }

                std::string line;
                std::getline(file, line); // 跳过标题行

                int current_row_id = 0;
                while (std::getline(file, line))
                {
                    auto vertex_it = row_to_vertices.find(current_row_id);
                    if (vertex_it != row_to_vertices.end())
                    {
                        // 解析CSV行
                        std::vector<std::string> fields = parseCSVLine(line);

                        if (fields.size() >= 3)
                        {

                            std::string text = fields[1] + " " + fields[2];

                            for (int vertex_id : vertex_it->second)
                            {
                                auto index_it = vertex_to_index.find(vertex_id);
                                if (index_it != vertex_to_index.end())
                                {
                                    vertex_to_text[index_it->second] = text;
                                }
                            }
                        }
                    }
                    current_row_id++;
                }
            }
        }

        void CandidateSpace::buildCompactVectorMapping_parler(std::vector<int> &needed_vertex_ids, std::vector<std::string> &vertex_to_text,
                                                              std::unordered_map<int, int> &vertex_to_index)
        {
            // CSV解析辅助函数，处理引号包裹的字段
            auto parseCSVLine = [](const std::string &line) -> std::vector<std::string>
            {
                std::vector<std::string> fields;
                std::string field;
                bool in_quotes = false;

                for (size_t i = 0; i < line.length(); i++)
                {
                    char c = line[i];
                    if (c == '"')
                    {
                        if (i + 1 < line.length() && line[i + 1] == '"')
                        {
                            field += '"';
                            i++; // 跳过转义
                        }
                        else
                        {
                            in_quotes = !in_quotes;
                        }
                    }
                    else if (c == ',' && !in_quotes)
                    {
                        fields.push_back(field);
                        field.clear();
                    }
                    else
                    {
                        field += c;
                    }
                }
                fields.push_back(field);
                return fields;
            };

            // 1. 加载 vertex_review_path_：internal_id -> orig_id
            std::unordered_map<int, std::string> vertex_to_orig;
            {
                std::ifstream file(vertex_review_path_);
                if (!file.is_open())
                {
                    throw std::runtime_error("无法打开文件: " + vertex_review_path_);
                }

                std::string line;
                std::getline(file, line); // 跳过标题行

                std::unordered_set<int> needed_set(needed_vertex_ids.begin(), needed_vertex_ids.end());

                while (std::getline(file, line))
                {
                    std::vector<std::string> fields = parseCSVLine(line);
                    if (fields.size() < 2)
                        continue; // 至少需要 internal_id 和 orig_id

                    try
                    {
                        int vertex_id = std::stoi(fields[0]);
                        if (needed_set.find(vertex_id) != needed_set.end())
                        {
                            std::string orig_id = fields[1];
                            vertex_to_orig[vertex_id] = orig_id;
                        }
                    }
                    catch (...)
                    {
                        continue; // 转换失败跳过
                    }
                }
            }

            std::cout << "Loaded " << vertex_to_orig.size() << " vertex to orig_id mappings." << std::endl;
            // 2. 初始化数据结构
            vertex_to_text.resize(vertex_to_orig.size(), "");
            vertex_to_index.reserve(vertex_to_orig.size());

            // 3. 分配 compact index
            size_t current_index = 0;
            for (const auto &[vertex_id, orig_id] : vertex_to_orig)
            {
                vertex_to_index[vertex_id] = current_index++;
            }

            // 4. 建立 orig_id -> [vertex_id] 的反向映射
            std::unordered_map<std::string, std::vector<int>> orig_to_vertices;
            for (const auto &[vertex_id, orig_id] : vertex_to_orig)
            {
                orig_to_vertices[orig_id].push_back(vertex_id);
            }

            // 5. 遍历 reviews_kcore_path_，加载 body 文本
            {
                std::ifstream file(reviews_kcore_path_);
                if (!file.is_open())
                {
                    throw std::runtime_error("无法打开文件: " + reviews_kcore_path_);
                }

                std::string header_line;
                std::getline(file, header_line);

                // 找到 id:ID 和 body 列的索引
                std::vector<std::string> headers = parseCSVLine(header_line);
                int idx_id = -1, idx_body = -1;
                for (size_t i = 0; i < headers.size(); i++)
                {
                    if (headers[i] == "id:ID")
                        idx_id = i;
                    if (headers[i] == "body")
                        idx_body = i;
                }
                if (idx_id == -1 || idx_body == -1)
                {
                    throw std::runtime_error("reviews_kcore CSV 缺少 id:ID 或 body 列");
                }

                std::string line;
                while (std::getline(file, line))
                {
                    std::vector<std::string> fields = parseCSVLine(line);
                    if ((int)fields.size() <= std::max(idx_id, idx_body))
                        continue;

                    std::string orig_id = fields[idx_id];
                    auto it = orig_to_vertices.find(orig_id);
                    if (it != orig_to_vertices.end())
                    {
                        std::string text = fields[idx_body];
                        for (int vertex_id : it->second)
                        {
                            int idx = vertex_to_index[vertex_id];
                            vertex_to_text[idx] = text;
                        }
                    }
                }
            }
        }

        bool CandidateSpace::getProductTextsForVertexIds(const std::vector<int> &vertexIds, std::vector<std::string> &vertex_to_text,
                                                         std::unordered_map<int, int> &vertex_to_index, int maxLength)
        {

            std::vector<std::string> &result = vertex_to_text;

            // 1. 读取vertex_product.csv，建立asin到vertex_id的映射
            std::unordered_map<std::string, int> asinToVertexId;
            std::ifstream vfile(vertex_product_path_);
            std::string line;

            std::getline(vfile, line); // 跳过标题行

            while (std::getline(vfile, line))
            {
                std::stringstream ss(line);
                std::string vertexIdStr, asin, parentAsin;

                std::getline(ss, vertexIdStr, ',');
                std::getline(ss, asin, ',');
                std::getline(ss, parentAsin, ',');

                int vertexId = std::stoi(vertexIdStr);
                asinToVertexId[asin] = vertexId;
                if (!parentAsin.empty())
                {
                    asinToVertexId[parentAsin] = vertexId;
                }
            }

            // 2. 创建vertex_id到文本的映射
            std::unordered_map<int, std::string> vertexIdToText;

            // 读取reviews_kcore.csv
            std::ifstream rfile(product_kcore_path__);
            std::getline(rfile, line); // 跳过标题行

            while (std::getline(rfile, line))
            {
                std::stringstream ss(line);
                std::vector<std::string> fields;
                std::string field;

                while (std::getline(ss, field, ','))
                {
                    fields.push_back(field);
                }

                if (fields.size() > 4)
                { // 确保有足够的字段
                    std::string title = fields[0];
                    std::string text = fields[4];
                    std::string parentAsin = fields[6];

                    // 查找对应的vertex_id
                    auto it = asinToVertexId.find(parentAsin);
                    if (it != asinToVertexId.end())
                    {
                        std::string combined = title + " " + text;
                        if (combined.length() > maxLength)
                        {
                            combined = combined.substr(0, maxLength);
                        }

                        vertexIdToText[it->second] = combined;
                    }
                }
            }

            // 3. 为每个输入的vertex_id获取文本
            for (int i = 0; i < vertexIds.size(); i++)
            {
                int vertexId = vertexIds[i];
                auto it = vertexIdToText.find(vertexId);
                if (it != vertexIdToText.end())
                {
                    vertex_to_text.push_back(it->second);
                }
                else
                {
                    vertex_to_text.push_back(""); // 没有找到对应文本
                }
                vertex_to_index[vertexIds[i]] = i;
            }

            return true;
        }

        bool CandidateSpace::getProductTextsForVertexIds_parler(const std::vector<int> &vertexIds, std::vector<std::string> &vertex_to_text,
                                                                std::unordered_map<int, int> &vertex_to_index, int maxLength)
        {
            // CSV解析辅助函数，处理引号包裹字段和逗号
            auto parseCSVLine = [](const std::string &line) -> std::vector<std::string>
            {
                std::vector<std::string> fields;
                std::string field;
                bool in_quotes = false;

                for (size_t i = 0; i < line.length(); i++)
                {
                    char c = line[i];
                    if (c == '"')
                    {
                        if (i + 1 < line.length() && line[i + 1] == '"')
                        {
                            field += '"';
                            i++;
                        }
                        else
                        {
                            in_quotes = !in_quotes;
                        }
                    }
                    else if (c == ',' && !in_quotes)
                    {
                        fields.push_back(field);
                        field.clear();
                    }
                    else
                    {
                        field += c;
                    }
                }
                fields.push_back(field);
                return fields;
            };

            std::vector<std::string> &result = vertex_to_text;

            // 1. 读取 vertex_product_path_，建立 vertex_id -> orig_id 映射
            std::unordered_map<int, std::string> vertexIdToOrig;
            {
                std::ifstream vfile(vertex_product_path_);
                if (!vfile.is_open())
                {
                    throw std::runtime_error("无法打开文件: " + vertex_product_path_);
                }

                std::string line;
                std::getline(vfile, line); // 跳过标题行

                std::unordered_set<int> neededSet(vertexIds.begin(), vertexIds.end());

                while (std::getline(vfile, line))
                {
                    std::vector<std::string> fields = parseCSVLine(line);
                    if (fields.size() < 2)
                        continue;

                    try
                    {
                        int vertexId = std::stoi(fields[0]);
                        if (neededSet.find(vertexId) != neededSet.end())
                        {
                            std::string orig_id = fields[1];
                            vertexIdToOrig[vertexId] = orig_id;
                        }
                    }
                    catch (...)
                    {
                        continue;
                    }
                }
            }

            // 2. 建立 orig_id -> [vertex_id] 的反向映射
            std::unordered_map<std::string, std::vector<int>> origToVertices;
            for (const auto &[vertexId, origId] : vertexIdToOrig)
            {
                origToVertices[origId].push_back(vertexId);
            }

            // 3. 遍历原始数据集，获取 body 文本
            {
                std::ifstream rfile(product_kcore_path__);
                if (!rfile.is_open())
                {
                    throw std::runtime_error("无法打开文件: " + product_kcore_path__);
                }

                std::string header_line;
                std::getline(rfile, header_line);

                // 找到 id:ID 和 body 列索引
                std::vector<std::string> headers = parseCSVLine(header_line);
                int idx_id = -1, idx_body = -1;
                for (size_t i = 0; i < headers.size(); i++)
                {
                    if (headers[i] == "id:ID")
                        idx_id = i;
                    if (headers[i] == "body")
                        idx_body = i;
                }
                if (idx_id == -1 || idx_body == -1)
                {
                    throw std::runtime_error("CSV 缺少 id:ID 或 body 列");
                }

                std::string line;
                while (std::getline(rfile, line))
                {
                    std::vector<std::string> fields = parseCSVLine(line);
                    if ((int)fields.size() <= std::max(idx_id, idx_body))
                        continue;

                    std::string orig_id = fields[idx_id];
                    auto it = origToVertices.find(orig_id);
                    if (it != origToVertices.end())
                    {
                        std::string body = fields[idx_body];
                        if ((int)body.length() > maxLength)
                        {
                            body = body.substr(0, maxLength);
                        }

                        for (int vertexId : it->second)
                        {
                            int idx = vertex_to_index.size();
                            vertex_to_index[vertexId] = idx;
                            vertex_to_text.push_back(body);
                        }
                    }
                }
            }

            // 4. 保证每个 vertexIds 都有一个条目
            for (int vertexId : vertexIds)
            {
                if (vertex_to_index.find(vertexId) == vertex_to_index.end())
                {
                    int idx = vertex_to_index.size();
                    vertex_to_index[vertexId] = idx;
                    vertex_to_text.push_back("");
                }
            }
            //for (int i = 0; i < 11; i++)
            //{
            //    std::cout << "vertex_to_text[" << i << "] = " << vertex_to_text[i] << std::endl;
            //}
            return true;
        }

        bool CandidateSpace::buildModelInputForBatch(std::string model_key, std::unordered_set<int> *needed_vertex_set)
        {
            all_string_inputs.resize(5);
            all_string_idx.resize(5);

            int model_key_int = -1;
            if (model_key == "0")
                model_key_int = 0;
            else if (model_key == "1")
                model_key_int = 1;
            else if (model_key == "2")
                model_key_int = 2;

            bool allocate_needed_set = false;
            if (needed_vertex_set == nullptr)
            {
                needed_vertex_set = new std::unordered_set<int>();
                allocate_needed_set = true;
                for (auto &pair : query_->predicateMap)
                {
                    int u = pair.first;
                    int vertex_model_key = getModelKey(u);
                    if (vertex_model_key == model_key_int)
                    {
                        for (int v_idx = 0; v_idx < GetCandidateSetSize(u); ++v_idx)
                        {
                            int v = GetCandidate(u, v_idx);
                            needed_vertex_set->insert(v);
                        }
                    }
                }
            }

            if (model_key == "0")
            {
                std::vector<std::string> &vertex_to_text = all_string_inputs[0];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[0];
                std::vector<int> needed_vertex_ids(needed_vertex_set->begin(), needed_vertex_set->end());
                std::vector<float> score_list;

                buildCompactVectorMapping(needed_vertex_ids, vertex_to_text, vertex_to_index);
                cout << "00" << endl;
                int ret_size = predictAll("00", vertex_to_text, score_list);
                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 0)
                    {
                        std::unordered_map<int, std::pair<double, TreeNode *>> &node_map_inner = node_map[qnode];
                        std::multimap<double, int, std::greater<double>> &map_score_inner = map_Score[qnode];
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            double score = -1.0;
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                score = score_list[index_it->second];
                            }
                            node_map_inner[C_idx].first = score;
                            map_score_inner.insert({score, C_idx});
                        }
                    }
                }
            }
            else if (model_key == "1")
            {
                int maxLength = 512;
                std::vector<std::string> &vertex_to_text = all_string_inputs[1];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[1];
                std::vector<int> needed_vertex_ids(needed_vertex_set->begin(), needed_vertex_set->end());

                cout << "needed_vertex_ids size: " << needed_vertex_ids.size() << endl;
                getProductTextsForVertexIds(needed_vertex_ids, vertex_to_text, vertex_to_index, maxLength);
                cout << vertex_to_text.size() << endl;
                cout << "10" << endl;
                std::vector<float> score_list;
                int ret_size = predictAll("10", vertex_to_text, score_list);
                // std::cout << "score_list size = " << score_list.size() << std::endl;
                // for (int i = 0; i < std::min(10, (int)score_list.size()); ++i)
                //{
                //     std::cout << "score_list[" << i << "] = " << score_list[i] << std::endl;
                // }
                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 1)
                    {
                        std::unordered_map<int, std::pair<double, TreeNode *>> &node_map_inner = node_map[qnode];
                        std::multimap<double, int, std::greater<double>> &map_score_inner = map_Score[qnode];
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            double score = -1.0;
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                score = score_list[index_it->second];
                            }
                            node_map_inner[C_idx].first = score;
                            map_score_inner.insert({score, C_idx});
                        }
                    }
                }
            }
            else if (model_key == "2")
            {
                std::vector<std::string> &vertex_to_text = all_string_inputs[2];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[2];
                std::vector<int> needed_vertex_ids(needed_vertex_set->begin(), needed_vertex_set->end());

                getPicPathForVertexIds(needed_vertex_ids, vertex_to_text, vertex_to_index);
                cout << vertex_to_text.size() << endl;
                cout << "20" << endl;
                std::vector<float> score_list;

                int ret_size = predictAll("20", vertex_to_text, score_list);

                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 2)
                    {
                        std::unordered_map<int, std::pair<double, TreeNode *>> &node_map_inner = node_map[qnode];
                        std::multimap<double, int, std::greater<double>> &map_score_inner = map_Score[qnode];
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            double score = -1.0;
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                score = score_list[index_it->second];
                            }
                            node_map_inner[C_idx].first = score;
                            map_score_inner.insert({score, C_idx});
                        }
                    }
                }
            }
            else if (model_key == "3")
            {
                std::vector<std::string> &vertex_to_text = all_string_inputs[3];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[3];
                std::vector<int> needed_vertex_ids(needed_vertex_set->begin(), needed_vertex_set->end());
                buildCompactVectorMapping_parler(needed_vertex_ids, vertex_to_text, vertex_to_index);
                cout << "30" << endl;
                std::vector<float> score_list;
                //saveToFile(vertex_to_text, "debug_vertex_to_text_model3.txt");
                int ret_size = predictAll("30", vertex_to_text, score_list);
                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 3)
                    {
                        std::unordered_map<int, std::pair<double, TreeNode *>> &node_map_inner = node_map[qnode];
                        std::multimap<double, int, std::greater<double>> &map_score_inner = map_Score[qnode];
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            double score = -1.0;
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                score = score_list[index_it->second];
                            }
                            node_map_inner[C_idx].first = score;
                            map_score_inner.insert({score, C_idx});
                        }
                    }
                }
            }
            else if (model_key == "4")
            {
                int maxLength = 512;
                std::vector<std::string> &vertex_to_text = all_string_inputs[4];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[4];
                std::vector<int> needed_vertex_ids(needed_vertex_set->begin(), needed_vertex_set->end());
                getProductTextsForVertexIds_parler(needed_vertex_ids, vertex_to_text, vertex_to_index, maxLength);
                //cout << "10,2" << endl;
                std::vector<float> score_list;
                int ret_size = predictAll("40", vertex_to_text, score_list);
                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 4)
                    {
                        std::unordered_map<int, std::pair<double, TreeNode *>> &node_map_inner = node_map[qnode];
                        std::multimap<double, int, std::greater<double>> &map_score_inner = map_Score[qnode];
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            double score = -1.0;
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                score = score_list[index_it->second];
                            }
                            node_map_inner[C_idx].first = score;
                            map_score_inner.insert({score, C_idx});
                        }
                    }
                }
            }
            else
            {
                throw std::runtime_error("Unknown model key in buildModelInputForBatch: " + model_key);
            }
            return true;
        }

        bool CandidateSpace::buildModelInputForBatch_baseline(std::string model_key, std::unordered_set<int> *needed_vertex_set)
        {
            all_string_inputs.resize(5);
            all_string_idx.resize(5);

            int model_key_int = -1;
            if (model_key == "0")
                model_key_int = 0;
            else if (model_key == "1")
                model_key_int = 1;
            else if (model_key == "2")
                model_key_int = 2;
            else if (model_key == "3")
                model_key_int = 3;
            else if (model_key == "4")
                model_key_int = 4;

            bool allocate_needed_set = false;
            if (needed_vertex_set == nullptr)
            {
                needed_vertex_set = new std::unordered_set<int>();
                allocate_needed_set = true;
                for (auto &pair : query_->predicateMap)
                {
                    int u = pair.first;
                    int vertex_model_key = getModelKey(u);
                    if (vertex_model_key == model_key_int)
                    {
                        for (int v_idx = 0; v_idx < GetCandidateSetSize(u); ++v_idx)
                        {
                            int v = GetCandidate(u, v_idx);
                            needed_vertex_set->insert(v);
                        }
                    }
                }
            }

            if (model_key == "0")
            {
                std::vector<std::string> &vertex_to_text = all_string_inputs[0];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[0];
                std::vector<int> needed_vertex_ids(needed_vertex_set->begin(), needed_vertex_set->end());
                std::vector<uint8_t> score_list;

                buildCompactVectorMapping(needed_vertex_ids, vertex_to_text, vertex_to_index);
                cout << "01" << endl;
                int ret_size = predictAll_baseline("01", vertex_to_text, score_list);
                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 0)
                    {
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                oracleRecord[qnode][C_idx] = score_list[index_it->second];
                            }
                        }
                    }
                }
            }
            else if (model_key == "1")
            {
                int maxLength = 512;
                std::vector<std::string> &vertex_to_text = all_string_inputs[1];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[1];
                std::vector<int> needed_vertex_ids(needed_vertex_set->begin(), needed_vertex_set->end());

                cout << "needed_vertex_ids size: " << needed_vertex_ids.size() << endl;
                getProductTextsForVertexIds(needed_vertex_ids, vertex_to_text, vertex_to_index, maxLength);
                cout << vertex_to_text.size() << endl;
                cout << "11" << endl;
                std::vector<uint8_t> score_list;
                int ret_size = predictAll_baseline("11", vertex_to_text, score_list);
                // std::cout << "score_list size = " << score_list.size() << std::endl;
                // for (int i = 0; i < std::min(10, (int)score_list.size()); ++i)
                //{
                //     std::cout << "score_list[" << i << "] = " << score_list[i] << std::endl;
                // }
                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 1)
                    {
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                oracleRecord[qnode][C_idx] = score_list[index_it->second];
                            }
                        }
                    }
                }
            }
            else if (model_key == "2")
            {
                std::vector<std::string> &vertex_to_text = all_string_inputs[2];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[2];
                std::vector<int> needed_vertex_ids(needed_vertex_set->begin(), needed_vertex_set->end());

                getPicPathForVertexIds(needed_vertex_ids, vertex_to_text, vertex_to_index);
                cout << vertex_to_text.size() << endl;
                cout << "21" << endl;
                std::vector<uint8_t> score_list;

                int ret_size = predictAll_baseline("21", vertex_to_text, score_list);

                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 2)
                    {
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                oracleRecord[qnode][C_idx] = score_list[index_it->second];
                            }
                        }
                    }
                }
            }
            else if (model_key == "3")
            {
                std::vector<std::string> &vertex_to_text = all_string_inputs[3];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[3];
                std::vector<int> needed_vertex_ids(needed_vertex_set->begin(), needed_vertex_set->end());
                std::vector<uint8_t> score_list;

                buildCompactVectorMapping_parler(needed_vertex_ids, vertex_to_text, vertex_to_index);
                cout << "31" << endl;
                int ret_size = predictAll_baseline("31", vertex_to_text, score_list);
                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 3)
                    {
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                oracleRecord[qnode][C_idx] = score_list[index_it->second];
                            }
                        }
                    }
                }
            }
            else if (model_key == "4")
            {
                int maxLength = 512;
                std::vector<std::string> &vertex_to_text = all_string_inputs[4];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[4];
                std::vector<int> needed_vertex_ids(needed_vertex_set->begin(), needed_vertex_set->end());

                getProductTextsForVertexIds_parler(needed_vertex_ids, vertex_to_text, vertex_to_index, maxLength);
                cout << vertex_to_text.size() << endl;
                cout << "41" << endl;
                std::vector<uint8_t> score_list;
                int ret_size = predictAll_baseline("41", vertex_to_text, score_list);
                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 4)
                    {
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                oracleRecord[qnode][C_idx] = score_list[index_it->second];
                            }
                        }
                    }
                }
            }
            else
            {
                throw std::runtime_error("Unknown model key in buildModelInputForBatch: " + model_key);
            }
            return true;
        }

        std::vector<uint8_t> CandidateSpace::predictBatch_with_ONNX_roberta_oracle(const std::string &model_key,
                                                                                   const std::vector<std::string> &inputs)
        {
            size_t batch_size = inputs.size();
            if (batch_size == 0)
                return {};

            Ort::Session *session = getSessionForKey(model_key);
            auto *tokenizer = getTokenizerForKey(model_key);

            if (!session || !tokenizer)
            {
                std::cerr << "[ONNX] Resources missing for: " << model_key << std::endl;
                return std::vector<uint8_t>(batch_size, false);
            }

            const int64_t MAX_LEN = 128;

            // ===== RoBERTa special token ids (来自你 collect 脚本的结果) =====
            const int64_t CLS_ID = 0; // <s>
            const int64_t SEP_ID = 2; // </s>
            const int64_t PAD_ID = 1; // <pad>

            // 1. 预分配连续内存
            std::vector<int64_t> flat_input_ids(batch_size * MAX_LEN, PAD_ID);
            std::vector<int64_t> flat_attention_mask(batch_size * MAX_LEN, 0);

#pragma omp parallel for schedule(dynamic) if (batch_size > 1)
            for (int i = 0; i < (int)batch_size; ++i)
            {
                // ⚠️ 关键：必须是不带 special tokens 的 raw ids
                std::vector<int> raw_ids = tokenizer->Encode(inputs[i]);
                // 这里假设你的 tokenizer->Encode() 等价于 add_special_tokens = false
                // （与你前面对 distilbert 的验证一致）

                size_t offset = i * MAX_LEN;
                size_t max_content_len = MAX_LEN - 2;
                size_t current_len = std::min(raw_ids.size(), max_content_len);

                // <s> tokens </s>
                flat_input_ids[offset] = CLS_ID;
                for (size_t j = 0; j < current_len; ++j)
                {
                    flat_input_ids[offset + 1 + j] = static_cast<int64_t>(raw_ids[j]);
                }
                flat_input_ids[offset + 1 + current_len] = SEP_ID;

                // attention mask
                size_t valid_len = 2 + current_len;
                for (size_t j = 0; j < valid_len; ++j)
                {
                    flat_attention_mask[offset + j] = 1;
                }
            }

            // 2. 构造 Tensor（注意：只有 2 个输入！）
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            std::vector<int64_t> shape = {static_cast<int64_t>(batch_size), MAX_LEN};

            const char *input_names[] = {"input_ids", "attention_mask"};

            const char *output_names[] = {"logits"};

            std::vector<Ort::Value> input_tensors;
            input_tensors.reserve(2);

            input_tensors.emplace_back(
                Ort::Value::CreateTensor<int64_t>(mem_info, flat_input_ids.data(), flat_input_ids.size(), shape.data(), shape.size()));

            input_tensors.emplace_back(
                Ort::Value::CreateTensor<int64_t>(mem_info, flat_attention_mask.data(), flat_attention_mask.size(), shape.data(), shape.size()));

            try
            {
                auto outputs = session->Run(Ort::RunOptions {nullptr}, input_names, input_tensors.data(), 2, output_names, 1);

                float *logits_ptr = outputs[0].GetTensorMutableData<float>();
                std::vector<uint8_t> results(batch_size);

                // ===== RoBERTa: 3-class [neg, neu, pos] =====
                for (size_t i = 0; i < batch_size; ++i)
                {
                    float *sample_logits = logits_ptr + (i * 3);
                    float logit_neg = sample_logits[0];
                    float logit_pos = sample_logits[2];

                    // oracle 判定：是否为正面情感
                    results[i] = (logit_pos > logit_neg);
                }

                return results;
            }
            catch (const Ort::Exception &e)
            {
                std::cerr << "[ONNX Error] " << e.what() << std::endl;
                return std::vector<uint8_t>(batch_size, false);
            }
        }

        bool CandidateSpace::buildModelInputForBatch_parler(std::string model_key)
        {
            all_string_inputs.resize(5);
            all_string_idx.resize(5);

            if (model_key == "3")
            {
                std::vector<std::string> &vertex_to_text = all_string_inputs[3];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[3];
                std::unordered_set<int> needed_vertex_set;
                for (auto &pair : query_->predicateMap)
                {
                    int u = pair.first;
                    int model_key_int = getModelKey(u);
                    if (model_key_int == 3)
                    {
                        for (int v_idx = 0; v_idx < GetCandidateSetSize(u); ++v_idx)
                        {
                            int v = GetCandidate(u, v_idx);
                            needed_vertex_set.insert(v);
                        }
                    }
                }
                std::vector<int> needed_vertex_ids(needed_vertex_set.begin(), needed_vertex_set.end());
                buildCompactVectorMapping_parler(needed_vertex_ids, vertex_to_text, vertex_to_index);
                cout << "30" << endl;
                std::vector<float> score_list;
                //saveToFile(vertex_to_text, "debug_vertex_to_text_model3.txt");
                int ret_size = predictAll("30", vertex_to_text, score_list);
                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 3)
                    {
                        std::unordered_map<int, std::pair<double, TreeNode *>> &node_map_inner = node_map[qnode];
                        std::multimap<double, int, std::greater<double>> &map_score_inner = map_Score[qnode];
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            double score = -1.0;
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                score = score_list[index_it->second];
                            }
                            node_map_inner[C_idx].first = score;
                            map_score_inner.insert({score, C_idx});
                        }
                    }
                }
            }
            else if (model_key == "4")
            {
                int maxLength = 512;
                std::vector<std::string> &vertex_to_text = all_string_inputs[4];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[4];
                std::unordered_set<int> needed_vertex_set;
                for (auto &pair : query_->predicateMap)
                {
                    int u = pair.first;
                    int model_key_int = getModelKey(u);
                    if (model_key_int == 4)
                    {
                        for (int v_idx = 0; v_idx < GetCandidateSetSize(u); ++v_idx)
                        {
                            int v = GetCandidate(u, v_idx);
                            needed_vertex_set.insert(v);
                        }
                    }
                }
                std::vector<int> needed_vertex_ids(needed_vertex_set.begin(), needed_vertex_set.end());

                getProductTextsForVertexIds_parler(needed_vertex_ids, vertex_to_text, vertex_to_index, maxLength);
                cout << "10,2" << endl;
                std::vector<float> score_list;
                int ret_size = predictAll("40", vertex_to_text, score_list);
                // std::cout << "score_list size = " << score_list.size() << std::endl;
                // for (int i = 0; i < std::min(10, (int)score_list.size()); ++i)
                //{
                //     std::cout << "score_list[" << i << "] = " << score_list[i] << std::endl;
                // }
                for (int qnode = 0; qnode < match_order.size(); qnode++)
                {
                    int model_key_int = getModelKey(qnode);
                    if (model_key_int == 4)
                    {
                        std::unordered_map<int, std::pair<double, TreeNode *>> &node_map_inner = node_map[qnode];
                        std::multimap<double, int, std::greater<double>> &map_score_inner = map_Score[qnode];
                        for (int C_idx = 0; C_idx < GetCandidateSetSize(qnode); ++C_idx)
                        {
                            int C = GetCandidate(qnode, C_idx);
                            double score = -1.0;
                            auto index_it = vertex_to_index.find(C);
                            if (index_it != vertex_to_index.end())
                            {
                                score = score_list[index_it->second];
                            }
                            node_map_inner[C_idx].first = score;
                            map_score_inner.insert({score, C_idx});
                        }
                    }
                }
            }
            else
            {
                throw std::runtime_error("Unknown model key in buildModelInputForBatch_parler: " + model_key);
            }
            return true;
        }
        std::vector<uint8_t> CandidateSpace::predictBatch_with_ONNX_bert(const std::string &model_key, const std::vector<std::string> &inputs)
        {
            size_t batch_size = inputs.size();
            if (batch_size == 0)
                return {};

            Ort::Session *session = getSessionForKey(model_key);
            auto *tokenizer = getTokenizerForKey(model_key);

            if (!session || !tokenizer)
            {
                std::cerr << "[ONNX] Resources missing for: " << model_key << std::endl;
                return std::vector<uint8_t>(batch_size, false);
            }

            const int64_t MAX_LEN = 128;
            const int64_t CLS_ID = 101;
            const int64_t SEP_ID = 102;
            const int64_t PAD_ID = 0;

            // 1. 预分配连续内存 (std::vector 初始化是顺序的，这里保持原样)
            std::vector<int64_t> flat_input_ids(batch_size * MAX_LEN, PAD_ID);
            std::vector<int64_t> flat_attention_mask(batch_size * MAX_LEN, 0);
            std::vector<int64_t> flat_token_type_ids(batch_size * MAX_LEN, 0); // 始终为 0，只需初始化一次

#pragma omp parallel for schedule(dynamic) if (batch_size > 1)
            // 2. 并行分词与填充 (效率提升的核心)
            // 分词（Encode）是 CPU 密集型操作，且各样本独立，非常适合 OpenMP
            for (int i = 0; i < (int)batch_size; ++i)
            {
                // 每一个线程处理一个 inputs[i]
                std::vector<int> raw_ids = tokenizer->Encode(inputs[i]);

                size_t offset = i * MAX_LEN;
                size_t max_content_len = MAX_LEN - 2;
                size_t current_len = std::min(raw_ids.size(), max_content_len);

                // 写入数据
                flat_input_ids[offset] = CLS_ID;
                // 使用 std::copy 或 memcpy 通常比手动循环略快
                for (size_t j = 0; j < current_len; ++j)
                {
                    flat_input_ids[offset + 1 + j] = static_cast<int64_t>(raw_ids[j]);
                }
                flat_input_ids[offset + 1 + current_len] = SEP_ID;

                // 填充 Attention Mask (一次性填充连续段)
                size_t valid_len = 2 + current_len;
                for (size_t j = 0; j < valid_len; ++j)
                {
                    flat_attention_mask[offset + j] = 1;
                }
            }

            // 3. 准备 Tensor (Zero-copy)
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> shape = {static_cast<int64_t>(batch_size), MAX_LEN};

            const char *input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
            const char *output_names[] = {"logits"};

            std::vector<Ort::Value> input_tensors;
            input_tensors.reserve(3);
            input_tensors.emplace_back(
                Ort::Value::CreateTensor<int64_t>(mem_info, flat_input_ids.data(), flat_input_ids.size(), shape.data(), shape.size()));
            input_tensors.emplace_back(
                Ort::Value::CreateTensor<int64_t>(mem_info, flat_attention_mask.data(), flat_attention_mask.size(), shape.data(), shape.size()));
            input_tensors.emplace_back(
                Ort::Value::CreateTensor<int64_t>(mem_info, flat_token_type_ids.data(), flat_token_type_ids.size(), shape.data(), shape.size()));

            try
            {
                // 4. 推理 (这一步的耗时由 GPU/ORT 决定)
                auto outputs = session->Run(Ort::RunOptions {nullptr}, input_names, input_tensors.data(), 3, output_names, 1);

                // 5. 后处理优化：快速解析
                float *logits_ptr = outputs[0].GetTensorMutableData<float>();
                std::vector<uint8_t> results(batch_size);

                // 如果 batch 非常大，后处理也可以并行，但通常 SST2 只有 2 列，直接跑很快
                for (size_t i = 0; i < batch_size; ++i)
                {
                    float *sample_logits = logits_ptr + (i * 2); // SST2 确定是 2 类
                    results[i] = (sample_logits[1] > sample_logits[0]);
                }
                return results;
            }
            catch (const Ort::Exception &e)
            {
                std::cerr << "[ONNX Error] " << e.what() << std::endl;
                return std::vector<uint8_t>(batch_size, false);
            }
        }

        std::string CandidateSpace::buildModelInputForPair_type1(int u, int cand_idx, std::string model_key_all)
        {
            std::string proxy_or_oracle(1, model_key_all.back());
            std::string model_type = model_key_all.substr(0, model_key_all.length() - 1);
            int cand_id = GetCandidate(u, cand_idx);
            if (model_type == "0")
            {
                int idx = all_string_idx[0][cand_id];
                return all_string_inputs[0][idx];
            }
            else if (model_type == "1")
            {
                int idx = all_string_idx[1][cand_id];
                return all_string_inputs[1][idx];
            }
            else if (model_type == "2")
            {
                int idx = all_string_idx[2][cand_id];
                return all_string_inputs[2][idx];
            }
            else if (model_type == "3")
            {
                //std::cout << (all_string_idx[3].find(cand_id) != all_string_idx[3].end()) << std::endl;
                int idx = all_string_idx[3][cand_id];
                return all_string_inputs[3][idx];
            }
            else if (model_type == "4")
            {
                // if (cand_id % 2 == 0)
                //{
                //     return std::string("I support Trump");
                // }
                // else
                //{
                //     return std::string("I don't support Trump");
                // }
                int idx = all_string_idx[4][cand_id];
                return all_string_inputs[4][idx];
            }
            else
            {
                throw std::runtime_error("Unknown model type in buildModelInputForPair_type1: " + model_type);
            }
        }

        std::string CandidateSpace::buildModelInputForPair(int u, int cand_idx, std::string model_key_all)
        {
            std::string proxy_or_oracle(1, model_key_all.back());
            std::string model_type = model_key_all.substr(0, model_key_all.length() - 1);
            if (model_type == "0")
            {
                return buildModelInputForPair_type1(u, cand_idx, model_key_all);
            }
            else if (model_type == "1")
            {
                return buildModelInputForPair_type1(u, cand_idx, model_key_all);
            }
            else if (model_type == "2")
            {
                return buildModelInputForPair_type1(u, cand_idx, model_key_all);
            }
            else if (model_type == "3")
            {
                return buildModelInputForPair_type1(u, cand_idx, model_key_all);
            }
            else if (model_type == "4")
            {
                return buildModelInputForPair_type1(u, cand_idx, model_key_all);
            }
            else
            {
                throw std::runtime_error("Unknown model type in buildModelInputForPair: " + model_type);
            }
            return "";
        }

        // 临时测试函数
        bool CandidateSpace::getOracle(int u, int v)
        {
            std::cout << "=== ONNX Runtime + Tokenizer Test ===\n";

            const std::string model_path = modelFile("distilbert-base-uncased-finetuned-sst-2-english", "model.onnx");
            const std::string tokenizer_path = modelFile("distilbert-base-uncased-finetuned-sst-2-english", "tokenizer.json");

            std::string tok_json = readFileToString(tokenizer_path);
            auto tokenizer = Tokenizer::FromBlobJSON(tok_json);
            if (!tokenizer)
            {
                std::cerr << "Failed to create tokenizer\n";
                return false;
            }

            std::string text = "I dislike this movie!";
            std::vector<int> ids = tokenizer->Encode(text);
            std::vector<int64_t> input_ids(ids.begin(), ids.end());
            std::vector<int64_t> attention_mask(input_ids.size(), 1);

            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sentiment");
            Ort::SessionOptions session_options;
            Ort::Session session(env, model_path.c_str(), session_options);

            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
            std::array<int64_t, 2> shape = {1, static_cast<int64_t>(input_ids.size())};

            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, input_ids.data(), input_ids.size(), shape.data(), shape.size());
            Ort::Value mask_tensor =
                Ort::Value::CreateTensor<int64_t>(mem_info, attention_mask.data(), attention_mask.size(), shape.data(), shape.size());

            const char *input_names[] = {"input_ids", "attention_mask"};
            const char *output_names[] = {"logits"};

            std::array<Ort::Value, 2> inputs = {std::move(input_tensor), std::move(mask_tensor)};
            std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions {nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);

            float *logits_ptr = outputs[0].GetTensorMutableData<float>();
            auto shape_out = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            size_t num_classes = shape_out.back();

            std::vector<float> logits(logits_ptr, logits_ptr + num_classes);
            std::cout << "Logits: ";
            print_vec_float(logits);

            return true;
        }

        void CandidateSpace::startOracleWorker()
        {
            oracle_worker_stop.store(false);
            oracle_no_more_requests.store(false);

            oracle_worker = std::thread(
                [this]()
                {
                    while (true)
                    {
                        std::vector<std::tuple<int, int, std::string>> batch;

                        {
                            // 🔒 进入临界区
                            std::unique_lock<std::mutex> lk(this->oracle_mutex);

                            // 等待：有任务、收尾信号、或停止信号
                            oracle_cv.wait_for(
                                lk, std::chrono::milliseconds(this->oracle_max_wait_ms), [this]()
                                { return !this->oracle_pending.empty() || this->oracle_no_more_requests.load() || this->oracle_worker_stop.load(); });

                            // 🧩 退出条件：stop信号且无任务
                            if (oracle_worker_stop.load() && oracle_pending.empty())
                            {
                                break;
                            }

                            // 🧱 组建 batch
                            while (!oracle_pending.empty() && (int)batch.size() < this->oracle_batch_size)
                            {
                                batch.push_back(oracle_pending.back());
                                oracle_pending.pop_back();
                            }

                            // 🧩 收尾逻辑：
                            // 如果主线程已声明“没有新请求”，
                            // 且队列中剩余任务 < batch_size（尾批），立即取完
                            if (this->oracle_no_more_requests.load())
                            {
                                if (!oracle_pending.empty() && (int)oracle_pending.size() < this->oracle_batch_size)
                                {
                                    while (!oracle_pending.empty())
                                    {
                                        batch.push_back(oracle_pending.back());
                                        oracle_pending.pop_back();
                                    }
                                }
                            }

                            // ⚙️ 没取到任务则继续等
                            if (batch.empty())
                                continue;

                            in_flight_batches.fetch_add(1);
                        } // 🔓 离开临界区

                        // prinf("[OracleWorker] processing batch of size %zu\n", batch.size());
                        //  🚀 执行 Oracle 批处理（同步）
                        try
                        {
                            this->runOracleBatchSync(batch);
                        }
                        catch (const std::exception &e)
                        {
                            std::cerr << "[OracleWorker] exception: " << e.what() << std::endl;
                        }
                        catch (...)
                        {
                            std::cerr << "[OracleWorker] unknown exception in batch run.\n";
                        }

                        // 📣 批次完成后更新计数并通知可能在等的 waitOracleComplete()
                        {
                            std::scoped_lock<std::mutex> lk(this->oracle_mutex);
                            in_flight_batches.fetch_sub(1);
                            oracle_cv.notify_all();
                        }
                    }

                    // 🧹 worker 即将退出时，做最后一次尾批 flush（双保险）
                    std::vector<std::tuple<int, int, std::string>> leftover;
                    {
                        std::scoped_lock<std::mutex> lk(this->oracle_mutex);
                        while (!oracle_pending.empty())
                        {
                            leftover.push_back(oracle_pending.back());
                            oracle_pending.pop_back();
                        }
                    }
                    if (!leftover.empty())
                    {
                        std::cerr << "[OracleWorker] flushing leftover tail batch: " << leftover.size() << " samples\n";
                        try
                        {
                            this->runOracleBatchSync(leftover);
                        }
                        catch (...)
                        {
                            std::cerr << "[OracleWorker] exception in tail flush\n";
                        }
                    }

                    {
                        std::scoped_lock<std::mutex> lk(this->oracle_mutex);
                        oracle_cv.notify_all(); // 唤醒 waitOracleComplete()
                    }
                });
        }

        inline void CandidateSpace::stopOracleWorkerAndFlush()
        {
            // 表示后续不会再有新请求
            oracle_no_more_requests.store(true);

            // 请求 worker 停止并通知
            oracle_worker_stop.store(true);
            oracle_cv.notify_all();

            // 等待线程退出并 flush 剩余（worker 在退出前会 flush）
            if (oracle_worker.joinable())
                oracle_worker.join();

            // 为保险，再做一次同步检查并 flush（主线程执行）
            std::vector<std::tuple<int, int, std::string>> leftover;
            {
                std::scoped_lock<std::mutex> lk(oracle_mutex);
                if (!oracle_pending.empty())
                    leftover.swap(oracle_pending);
            }
            if (!leftover.empty())
            {
                runOracleBatchSync(leftover);
                {
                    std::scoped_lock<std::mutex> lk2(oracle_mutex);
                    oracle_cv.notify_all();
                }
            }
        }

        void CandidateSpace::runOracleSingleReq(int u, int cand_idx, std::string model_key)
        {
            int v_id = candidate_set_[u][cand_idx];
            std::string model_key_all = std::to_string(query_->predicateMap[u]) + model_key;

            std::vector<std::tuple<int, int, std::string>> batch;
            batch.emplace_back(u, cand_idx, model_key_all);

            {
                std::scoped_lock<std::mutex> lk(oracle_mutex);

                oracle_task_map[v_id].emplace_back(u, cand_idx, model_key_all);

                if (!oracle_pending_nodes[model_key_all].count(v_id))
                {
                    oracle_pending_nodes[model_key_all].insert(v_id);
                }
            }

            // all_string_inputs.resize(5);
            // all_string_idx.resize(5);
            std::vector<int> needed_vertex_ids {v_id};
            int real_model_key = getModelKey(u);
            switch (real_model_key)
            {
            case 0:
            {
                std::vector<std::string> &vertex_to_text = all_string_inputs[0];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[0];
                buildCompactVectorMapping(needed_vertex_ids, vertex_to_text, vertex_to_index);
                break;
            }
            case 1:
            {
                int maxLength = 512;
                std::vector<std::string> &vertex_to_text = all_string_inputs[1];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[1];
                getProductTextsForVertexIds(needed_vertex_ids, vertex_to_text, vertex_to_index, maxLength);
                break;
            }
            case 2:
            {
                std::vector<std::string> &vertex_to_text = all_string_inputs[2];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[2];
                getPicPathForVertexIds(needed_vertex_ids, vertex_to_text, vertex_to_index);
                break;
            }
            case 3:
            {
                std::vector<std::string> &vertex_to_text = all_string_inputs[3];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[3];
                buildCompactVectorMapping_parler(needed_vertex_ids, vertex_to_text, vertex_to_index);
                break;
            }
            case 4:
            {
                int maxLength = 512;
                std::vector<std::string> &vertex_to_text = all_string_inputs[4];
                std::unordered_map<int, int> &vertex_to_index = all_string_idx[4];
                getProductTextsForVertexIds_parler(needed_vertex_ids, vertex_to_text, vertex_to_index, maxLength);
                break;
            }
            default:
                throw std::runtime_error("Unknown model key in runOracleSingleReq: " + std::to_string(real_model_key));
            }

            runOracleBatchSync(batch);
        }

        // =========================================================
        // 函数：runOracleBatchSync
        // 功能：执行当前批次的 Oracle 模型推理（批量）
        //       每个 data 节点只推理一次，将结果共享给所有匹配 (u, cand_idx)。
        // 调用时机：worker 线程批量执行时，或 flush 阶段（同步执行）。
        // =========================================================
        void CandidateSpace::runOracleBatchSync(const std::vector<std::tuple<int, int, std::string>> &batch)
        {
            if (batch.empty())
                return;

            // ===============================
            // 1️⃣ 预处理阶段：过滤历史缓存 + 按模型分组
            // ===============================

            // grouped_batches: key=模型名 / 模型类别 -> (u, cand_idx) 列表
            std::unordered_map<std::string, std::vector<std::pair<int, int>>> grouped_batches;

            {
                std::scoped_lock<std::mutex> lk(oracle_mutex);

                for (const auto &p : batch)
                {
                    int u = get<0>(p);
                    int cand_idx = get<1>(p);
                    std::string key = get<2>(p);

                    // 检查缓存：如果 oracleRecord 已有结果则跳过
                    // auto it_u = oracleRecord.find(u);
                    // if (it_u != oracleRecord.end() && it_u->second.count(cand_idx))
                    //    continue;

                    auto it_u = oracleRecord.find(u);
                    if (it_u != oracleRecord.end() && it_u->second.count(cand_idx))
                    {
                        // [新增] 必须先计算出 v_id，因为 task_map 是用 v_id 做 key 的
                        int v_id = candidate_set_[u][cand_idx];

                        // [新增] 核心修复：手动清理任务映射，防止 waitOracleComplete 死锁
                        auto it_task = oracle_task_map.find(v_id);
                        if (it_task != oracle_task_map.end())
                        {
                            // 既然缓存里已经有结果了，理论上应该把结果同步给等待这个 v_id 的其他任务（可选）
                            // 但为了解死锁，最重要的是必须把它 erase 掉！
                            oracle_task_map.erase(it_task);
                        }

                        // [新增] 保持状态一致性，清理 pending_nodes
                        oracle_pending_nodes[key].erase(v_id);

                        continue; // 现在可以安全跳过了
                    }

                    grouped_batches[key].push_back({u, cand_idx});
                }
            }

            if (grouped_batches.empty())
            {
                // 全部命中缓存，无需执行模型
                return;
            }

            // ===============================
            // 2️⃣ 对每个模型执行批处理
            // ===============================
            for (auto &[model_key, pairs] : grouped_batches)
            {
                if (pairs.empty())
                    continue;

                Ort::Session *session = getSessionForKey(model_key); // 用户需实现 session 管理
                // if (!session)
                //{
                //     std::cerr << "[Oracle] Warning: Session for " << model_key << " not found, skip batch\n";
                //     continue;
                // }

                // 构造输入文本
                std::vector<int> v_ids;
                std::vector<std::string> texts;
                v_ids.reserve(pairs.size());
                texts.reserve(pairs.size());

                for (auto &[u, cand_idx] : pairs)
                {
                    int v_id = candidate_set_[u][cand_idx];
                    v_ids.push_back(v_id);
                    texts.push_back(buildModelInputForPair(u, cand_idx, model_key));
                }

                // ===============================
                // 3️⃣ 调用 ONNX 批预测
                // ===============================
                std::vector<uint8_t> preds;
                try
                {
                    //texts[0] = std::string("I will support Trump."); // debug
                    //texts[1] = std::string("I hate Trump.");
                    preds = predictBatch_with_ONNX(model_key, texts);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "[Oracle] Batch failed for model=" << model_key << " reason=" << e.what() << std::endl;
                    continue;
                }

                if (preds.size() != pairs.size())
                {
                    std::cerr << "[Oracle] Warning: output size mismatch for model=" << model_key << " expect=" << pairs.size()
                              << " got=" << preds.size() << std::endl;
                    continue;
                }
                // printf("[Oracle] model=%s batch_size=%zu completed.\n",
                //        model_key.c_str(), pairs.size());
                //  ===============================
                //  4️⃣ 回填结果 + 清理任务映射
                //  ===============================
                std::scoped_lock<std::mutex> lk(oracle_mutex);
                // printf("Backfilling %zu results...\n", pairs.size());
                for (size_t i = 0; i < pairs.size(); ++i)
                {
                    int u = pairs[i].first;
                    int cand_idx = pairs[i].second;
                    int v_id = v_ids[i];
                    bool label = preds[i];

                    oracleRecord[u][cand_idx] = label;
                    // oracleRecord[u][cand_idx] = oracleRecord_[u][cand_idx];
                    // printf("Oracle result for (u=%d, cand_idx=%d, v_id=%d): %d\n",
                    //       u, cand_idx, v_id, label ? 1 : 0);
                    // 更新任务映射和去重表
                    auto it = oracle_task_map.find(v_id);
                    if (it != oracle_task_map.end())
                    {
                        for (auto &[uu, cc, str] : it->second)
                        {
                            oracleRecord[uu][cc] = label;
                            // oracleRecord[uu][cc] = oracleRecord_[uu][cc];
                        }
                        oracle_task_map.erase(it);
                    }
                    oracle_pending_nodes[model_key].erase(v_id);
                }
            }
        }

        // =========================================================
        // 函数：enqueueOracleTask
        // 功能：将 (u, cand_idx) 对应的匹配任务加入 Oracle 队列。
        //       若该 data 节点 v 已存在任务，则附加引用而不重复入队。
        // 调用时机：匹配阶段或采样阶段，需要 Oracle 结果时。
        // =========================================================
        void CandidateSpace::enqueueOracleTask(int u, int cand_idx, string model_key)
        {
            int v_id = candidate_set_[u][cand_idx];
            // printf("Enqueue Oracle task for (u=%d, cand_idx=%d, v_id=%d)\n", u, cand_idx, v_id);
            string model_key_all = std::to_string(query_->predicateMap[u]) + model_key;

            {
                std::scoped_lock<std::mutex> lk(oracle_mutex);

                // 若已有结果则直接返回
                if (model_key == "1" && oracleRecord[u].count(cand_idx))
                    return;

                else if (model_key == "0" && node_map[u].count(cand_idx))
                    return;

                // 若该节点已有映射，附加引用（也notify，防止极端时序遗漏）
                if (oracle_task_map.count(v_id))
                {
                    oracle_task_map[v_id].push_back({u, cand_idx, model_key_all});
                    // wake worker just in case (low cost)
                    oracle_cv.notify_one();
                    return;
                }

                // 否则创建新的任务组
                oracle_task_map[v_id].push_back({u, cand_idx, model_key_all});

                // 去重表检查并入队（仅第一次入队）
                if (!oracle_pending_nodes[model_key_all].count(v_id))
                {
                    oracle_pending_nodes[model_key_all].insert(v_id);
                    oracle_pending.emplace_back(std::make_tuple(u, cand_idx, model_key_all));
                }
            } // unlock

            // notify outside lock to reduce contention (仍可 notify while持锁也行)
            oracle_cv.notify_one();
        }

        tokenizers::Tokenizer *CandidateSpace::getTokenizerForKey(const std::string &key)
        {
            // 1. 定义 Key 到 路径 的映射表
            // 使用 static const 保证只在第一次调用时初始化，提高效率
            static const std::map<std::string, std::string> keyToPathMap = {
                {"00", modelFile("distilbert-base-uncased-finetuned-sst-2-english", "tokenizer.json")},
                {"01", modelFile("bert-large-uncased-sst2", "tokenizer.json")},
                {"10", modelFile("deberta-v3-base-mnli-fever-anli", "tokenizer.json")},
                {"11", modelFile("deberta-v2-xlarge-mnli", "tokenizer.json")},
                {"20", modelFile("clip-vit-base-patch32", "tokenizer.json")},
                {"21", modelFile("clip-vit-large-patch14", "tokenizer.json")},
                {"30", modelFile("distilbert-base-uncasedv1-finetuned-twitter-sentiment", "tokenizer.json")},
                {"31", modelFile("twitter-roberta-base-sentiment-latest", "tokenizer.json")},
                {"40", modelFile("deberta-v3-base-mnli-fever-anli", "tokenizer.json")},
                {"41", modelFile("deberta-v2-xlarge-mnli", "tokenizer.json")}};

            // 2. 检查传入的 key 是否有效
            auto pathIt = keyToPathMap.find(key);
            if (pathIt == keyToPathMap.end())
            {
                std::cerr << "[Error] Unknown key for Tokenizer: " << key << std::endl;
                return nullptr; // 或者抛出异常
            }

            // 3. 缓存池：存储已经加载过的 Tokenizer 指针
            // 这里的 Key 是 "00", "01" 等，Value 是加载好的对象指针
            // static std::map<std::string, Tokenizer *> tokenizerCache;

            // 4. 如果缓存中没有，则加载；如果有了，直接返回
            if (tokenizerCache.find(key) == tokenizerCache.end())
            {
                std::string filePath = pathIt->second;

                // ===========================================================
                // 【注意】：这里需要根据你实际的 Tokenizer 类构造函数进行修改
                // 情况 A：如果通过构造函数加载文件
                std::ifstream t(filePath);
                if (!t.is_open())
                {
                    std::cerr << "[Error] Failed to open tokenizer file: " << filePath << std::endl;
                    return nullptr;
                }
                std::string jsonContent((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
                auto tokenizerPtr = tokenizers::Tokenizer::FromBlobJSON(jsonContent);
                tokenizers::Tokenizer *newTokenizer = tokenizerPtr.release();
                // tokenizers::Tokenizer *newTokenizer = new Tokenizer(filePath);

                // 情况 B：如果是 HuggingFace tokenizers 常见的静态方法加载 (例如)
                // Tokenizer* newTokenizer = Tokenizer::FromFile(filePath);
                // ===========================================================

                if (newTokenizer)
                {
                    tokenizerCache[key] = newTokenizer;
                    std::cout << "Loaded tokenizer for key " << key << " from " << filePath << std::endl;
                }
                else
                {
                    std::cerr << "[Error] Failed to load tokenizer file: " << filePath << std::endl;
                    return nullptr;
                }
            }

            return tokenizerCache[key];
        }

        /**
         * @brief 使用 IOBinding 优化后的 Parler 蕴含推理 Oracle
         * 输出：vector<float> 代表 "Entailment" (蕴含) 的置信度 [0.0, 1.0]
         * 针对 DeBERTa v3/v2 ONNX 模型
         */
        std::vector<float> CandidateSpace::predictBatch_with_ONNX_score_textClass2(const std::string &model_key,
                                                                                   const std::vector<std::string> &inputs)
        {
            size_t num_inputs = inputs.size();
            if (num_inputs == 0)
                return {};

            Ort::Session *session = getSessionForKey(model_key);
            auto *tokenizer = getTokenizerForKey(model_key);
            if (!session || !tokenizer)
                return std::vector<float>(num_inputs, 0.0f);

            constexpr int64_t MAX_LEN = 256;
            constexpr int64_t CLS_ID = 1;
            constexpr int64_t SEP_ID = 2;
            constexpr int64_t PAD_ID = 0;

            const std::string hypothesis = "The author expresses a positive opinion about Donald Trump.";

            // 预编码 hypothesis
            std::vector<int> h_ids = tokenizer->Encode(hypothesis);
            size_t h_len = std::min(h_ids.size(), (size_t)(MAX_LEN - 3));

            std::vector<int64_t> input_ids(num_inputs * MAX_LEN, PAD_ID);
            std::vector<int64_t> attention_mask(num_inputs * MAX_LEN, 0);

#pragma omp parallel for schedule(dynamic) if (num_inputs > 1)
            for (int i = 0; i < (int)num_inputs; ++i)
            {
                std::vector<int> p_ids = tokenizer->Encode(inputs[i]);
                size_t p_len = std::min(p_ids.size(), (size_t)(MAX_LEN - h_len - 3));

                size_t offset = i * MAX_LEN;
                size_t curr = offset;

                input_ids[curr++] = CLS_ID;

                for (size_t j = 0; j < p_len; ++j)
                    input_ids[curr++] = p_ids[j];

                input_ids[curr++] = SEP_ID;

                for (size_t j = 0; j < h_len; ++j)
                    input_ids[curr++] = h_ids[j];

                input_ids[curr++] = SEP_ID;

                // attention_mask
                for (size_t j = offset; j < curr; ++j)
                    attention_mask[j] = 1;
            }

            Ort::IoBinding io_binding(*session);
            std::vector<int64_t> shape = {static_cast<int64_t>(num_inputs), MAX_LEN};
            Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            io_binding.BindInput("input_ids", Ort::Value::CreateTensor<int64_t>(cpu_mem, input_ids.data(), input_ids.size(), shape.data(), 2));
            io_binding.BindInput("attention_mask",
                                 Ort::Value::CreateTensor<int64_t>(cpu_mem, attention_mask.data(), attention_mask.size(), shape.data(), 2));
            io_binding.BindOutput("logits", cpu_mem);

            std::vector<float> results(num_inputs, 0.0f);

            try
            {
                session->Run(Ort::RunOptions {nullptr}, io_binding);
                auto outputs = io_binding.GetOutputValues();
                const float *logits = outputs[0].GetTensorData<float>();

                // softmax + Entailment 概率（第 0 列）
#pragma omp parallel for
                for (int i = 0; i < (int)num_inputs; ++i)
                {
                    const float *l = &logits[i * 3]; // MNLI logits: [Entailment, Neutral, Contradiction]
                    float max_logit = std::max({l[0], l[1], l[2]});
                    float sum_exp = std::exp(l[0] - max_logit) + std::exp(l[1] - max_logit) + std::exp(l[2] - max_logit);
                    results[i] = std::exp(l[0] - max_logit) / sum_exp; // 第0列 Entailment
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "[DeBERTa-MNLI Score Error] " << e.what() << std::endl;
                return std::vector<float>(num_inputs, 0.0f);
            }

            return results;
        }

        // 返回分数版本
        std::vector<float> CandidateSpace::predictBatch_with_ONNX_score(const std::string &model_key, const std::vector<std::string> &inputs)
        {
            auto now = std::chrono::system_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(now);
            // std::cout << std::put_time(std::localtime(&now_c), "%F %T") << " ";
            if (model_key == "00")
            {
                cout << "[ONNX Score] Using DistilBERT model for scoring." << endl;
                return predictBatch_with_ONNX_score_bert(model_key, inputs);
            }
            else if (model_key == "10")
            {
                cout << "[ONNX Score] Using DistilBART model for text classification scoring." << endl;
                // return predictBatch_with_ONNX_score_textClass(model_key, inputs);
                return predictBatch_with_ONNX_score_textClass2(model_key, inputs);
            }
            else if (model_key == "20")
            {
                cout << "[ONNX Score] Using CLIP model for picture classification scoring." << endl;
                return predictBatch_with_ONNX_score_picClass(model_key, inputs);
            }
            else if (model_key == "30")
            {
                return predictBatch_with_ONNX_score_bert(model_key, inputs);
            }
            else if (model_key == "40")
            {
                return predictBatch_with_ONNX_score_textClass2(model_key, inputs);
            }
            else
            {
                throw std::runtime_error("Unknown model key in predictBatch_with_ONNX_score: " + model_key);
            }
        }

        std::vector<float> CandidateSpace::predictBatch_with_ONNX_score_bert(const std::string &model_key, const std::vector<std::string> &inputs)
        {
            size_t batch_size = inputs.size();
            if (batch_size == 0)
                return {};

            Ort::Session *session = getSessionForKey(model_key);
            auto *tokenizer = getTokenizerForKey(model_key);
            if (!session || !tokenizer)
                return std::vector<float>(batch_size, 0.0f);

            const int64_t MAX_LEN = 128;
            std::vector<int64_t> flat_input_ids(batch_size * MAX_LEN, 0);
            std::vector<int64_t> flat_attention_mask(batch_size * MAX_LEN, 0);

            // 1. 分词（保持 OpenMP 加速）
#pragma omp parallel for schedule(static) if (batch_size > 8)
            for (int i = 0; i < (int)batch_size; ++i)
            {
                std::vector<int> raw_ids = tokenizer->Encode(inputs[i]);
                size_t offset = i * MAX_LEN;
                flat_input_ids[offset] = 101;
                size_t cur_len = std::min(raw_ids.size(), (size_t)126);
                for (size_t j = 0; j < cur_len; ++j)
                    flat_input_ids[offset + 1 + j] = static_cast<int64_t>(raw_ids[j]);
                flat_input_ids[offset + 1 + cur_len] = 102;
                for (size_t j = 0; j < cur_len + 2; ++j)
                    flat_attention_mask[offset + j] = 1;
            }

            // 2. 准备 IOBinding
            Ort::IoBinding io_binding(*session);
            std::vector<int64_t> shape = {(int64_t)batch_size, MAX_LEN};
            Ort::MemoryInfo cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            // 创建输入 Tensor 并直接绑定（ORT 会优化这里的 CPU->GPU 拷贝）
            auto input_id_tensor =
                Ort::Value::CreateTensor<int64_t>(cpu_mem_info, flat_input_ids.data(), flat_input_ids.size(), shape.data(), shape.size());
            auto mask_tensor =
                Ort::Value::CreateTensor<int64_t>(cpu_mem_info, flat_attention_mask.data(), flat_attention_mask.size(), shape.data(), shape.size());

            io_binding.BindInput("input_ids", input_id_tensor);
            io_binding.BindInput("attention_mask", mask_tensor);

            // 3. 核心优化：强制将输出绑定到 CPU 内存，由 IOBinding 托管，避免 Run 内部的额外分配
            // 这样可以直接获得结果而不需要在 Run 之后进行昂贵的 Session 状态验证
            io_binding.BindOutput("logits", cpu_mem_info);

            try
            {
                // 4. 使用 IoBinding 运行（速度极快）
                session->Run(Ort::RunOptions {nullptr}, io_binding);

                // 5. 获取结果
                auto output_values = io_binding.GetOutputValues();
                float *ptr = output_values[0].GetTensorMutableData<float>();

                std::vector<float> res(batch_size);
                for (size_t i = 0; i < batch_size; ++i)
                {
                    float n = ptr[i * 2], p = ptr[i * 2 + 1];
                    float m = std::max(n, p);
                    res[i] = std::exp(p - m) / (std::exp(n - m) + std::exp(p - m));
                }

                // --- 更靠谱的 GPU 验证：检查 Session 的 Provider 列表 ---
                static bool has_verified_provider = false;
                if (!has_verified_provider)
                {
                    std::cout << "[Check] 正在验证 Session 执行提供者..." << std::endl;
                    // 如果你之前的 GetExecutionProviderNames 报错，可以忽略此块
                    // 但如果之前的代码配置了 CUDA，这里其实已经默认在用了
                    has_verified_provider = true;
                }

                return res;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Inference error: " << e.what() << std::endl;
                return std::vector<float>(batch_size, 0.0f);
            }
        }

        /**
         * @brief 文本场景分类推理（DistilBART-MNLI）
         * 输入: premise = inputs[i]
         *       hypothesis = query_type_
         * 输出: entailment 概率
         */
        std::vector<float> CandidateSpace::predictBatch_with_ONNX_score_textClass(const std::string &model_key,
                                                                                  const std::vector<std::string> &inputs)
        {

            size_t batch_size = inputs.size();
            if (batch_size == 0 || query_type_.empty())
                return {};

            Ort::Session *session = getSessionForKey(model_key);
            auto *tokenizer = getTokenizerForKey(model_key);
            if (!session || !tokenizer)
                return std::vector<float>(batch_size, 0.0f);

            constexpr int64_t MAX_LEN = 256; // <-- 改成 256
            const int64_t PAD_ID = 1;

            // 1. 构造 Premise + Hypothesis
            std::vector<std::string> encoded_texts(batch_size);
            for (size_t i = 0; i < batch_size; ++i)
                encoded_texts[i] = inputs[i] + " </s> </s> " + query_type_;

            // 2. tokenizer 编码
            std::vector<int64_t> input_ids(batch_size * MAX_LEN, PAD_ID);
            std::vector<int64_t> attention_mask(batch_size * MAX_LEN, 0);

            for (size_t i = 0; i < batch_size; ++i)
            {
                std::vector<int> tokens = tokenizer->Encode(encoded_texts[i]);

                // 截断到 MAX_LEN
                size_t len = std::min(tokens.size(), (size_t)MAX_LEN);
                size_t offset = i * MAX_LEN;

                for (size_t j = 0; j < len; ++j)
                {
                    input_ids[offset + j] = static_cast<int64_t>(tokens[j]);
                    attention_mask[offset + j] = 1;
                }
            }

            // 3. ONNX Runtime 推理
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            int64_t shape[2] = {(int64_t)batch_size, MAX_LEN};

            Ort::Value ort_inputs[] = {Ort::Value::CreateTensor<int64_t>(mem_info, input_ids.data(), input_ids.size(), shape, 2),
                                       Ort::Value::CreateTensor<int64_t>(mem_info, attention_mask.data(), attention_mask.size(), shape, 2)};

            const char *input_names[] = {"input_ids", "attention_mask"};
            const char *output_names[] = {"logits"};

            auto outputs = session->Run(Ort::RunOptions {nullptr}, input_names, ort_inputs, 2, output_names, 1);

            float *logits = outputs[0].GetTensorMutableData<float>();

            // 4. softmax + entailment（index=2）
            std::vector<float> scores(batch_size, 0.0f);
            for (size_t i = 0; i < batch_size; ++i)
            {
                float *row = logits + i * 3; // [contradiction, neutral, entailment]

                float maxv = std::max({row[0], row[1], row[2]});
                float e0 = std::exp(row[0] - maxv);
                float e1 = std::exp(row[1] - maxv);
                float e2 = std::exp(row[2] - maxv);

                float sum = e0 + e1 + e2;
                scores[i] = e2 / sum;

                // std::cout << "LOGITS i=" << i << " : [" << row[0] << ", " << row[1] << ", " << row[2] << "]\n";
                // std::cout << "Entailment score = " << scores[i] << "\n\n";
            }

            return scores;
        }

        /*parler的蕴含推理 oracle */
        std::vector<uint8_t> CandidateSpace::predictBatch_with_ONNX_textClass2(const std::string &model_key, const std::vector<std::string> &inputs)
        {
            size_t num_inputs = inputs.size();
            if (num_inputs == 0)
                return {};

            Ort::Session *session = getSessionForKey(model_key);
            auto *tokenizer = getTokenizerForKey(model_key);
            if (!session || !tokenizer)
                return std::vector<uint8_t>(num_inputs, 0);

            constexpr int64_t MAX_LEN = 256;
            constexpr int64_t CLS_ID = 1; // 从 tokenizer 里来的
            constexpr int64_t SEP_ID = 2;
            constexpr int64_t PAD_ID = 0;

            const std::string hypothesis = "The author expresses a positive opinion about Donald Trump.";

            // 预编码 hypothesis
            std::vector<int> h_ids = tokenizer->Encode(hypothesis);
            size_t h_len = std::min(h_ids.size(), (size_t)(MAX_LEN - 3));

            std::vector<int64_t> input_ids(num_inputs * MAX_LEN, PAD_ID);
            std::vector<int64_t> attention_mask(num_inputs * MAX_LEN, 0);

#pragma omp parallel for schedule(dynamic) if (num_inputs > 1)
            for (int i = 0; i < (int)num_inputs; ++i)
            {
                std::vector<int> p_ids = tokenizer->Encode(inputs[i]);
                size_t p_len = std::min(p_ids.size(), (size_t)(MAX_LEN - h_len - 3));

                size_t offset = i * MAX_LEN;
                size_t curr = offset;

                // BERT-style: [CLS] premise [SEP] hypothesis [SEP]
                input_ids[curr++] = CLS_ID;

                for (size_t j = 0; j < p_len; ++j)
                    input_ids[curr++] = p_ids[j];

                input_ids[curr++] = SEP_ID;

                for (size_t j = 0; j < h_len; ++j)
                    input_ids[curr++] = h_ids[j];

                input_ids[curr++] = SEP_ID;

                // attention mask
                for (size_t j = offset; j < curr; ++j)
                    attention_mask[j] = 1;
            }

            Ort::IoBinding io_binding(*session);
            std::vector<int64_t> shape = {static_cast<int64_t>(num_inputs), MAX_LEN};
            Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            io_binding.BindInput("input_ids", Ort::Value::CreateTensor<int64_t>(cpu_mem, input_ids.data(), input_ids.size(), shape.data(), 2));
            io_binding.BindInput("attention_mask",
                                 Ort::Value::CreateTensor<int64_t>(cpu_mem, attention_mask.data(), attention_mask.size(), shape.data(), 2));
            io_binding.BindOutput("logits", cpu_mem);

            std::vector<uint8_t> results(num_inputs, 0);

            try
            {
                session->Run(Ort::RunOptions {nullptr}, io_binding);
                auto outputs = io_binding.GetOutputValues();
                const float *logits = outputs[0].GetTensorData<float>();

#pragma omp parallel for
                for (int i = 0; i < (int)num_inputs; ++i)
                {
                    const float *s = &logits[i * 3];
                    int label = std::distance(s, std::max_element(s, s + 3));
                    results[i] = (label == 2); // 2 = Entailment
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "[DeBERTa-MNLI Error] " << e.what() << std::endl;
                return std::vector<uint8_t>(num_inputs, 0);
            }

            return results;
        }

        std::vector<float> CandidateSpace::predictBatch_with_ONNX_score_picClass(const std::string &model_key, const std::vector<std::string> &inputs)
        {
            size_t batch_size = inputs.size();
            if (batch_size == 0 || pic_meterials_.empty())
                return {};

            // 找 target_index
            int target_idx = -1;
            for (size_t k = 0; k < pic_meterials_.size(); ++k)
            {
                if (pic_meterials_[k] == query_meterial)
                {
                    target_idx = static_cast<int>(k);
                    break;
                }
            }
            if (target_idx == -1)
                return std::vector<float>(batch_size, 0.0f);

            Ort::Session *session = getSessionForKey(model_key);
            auto *tokenizer = getTokenizerForKey(model_key);
            if (!session || !tokenizer)
                return std::vector<float>(batch_size, 0.0f);

            const int64_t MAX_TEXT_LEN = 77;
            const int IMG_SIZE = 224;
            const size_t num_labels = pic_meterials_.size();

            // 文本预处理
            std::vector<int64_t> flat_input_ids(num_labels * MAX_TEXT_LEN, 0);
            std::vector<int64_t> flat_attention_mask(num_labels * MAX_TEXT_LEN, 1);
            for (size_t i = 0; i < num_labels; ++i)
            {
                std::vector<int> tokens = tokenizer->Encode(pic_meterials_[i]);
                size_t offset = i * MAX_TEXT_LEN;
                flat_input_ids[offset] = 49406; // BOS
                size_t content_len = std::min(tokens.size(), (size_t)75);
                for (size_t j = 0; j < content_len; ++j)
                    flat_input_ids[offset + 1 + j] = static_cast<int64_t>(tokens[j]);
                flat_input_ids[offset + 1 + content_len] = 49407; // EOS
            }

            // 图像批量预处理
            std::vector<float> all_pixel_values(batch_size * 3 * IMG_SIZE * IMG_SIZE);
            std::vector<uint8_t> valid_images(batch_size, 1);

#pragma omp parallel for schedule(dynamic) if (batch_size > 1)
            for (int i = 0; i < (int)batch_size; ++i)
            {
                cv::Mat img = cv::imread(inputs[i]);
                if (img.empty())
                {
                    valid_images[i] = 0;
                    continue;
                }
                try
                {
                    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                    float scale = (float)IMG_SIZE / std::min(img.rows, img.cols);
                    cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_CUBIC);
                    cv::Mat cropped = img(cv::Rect((img.cols - IMG_SIZE) / 2, (img.rows - IMG_SIZE) / 2, IMG_SIZE, IMG_SIZE));

                    float mean[] = {0.48145466f, 0.45782750f, 0.40821073f};
                    float std_dev[] = {0.26862954f, 0.26130258f, 0.27577711f};

                    float *dest = all_pixel_values.data() + (i * 3 * IMG_SIZE * IMG_SIZE);
                    for (int c = 0; c < 3; ++c)
                        for (int h = 0; h < IMG_SIZE; ++h)
                            for (int w = 0; w < IMG_SIZE; ++w)
                            {
                                float pix = static_cast<float>(cropped.at<cv::Vec3b>(h, w)[c]) / 255.0f;
                                dest[c * IMG_SIZE * IMG_SIZE + h * IMG_SIZE + w] = (pix - mean[c]) / std_dev[c];
                            }
                }
                catch (...)
                {
                    valid_images[i] = 0;
                }
            }

            // 推理
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> txt_shape = {(int64_t)num_labels, MAX_TEXT_LEN};
            std::vector<int64_t> img_shape = {(int64_t)batch_size, 3, IMG_SIZE, IMG_SIZE};

            const char *input_names[] = {"input_ids", "pixel_values", "attention_mask"};
            const char *output_names[] = {"logits_per_image"};

            std::vector<Ort::Value> input_tensors;
            input_tensors.emplace_back(
                Ort::Value::CreateTensor<int64_t>(mem_info, flat_input_ids.data(), flat_input_ids.size(), txt_shape.data(), txt_shape.size()));
            input_tensors.emplace_back(
                Ort::Value::CreateTensor<float>(mem_info, all_pixel_values.data(), all_pixel_values.size(), img_shape.data(), img_shape.size()));
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, flat_attention_mask.data(), flat_attention_mask.size(),
                                                                         txt_shape.data(), txt_shape.size()));

            std::vector<float> final_scores(batch_size, 0.0f);

            try
            {
                auto outputs = session->Run(Ort::RunOptions {nullptr}, input_names, input_tensors.data(), 3, output_names, 1);
                float *logits_ptr = outputs[0].GetTensorMutableData<float>();

                for (size_t i = 0; i < batch_size; ++i)
                {
                    if (!valid_images[i])
                        continue;
                    float *row = logits_ptr + (i * num_labels);

                    // === 开放世界二元谓词 score ===
                    // 使用 target 类别 vs 所有其他类别的 logit 差
                    float l_target = row[target_idx];
                    float max_other = -1e9f;
                    for (size_t j = 0; j < num_labels; ++j)
                    {
                        if (j == (size_t)target_idx)
                            continue;
                        max_other = std::max(max_other, row[j]);
                    }

                    // 二分类概率 (数值稳定)
                    float max_val = std::max(l_target, max_other);
                    float score = std::exp(l_target - max_val) / (std::exp(l_target - max_val) + std::exp(max_other - max_val));

                    final_scores[i] = score;
                }
            }
            catch (...)
            { /* 保持 0.0f */
            }

            return final_scores;
        }

        std::vector<uint8_t> CandidateSpace::predictBatch_with_ONNX(const std::string &model_key, const std::vector<std::string> &inputs)
        {
            if (model_key == "01")
            {
                return predictBatch_with_ONNX_bert(model_key, inputs);
            }
            else if (model_key == "11")
            {
                // return predictBatch_with_ONNX_bart(model_key, inputs);
                return predictBatch_with_ONNX_textClass2(model_key, inputs);
            }
            else if (model_key == "21")
            {
                return predictBatch_with_ONNX_picClass(model_key, inputs);
            }
            else if (model_key == "31")
            {
                return predictBatch_with_ONNX_roberta_oracle(model_key, inputs);
            }
            else if (model_key == "41")
            {
                return predictBatch_with_ONNX_textClass2(model_key, inputs);
            }
            else
            {
                throw std::runtime_error("model_key error: " + model_key);
            }
        }

        std::vector<uint8_t> CandidateSpace::predictBatch_with_ONNX_bart(const std::string &model_key, const std::vector<std::string> &inputs)
        {
            size_t num_inputs = inputs.size();
            if (num_inputs == 0)
                return {};

            Ort::Session *session = getSessionForKey(model_key);
            auto *tokenizer = getTokenizerForKey(model_key);
            if (!session || !tokenizer)
                return std::vector<uint8_t>(num_inputs, 0);

            const int64_t MAX_LEN = 128;
            const int64_t BOS_ID = 0, EOS_ID = 2, PAD_ID = 1;

            // 1. 对 query_type_ 编码（hypothesis）
            std::vector<int> query_tokens = tokenizer->Encode(query_type_);

            // 2. 批量构造 input_ids 和 attention_mask（与 proxy 保持一致）
            std::vector<int64_t> flat_input_ids(num_inputs * MAX_LEN, PAD_ID);
            std::vector<int64_t> flat_attention_mask(num_inputs * MAX_LEN, 0);

#pragma omp parallel for schedule(dynamic) if (num_inputs > 1)
            for (int i = 0; i < (int)num_inputs; ++i)
            {
                std::vector<int> premise_ids = tokenizer->Encode(inputs[i]);
                size_t offset = i * MAX_LEN;
                size_t curr = 0;

                flat_input_ids[offset + curr++] = BOS_ID;

                // premise
                size_t max_p_len = MAX_LEN - query_tokens.size() - 2; // 留出 hypothesis + EOS
                size_t p_len = std::min(premise_ids.size(), max_p_len);
                for (size_t k = 0; k < p_len; ++k)
                    flat_input_ids[offset + curr++] = static_cast<int64_t>(premise_ids[k]);
                flat_input_ids[offset + curr++] = EOS_ID;

                // hypothesis
                size_t h_len = std::min(query_tokens.size(), (size_t)(MAX_LEN - curr - 1));
                for (size_t k = 0; k < h_len; ++k)
                    flat_input_ids[offset + curr++] = static_cast<int64_t>(query_tokens[k]);
                flat_input_ids[offset + curr++] = EOS_ID;

                // attention mask
                std::fill(flat_attention_mask.begin() + offset, flat_attention_mask.begin() + offset + curr, 1);
            }

            // 3. ONNX 推理
            try
            {
                Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                std::vector<int64_t> shape = {(int64_t)num_inputs, MAX_LEN};
                const char *input_names[] = {"input_ids", "attention_mask"};
                const char *output_names[] = {"logits"};

                std::vector<Ort::Value> input_tensors;
                input_tensors.emplace_back(
                    Ort::Value::CreateTensor<int64_t>(mem_info, flat_input_ids.data(), flat_input_ids.size(), shape.data(), shape.size()));
                input_tensors.emplace_back(
                    Ort::Value::CreateTensor<int64_t>(mem_info, flat_attention_mask.data(), flat_attention_mask.size(), shape.data(), shape.size()));

                auto outputs = session->Run(Ort::RunOptions {nullptr}, input_names, input_tensors.data(), 2, output_names, 1);
                float *logits_ptr = outputs[0].GetTensorMutableData<float>();

                // 4. 计算 entail 概率并返回 uint8_t 标签
                std::vector<uint8_t> results(num_inputs, 0);
                for (size_t i = 0; i < num_inputs; ++i)
                {
                    float *row = logits_ptr + i * 3; // 3 分类: contradiction, neutral, entailment
                    float l_contra = row[0];
                    float l_entail = row[2];

                    // 数值稳定 Softmax 计算二元概率
                    float max_val = std::max(l_contra, l_entail);
                    float prob_entail = std::exp(l_entail - max_val) / (std::exp(l_contra - max_val) + std::exp(l_entail - max_val));

                    results[i] = prob_entail > 0.5f ? 1 : 0; // 超过 0.5 判定为 true
                }

                return results;
            }
            catch (...)
            {
                return std::vector<uint8_t>(num_inputs, 0);
            }
        }

        std::vector<uint8_t> CandidateSpace::predictBatch_with_ONNX_picClass(const std::string &model_key, const std::vector<std::string> &inputs)
        {
            size_t batch_size = inputs.size();
            if (batch_size == 0 || pic_meterials_.empty())
                return {};

            // 找 target_index
            int target_idx = -1;
            for (size_t k = 0; k < pic_meterials_.size(); ++k)
            {
                if (pic_meterials_[k] == query_meterial)
                {
                    target_idx = static_cast<int>(k);
                    break;
                }
            }
            if (target_idx == -1)
                return std::vector<uint8_t>(batch_size, 0);

            Ort::Session *session = getSessionForKey(model_key);
            auto *tokenizer = getTokenizerForKey(model_key);
            if (!session || !tokenizer)
                return std::vector<uint8_t>(batch_size, 0);

            const int64_t MAX_TEXT_LEN = 77;
            const int IMG_SIZE = 224;
            const size_t num_labels = pic_meterials_.size();

            // 文本预处理
            std::vector<int64_t> flat_input_ids(num_labels * MAX_TEXT_LEN, 0);
            std::vector<int64_t> flat_attention_mask(num_labels * MAX_TEXT_LEN, 1); // 已修正: 全 1
            for (size_t i = 0; i < num_labels; ++i)
            {
                std::vector<int> tokens = tokenizer->Encode(pic_meterials_[i]);
                size_t offset = i * MAX_TEXT_LEN;
                flat_input_ids[offset] = 49406; // BOS
                size_t content_len = std::min(tokens.size(), (size_t)75);
                for (size_t j = 0; j < content_len; ++j)
                    flat_input_ids[offset + 1 + j] = static_cast<int64_t>(tokens[j]);
                flat_input_ids[offset + 1 + content_len] = 49407; // EOS
            }

            // 图像批量预处理
            std::vector<float> all_pixel_values(batch_size * 3 * IMG_SIZE * IMG_SIZE);
            std::vector<uint8_t> valid_images(batch_size, 1);

#pragma omp parallel for schedule(dynamic) if (batch_size > 1)
            for (int i = 0; i < (int)batch_size; ++i)
            {
                cv::Mat img = cv::imread(inputs[i]);
                if (img.empty())
                {
                    valid_images[i] = 0;
                    continue;
                }
                try
                {
                    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                    float scale = static_cast<float>(IMG_SIZE) / std::min(img.rows, img.cols);
                    cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_CUBIC);

                    // 计算安全的 crop 坐标
                    int crop_x = (img.cols - IMG_SIZE) / 2;
                    int crop_y = (img.rows - IMG_SIZE) / 2;
                    crop_x = std::max(0, std::min(crop_x, img.cols - IMG_SIZE));
                    crop_y = std::max(0, std::min(crop_y, img.rows - IMG_SIZE));

                    cv::Mat cropped = img(cv::Rect(crop_x, crop_y, IMG_SIZE, IMG_SIZE));
                    assert(cropped.rows == IMG_SIZE && cropped.cols == IMG_SIZE);

                    float mean[] = {0.48145466f, 0.45782750f, 0.40821073f};
                    float std_dev[] = {0.26862954f, 0.26130258f, 0.27577711f};

                    float *dest = all_pixel_values.data() + (i * 3 * IMG_SIZE * IMG_SIZE);
                    for (int c = 0; c < 3; ++c)
                        for (int h = 0; h < IMG_SIZE; ++h)
                            for (int w = 0; w < IMG_SIZE; ++w)
                            {
                                float pix = static_cast<float>(cropped.at<cv::Vec3b>(h, w)[c]) / 255.0f;
                                dest[c * IMG_SIZE * IMG_SIZE + h * IMG_SIZE + w] = (pix - mean[c]) / std_dev[c];
                            }
                }
                catch (...)
                {
                    valid_images[i] = 0; // 图像预处理失败标记
                }
            }

            // 推理
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> txt_shape = {(int64_t)num_labels, MAX_TEXT_LEN};
            std::vector<int64_t> img_shape = {(int64_t)batch_size, 3, IMG_SIZE, IMG_SIZE};

            const char *input_names[] = {"input_ids", "pixel_values", "attention_mask"};
            const char *output_names[] = {"logits_per_image"};

            std::vector<Ort::Value> input_tensors;
            input_tensors.emplace_back(
                Ort::Value::CreateTensor<int64_t>(mem_info, flat_input_ids.data(), flat_input_ids.size(), txt_shape.data(), txt_shape.size()));
            input_tensors.emplace_back(
                Ort::Value::CreateTensor<float>(mem_info, all_pixel_values.data(), all_pixel_values.size(), img_shape.data(), img_shape.size()));
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, flat_attention_mask.data(), flat_attention_mask.size(),
                                                                         txt_shape.data(), txt_shape.size()));

            std::vector<uint8_t> results(batch_size, 0);

            try
            {
                auto outputs = session->Run(Ort::RunOptions {nullptr}, input_names, input_tensors.data(), 3, output_names, 1);
                float *logits_ptr = outputs[0].GetTensorMutableData<float>();

                for (size_t i = 0; i < batch_size; ++i)
                {
                    if (!valid_images[i])
                        continue;
                    float *row = logits_ptr + (i * num_labels);
                    float l_target = row[target_idx];

                    // 开放世界二元判断逻辑：Top-1 竞争
                    bool is_entailed = true;
                    for (size_t j = 0; j < num_labels; ++j)
                    {
                        if (j == (size_t)target_idx)
                            continue;
                        if (row[j] >= l_target)
                        { // 严格 Top-1
                            is_entailed = false;
                            break;
                        }
                    }
                    results[i] = is_entailed ? 1 : 0;
                }
            }
            catch (...)
            {
                return std::vector<uint8_t>(batch_size, 0);
            }

            return results;
        }

        // =========================================================
        // 函数：waitOracleComplete
        // 功能：同步等待所有 Oracle 批量任务完成。
        //       保证 oracle_pending、oracle_task_map 均为空，
        //       所有结果都已写入 oracleRecord。
        // 调用时机：需要使用完整 Oracle 结果的阶段（例如 recall/precision 计算前）。
        // =========================================================
        void CandidateSpace::waitOracleComplete()
        {
            // printf("[OracleBarrier] waiting for all Oracle tasks to complete...\n");
            std::unique_lock<std::mutex> lk(oracle_mutex);
            // printf("oracle_pending size: %zu, oracle_task_map size: %zu, in_flight_batches: %d\n",
            //        oracle_pending.size(), oracle_task_map.size(), in_flight_batches.load());
            oracle_cv.wait(lk, [this]() { return oracle_pending.empty() && oracle_task_map.empty() && in_flight_batches == 0; });

            // 双重保险：若在解锁前又有新任务，则主线程负责同步 flush（取出并直接运行）
            std::vector<std::tuple<int, int, std::string>> leftover;
            if (!oracle_pending.empty())
            {
                leftover.swap(oracle_pending);
            }
            lk.unlock();

            if (!leftover.empty())
            {
                // 主线程直接同步执行尾批，确保所有结果写回
                runOracleBatchSync(leftover);
                {
                    std::scoped_lock<std::mutex> lk2(oracle_mutex);
                    oracle_cv.notify_all();
                }
            }

            // std::cerr << "[OracleBarrier] all Oracle tasks completed.\n";
        }

        // 根据查询点 u 决定该点使用的模型（可以是模型类别、模型路径、任务类型等）
        // 例如：根据 label 或任务类型返回 model code
        // model_key 可以是任意字符串，用于标识不同模型
        int CandidateSpace::getModelKey(int u)
        {
            if (query_->predicateMap.find(u) != query_->predicateMap.end())
            {
                return query_->predicateMap[u];
            }
            return -1; // 默认或未知模型
        }

        // 根据模型 key 获取 ONNX session
        Ort::Session *CandidateSpace::getSessionForKey(const std::string &key)
        {
            static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CandidateSpace_Env");

            //  key → 模型路径
            static const std::unordered_map<std::string, std::string> keyToPath = {
                {"00", modelFile("distilbert-base-uncased-finetuned-sst-2-english", "model.onnx")},
                {"01", modelFile("bert-large-uncased-sst2", "model.onnx")},
                {"10", modelFile("deberta-v3-base-mnli-fever-anli", "model.onnx")},
                {"11", modelFile("deberta-v2-xlarge-mnli", "model.onnx")},
                {"20", modelFile("clip-vit-base-patch32", "model.onnx")},
                {"21", modelFile("clip-vit-large-patch14", "model.onnx")},
                {"30", modelFile("distilbert-base-uncasedv1-finetuned-twitter-sentiment", "model.onnx")},
                {"31", modelFile("twitter-roberta-base-sentiment-latest", "model.onnx")},
                {"40", modelFile("deberta-v3-base-mnli-fever-anli", "model.onnx")},
                {"41", modelFile("deberta-v2-xlarge-mnli", "model.onnx")}};

            //  model_key → GPU 映射（你有 4 张 3090）
            static const std::unordered_map<std::string, int> keyToGPU = {{"00", 0}, {"01", 1}, {"10", 0}, {"11", 1}, {"20", 0},
                                                                          {"21", 1}, {"30", 2}, {"31", 2}, {"40", 2}, {"41", 3}};

            auto pit = keyToPath.find(key);
            if (pit == keyToPath.end())
            {
                std::cerr << "[Error] Unknown model key: " << key << std::endl;
                return nullptr;
            }

            //  Session cache
            static std::unordered_map<std::string, std::unique_ptr<Ort::Session>> sessionCache;

            if (sessionCache.count(key))
            {
                return sessionCache[key].get();
            }

            int gpu = keyToGPU.at(key);
            const std::string &modelPath = pit->second;

            std::cout << "[ONNX] Loading model " << key << " on GPU " << gpu << std::endl;

            //  每个模型独立 SessionOptions + CUDA allocator
            Ort::SessionOptions opts;
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            //  禁用 CPU 内存池（如果需要）
            opts.DisableCpuMemArena(); // Optional for CPU

            //  CUDA 配置
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = gpu;           // 为每个模型分配不同的 GPU
            cuda_options.gpu_mem_limit = 0;         // 可选，限制显存使用（如果需要）
            cuda_options.arena_extend_strategy = 0; // 禁用 Arena 扩展
            cuda_options.do_copy_in_default_stream = 1;

            opts.AppendExecutionProvider_CUDA(cuda_options);

            try
            {
                //  加载模型并缓存 Session
                sessionCache[key] = std::make_unique<Ort::Session>(env, modelPath.c_str(), opts);
                std::cout << "✅ Model " << key << " loaded on GPU " << gpu << std::endl;
            }
            catch (const Ort::Exception &e)
            {
                std::cerr << "❌ Failed to load model " << key << ": " << e.what() << std::endl;
                return nullptr;
            }

            return sessionCache[key].get();
        }

        

    } // namespace SubgraphMatching
} // namespace GraphLib
