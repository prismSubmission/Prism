// CandidateSpace.h
#pragma once
#include "SubgraphMatching/DataGraph.h"
#include "SubgraphMatching/PatternGraph.h"
#include "DataStructure/Graph.h"
#include "Base/Base.h"
#include "onnxruntime_cxx_api.h"
#include "Base/Timer.h"
#include "Base/BasicAlgorithms.h"
#include "DataStructure/trie.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <cmath>
#include <map>
#include <condition_variable>
#include <thread>
#include <set>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/beta.hpp>
#include <utility>
#include <random>
#include <chrono>
#include <numeric>
#include <deque>
#include <tuple>
#include <unordered_set>
#include "PreRecEstimation/progressiveMatching.h"
#include "Base/LoadGraph.h"
#include "PreRecEstimation/estimation.h"
#include <queue>
#include <tokenizers_cpp.h>
#include <vector>
using namespace std;
#define MAX_QUERY_NODES 20
/**
 * @brief The Candidate Space structure
 * @date 2023-05
 */

namespace GraphLib
{
    namespace SubgraphMatching
    {
        struct pair_hash
        {
            template <typename T1, typename T2>
            std::size_t operator()(const std::pair<T1, T2> &p) const
            { // 补全 operator()
                auto h1 = std::hash<T1> {}(p.first);
                auto h2 = std::hash<T2> {}(p.second);
                return h1 ^ (h2 << 1); // 合并哈希值
            }
        };

        inline BipartiteMaximumMatching BPSolver;
        enum STRUCTURE_FILTER
        {
            NO_STRUCTURE_FILTER,
            TRIANGLE_SAFETY,
            FOURCYCLE_SAFETY
        };
        enum NEIGHBOR_FILTER
        {
            NEIGHBOR_SAFETY,
            NEIGHBOR_BIPARTITE_SAFETY,
            EDGE_BIPARTITE_SAFETY
        };

        enum MatchOrderStrategy
        {
            INFERENCE_FIRST,
            STRUCTURE_FIRST,
            INFERENCE_ON_DEMAND,
        };

        enum partial_match_state
        {
            failed_expension,
            full_match,
            failed_structure,
            expending
        };

        class SubgraphMatchingOption
        {
        public:
            STRUCTURE_FILTER structure_filter = FOURCYCLE_SAFETY;
            NEIGHBOR_FILTER neighborhood_filter = EDGE_BIPARTITE_SAFETY;
            int MAX_QUERY_VERTEX = 50, MAX_QUERY_EDGE = 250;
            long long max_num_matches = -1;
            SubgraphMatching::MatchOrderStrategy match_order_strategy = SubgraphMatching::MatchOrderStrategy::STRUCTURE_FIRST;
            bool oracle_only = false; // baseline only uses oracle
        };

        class Tq
        {
        private:
        public:
            int cur; // EdgeId
            int sub_count;
            std::vector<Tq *> child_;
            Tq() : cur(-1), sub_count(0) {}
            ~Tq()
            {
                for (Tq *child : child_)
                {
                    delete child; // 递归释放子节点
                }
                child_.clear(); // 清空vector，避免悬空指针
            }
        };

        class CandidateSpace
        {
        public:
            std::vector<bool> used_ids;
            int partial_match_array[MAX_QUERY_NODES];
            long long total_matches_found = 0;
            std::mt19937 rng_;
            Tq *Tq_root;
            double gamma_ = 0.9;
            std::unordered_map<int, std::unordered_map<int, std::pair<double, TreeNode *>>> node_map; // u的候选集合，以及它对应的指向前缀树的指针。
            std::unordered_map<int, multimap<double, int, std::greater<double>>> map_Score; // u的候选集合，按照score降序，得到v
            std::vector<int> match_order;
            std::vector<int> match_order_map;
            string dataset_;
            bool query_type_precision = true;                   // true: precision; false: recall
            std::vector<std::pair<string, string>> review_text; // 存储所有的review文本
            std::vector<double> threshold_vertex;               // 某个vertex对应的threshold
            std::vector<double> threshold_vertex_new_global;           
            TreeNode *TrieRoot;
            std::unordered_map<int, std::unordered_map<int, bool>> oracleRecord_;
            std::unordered_map<int, std::unordered_map<int, bool>> oracleRecord;
            std::unordered_map<int, int> precision_marginal;                           // 记录每个query节点的边精确率
            std::unordered_map<int, double> precision_marginal_frac;                   // 记录每个query节点的边精确率（总体）
            std::unordered_map<int, std::pair<int, int>> treeSampling_recall_marginal; // 记录每个query节点的边召回率
            std::unordered_map<int, double> treeSampling_recall_marginal_all;
            void addoracle_scoremap_wanderjoin(int u, int v, int weight);       // 添加oracle结果到oracle_scoremap
            std::unordered_map<int, std::tuple<int, int, int>> oracle_scoremap; // 需不需要multiset后续再看,<TP,P,总体>
            std::unordered_map<int, double> wanderJoin_recall_marginal;
            int est_type_ = 0; // 0: precision; 1: recall_wanderjoin; 2: recall_treesampling

            std::pair<std::vector<double>, int> aggregate_marginal_to_models(const std::unordered_map<int, double> &accurate_marginal,
                                                                             const std::vector<double> &current_theta, int model_count);
            unordered_set<int> full_match_leaf;
            std::vector<std::vector<int>> local_candidates;
            // std::vector<std::vector<int>> result_candidates; // 存储候选集
            std::vector<double> weightedSet;
            std::vector<bool> weightedSet_flag;
            // std::vector<bool> weighted_tp_list;
            int checkTP = 0;
            int checkP = 0;
            int checkTP_2 = 0;
            int checkP_2 = 0;
            string vertex_review_path_;
            string vertex_product_path_;
            string reviews_kcore_path_;
            string modelLocation_path_;
            string product_kcore_path__;
            string image_path_;
            string root_name_;
            string query_type_ = "This product is kitchenware, cookware, or used for dining."; // Amazon
            string query_meterial = "wooden item";
            std::vector<string> pic_meterials_ = {
                "plastic product", // 涵盖：塑料、树脂、有机玻璃（视觉特征：均匀、哑光或半反光）
                "metal object",    // 涵盖：钢、铝、铁、合金（视觉特征：强高光、金属质感）
                "wooden item",     // 涵盖：木头、竹子、软木（视觉特征：纤维纹理、深色哑光）
                "fabric textile",  // 涵盖：布料、丝绸、皮革、纸张（视觉特征：粗糙度、纹理感、无反光）
                "glass material"   // 涵盖：玻璃、透明瓶身、高亮陶瓷（视觉特征：透明、折射、环境反射）
            };

            std::vector<std::string> text_labels_ = {
                "This product is a household electrical appliance.",        "This product is used in the bathroom.",
                "This product is bedding or related to sleeping.",          "This product is cleaning or household chores.",
                "This product is furniture or used for furnishing a home.", "This product is gardening or outdoor yard work.",
                "This product is home decoration or aesthetic purposes.",   "This product is kitchenware, cookware, or used for dining.",
                "This product is used for storage or organization."};

            // ====== 新增成员：Oracle 批量延迟执行相关 ======
            std::mutex oracle_mutex;
            std::condition_variable oracle_cv;
            std::thread oracle_worker;
            std::atomic<bool> oracle_worker_stop {false};
            int oracle_batch_size = 64;   // 推荐值：32/64，可按 GPU/模型调整
            int oracle_max_wait_ms = 200; // 最长等待时间，超时即触发 batch
            // ONNX / runtime 相关句柄（可选：若你用 ONNX Runtime）
            std::unordered_map<std::string, std::unique_ptr<Ort::Session>> oracle_sessions;
            std::unordered_map<std::string, std::unique_ptr<tokenizers::Tokenizer>> oracle_tokenizers; // 新增：用于管理已加载的 Tokenizer 实例
            std::unique_ptr<Ort::Session> oracle_session;                        // 需要包含 ONNX header（文件头部已含）
            Ort::Env oracle_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "oracle"); // 若全局已有可复用
            std::mutex oracle_session_mutex;                                     // 如果 session 生命周期在线程中访问需锁
            std::atomic<int> in_flight_batches {0};
            // === 新增成员（放在 CandidateSpace 类的成员变量区域） ===
            // 用于 Oracle 异步批量推理与去重共享
            std::unordered_map<int, std::vector<std::tuple<int, int, std::string>>> oracle_task_map; // data node id -> list of (u, cand_idx)
            std::unordered_map<std::string, std::unordered_set<int>> oracle_pending_nodes;
            std::vector<std::tuple<int, int, std::string>> oracle_pending; // 待处理 (query_u, cand_idx)                            //
                                                                           // 去重表（记录已在 pending 列表中的 data node ids）
            std::atomic<bool> oracle_no_more_requests {false};             // 主线程显式告诉 worker：不会再有新请求
            std::map<std::string, tokenizers::Tokenizer *> tokenizerCache; // 缓存：模型 key -> Tokenizer 实例指针
            std::map<std::string, Ort::Session *> sessionCache;            // 缓存：模型 key -> ONNX Session 实例指针

            string review_rowID_path_;
            string output_path_ = "";
            // 文件管理数据结构，存储所有的字符串
            std::vector<std::string> all_strings;
            // 将cand_id映射到字符串的索引
            std::unordered_map<int, int> candid_to_string_index;
            std::vector<std::vector<std::string>> all_string_inputs;
            std::vector<std::unordered_map<int, int>> all_string_idx;

            // ====== 新增方法声明 ======
            void startOracleWorker();
            void waitOracleComplete();
            void stopOracleWorkerAndFlush();

            void enqueueOracleTask(int u, int cand_idx, string model_key); // 添加任务（线程安全）
            // 手动调用oracle model进行推理
            void runOracleSingleReq(int u, int cand_idx, string model_key);
            void runOracleBatchSync(const std::vector<std::tuple<int, int, std::string>> &batch); // 真正的批量推理

            // helper，用于将文本批量预测
            std::vector<uint8_t> predictBatch_with_ONNX(const std::string &model_key, const std::vector<std::string> &inputs);

            // 返回bool版本
            std::vector<uint8_t> predictBatch_with_ONNX_bert(const std::string &model_key, const std::vector<std::string> &inputs);
            std::vector<uint8_t> predictBatch_with_ONNX_bart(const std::string &model_key, const std::vector<std::string> &inputs);
            std::vector<uint8_t> predictBatch_with_ONNX_picClass(const std::string &model_key, const std::vector<std::string> &inputs);
            // 返回bool版本,parler
            std::vector<uint8_t> predictBatch_with_ONNX_textClass2(const std::string &model_key, const std::vector<std::string> &inputs);
            std::vector<uint8_t> predictBatch_with_ONNX_roberta_oracle(const std::string &model_key, const std::vector<std::string> &inputs);           // 返回bool版本

            // 返回分数版本
            std::vector<float> predictBatch_with_ONNX_score(const std::string &model_key, const std::vector<std::string> &inputs);
            std::vector<float> predictBatch_with_ONNX_score_bert(const std::string &model_key, const std::vector<std::string> &inputs);
            std::vector<float> predictBatch_with_ONNX_score_picClass(const std::string &model_key, const std::vector<std::string> &inputs);
            std::vector<float> predictBatch_with_ONNX_score_textClass(const std::string &model_key, const std::vector<std::string> &inputs);
            // 返回分数版本,parler
            std::vector<float> predictBatch_with_ONNX_score_textClass2(const std::string &model_key, const std::vector<std::string> &inputs);
            // 被废弃的
            std::vector<bool> predictBatch_with_ONNX_old(const std::string &model_key, const std::vector<std::string> &inputs);

            tokenizers::Tokenizer *getTokenizerForKey(const std::string &key);
            // TODO: 从 vertex->row->review 构建文本输入的函数
            std::string buildModelInputForPair(int u, int cand_idx, std::string model_key);
            // std::string buildModelInputForPair_type1(int u, int cand_idx, std::string model_key_all);
            int predictAll(std::string model_key, std::vector<std::string> &all_inputs, std::vector<float> &all_results);
            int predictAll_baseline(std::string model_key, std::vector<std::string> &all_inputs, std::vector<uint8_t> &all_results);

            bool buildModelInputForBatch(std::string model_key, std::unordered_set<int> *needed_vertex_set);
            bool buildModelInputForBatch_baseline(std::string model_key, std::unordered_set<int> *needed_vertex_set);
            bool buildModelInputForBatch_parler(std::string model_key);
            void buildCompactVectorMapping(std::vector<int> &needed_vertex_ids, std::vector<std::string> &vertex_to_text,
                                           std::unordered_map<int, int> &vertex_to_index);
            void buildCompactVectorMapping_parler(std::vector<int> &needed_vertex_ids, std::vector<std::string> &vertex_to_text,
                                                  std::unordered_map<int, int> &vertex_to_index);

            std::vector<std::string> getVertexIdStringList(const std::vector<int> &vertex_id_list);
            std::string buildModelInputForPair_type1(int u, int cand_idx, std::string model_key_all);
            int getModelKey(int u);
            Ort::Session *getSessionForKey(const std::string &key);
            std::pair<int, int> optimalAllocation(int total_samples, double p1, double p2, int S1, int S2);
            std::vector<double> tune_thresholds(std::vector<double> theta, double r, const std::vector<double> &r_marg, double gamma, double kappa);

            double tune_threshold_single(double theta, double r_right, double gamma, double r_left);
            void getArgs(string vertex_review_path, string reviews_kcore_path, string modelLocation_path,
                         string vertex_product_path, string product_meta_path, string image_path,
                         int est_type, double gamma, string output_path, string root_name, string dataset);
            void initONNX();

            SubgraphMatchingOption opt;
            CandidateSpace(DataGraph *data, SubgraphMatchingOption filter_option);
            ~CandidateSpace();

            CandidateSpace &operator=(const CandidateSpace &) = delete;

            CandidateSpace(const CandidateSpace &) = delete;

            inline int GetCandidateSetSize(const int u) const { return candidate_set_[u].size(); };

            inline int GetCandidate(const int u, const int v_idx) const { return candidate_set_[u][v_idx]; };

            bool BuildCS(PatternGraph *query, SubgraphMatchingOption opt);

            std::vector<int> &GetCandidates(int u) { return candidate_set_[u]; }

            std::vector<int> &GetCandidateNeighbors(int cur, int cand_idx, int nxt)
            {
                return candidate_neighbors[cur][cand_idx][query_->GetAdjIdx(cur, nxt)];
            }
            int GetCandidateNeighbor(int cur, int cand_idx, int nxt, int nxt_idx)
            {
                return candidate_neighbors[cur][cand_idx][query_->GetAdjIdx(cur, nxt)][nxt_idx];
            }

            dict GetCSInfo() { return CSInfo; };

            int GetNumCSVertex() { return num_candidate_vertex; };
            int GetNumCSEdge() { return num_candidate_edge; };
            std::pair<double, int> precision_Estimation(std::vector<std::pair<TreeNode *, int>> full_match, int sampleSize);

            // 阈值降低，结果增加,从查询图中找到匹配结果放在候选树中，
            int progressive_subgraph_match_add(std::vector<double> threshold_vertex_new, MatchOrderStrategy strategy);
            // 阈值升高，结果减少
            int progressive_subgraph_match_del(std::vector<double> threshold_vertex_new);
            bool getOracle(int u, int v);
            double getProxyScore(int u, int v);

            bool searchMatchTrieLeaf(std::unordered_map<int, int> &partial_match, TreeNode *&ret, int lastNode_with_candidate);
            bool searchMatchTrieLeaf_ArrayVersion(int current_size, TreeNode *&ret, int lastNode_with_candidate);
            TreeNode *addPartialMatch2Trie_ArrayVersion(int current_size, partial_match_state state);
            TreeNode *addPartialMatch2Trie(std::unordered_map<int, int> &partial_match, partial_match_state state);

            int original_subgraph_match_ArrayVersion(int index, MatchOrderStrategy strategy, bool first_loop = true);
            int baseline_subgraph_match_ArrayVersion(int index, MatchOrderStrategy strategy, bool first_loop = true);

            int original_subgraph_match(int index, std::unordered_map<int, int> &partial_match, std::unordered_set<int> &all_cand,
                                        MatchOrderStrategy strategy, bool first_loop = true);
            int baseline_subgraph_match(int index, std::unordered_map<int, int> partial_match, std::unordered_set<int> all_cand,
                                        MatchOrderStrategy strategy, bool first_loop = true);

            bool getFullMatch(std::unordered_set<int> full_match_leaf, std::vector<std::pair<TreeNode *, int>> &full_match, bool oracle_only,
                              MatchOrderStrategy strategy);
            TreeNode *findFailedProxyNode(TreeNode *full_match_leaf, const std::vector<double> &threshold_vertex);

            int constructScoreIndex_onnx();
            int constructScoreIndex_onnx_baseline();

            std::unordered_map<int, std::unordered_set<int>> collectActiveCandidates();
            int constructScoreIndex_onnx_filtered(std::unordered_map<int, std::unordered_set<int>> &active_candidates);

            std::pair<double, double> recall_Estimation_Wanderjoin(std::vector<std::pair<TreeNode *, int>> partial_match, int sampleSize, double tp);
            std::tuple<bool, int, bool> subgraphSampling_withReplacement(int cur, int cand, int deepth, int weight,
                                                                         std::unordered_map<int, int> &partial_match_empty, int state,
                                                                         std::unordered_set<int> &all_cand);
            int weighted_sample(const std::vector<double> &weights, mt19937 &rng);
            std::tuple<double, double, int, int> getSingleNodeEstimation();
            bool changeThreshold_multi(double precision, double recall, std::vector<double> &threshold, std::vector<double> &threshold_new);

            void selectStartVertex(std::vector<int> &match_order, PatternGraph *query, std::unordered_map<int, int> &vertex_model_map,
                                   std::unordered_set<int> &visited_vertex, bool prefer_model);
            void selectNextVertex(std::vector<int> &match_order, PatternGraph *query, std::unordered_map<int, int> &vertex_model_map,
                                  std::unordered_set<int> &visited_vertex, bool prefer_model);
            void buildMatchOrderInferenceOnDemand(std::vector<int> &match_order, PatternGraph *query, std::unordered_map<int, int> &vertex_model_map);
            void buildMatchOrder(std::vector<int> &match_order, PatternGraph *query, MatchOrderStrategy strategy,
                                 std::unordered_map<int, int> &vertex_model_map);

            int getCandidateTreeCounts(int next_order, Tq *tq_node, int M, std::unordered_map<int, int> &fixed_assignments,
                                       std::vector<std::vector<long long>> &num_trees, int last_cand, int last_u, int u_next);
            double calculateVariance_Wanderjoin(std::vector<int> w_vec, std::vector<int> y_vec, std::vector<int> x_vec, double R_ht);
            double calculateVariance_stractified(double pi1, double pi2, int S1, int S2, int N1, int N2);
            double check_stopCondition(double variance, double epsilon, double confidence);

            bool exportMatchesToFile(std::vector<std::pair<TreeNode *, int>> full_match);

            bool sampleOneMatch(Tq *tq_node, std::unordered_map<int, int> &assignments, std::unordered_map<int, int> &fixed_assignments,
                                std::vector<int> &match_order, int M, std::unordered_map<int, int> &output_mapping,
                                std::vector<std::vector<long long>> &num_trees, int last_cand, int last_u, std::unordered_set<int> all_cand);
            std::pair<double, double> recall_Estimation_Wanderjoin_adapted(std::vector<std::pair<TreeNode *, int>> partial_match,
                                                                           double target_confidence, // 替代 sampleSize
                                                                           double epsilon            // 方差允许误差
            );
            bool getProductTextsForVertexIds(const std::vector<int> &vertexIds, std::vector<std::string> &vertex_to_text,
                                             std::unordered_map<int, int> &vertex_to_index, int maxLength = 512);

            bool getProductTextsForVertexIds_parler(const std::vector<int> &vertexIds, std::vector<std::string> &vertex_to_text,
                                                    std::unordered_map<int, int> &vertex_to_index, int maxLength = 512);

            bool getPicPathForVertexIds(const std::vector<int> &vertexIds, std::vector<std::string> &vertex_to_pic,
                                        std::unordered_map<int, int> &vertex_to_index);

            // 初始化 all_strings 和 candid_to_string_index
            void initializeStringStorage(const std::vector<std::pair<int, std::string>> &cand_strings);

            int weightedSample(const std::vector<int> &weights)
            {
                long long total = 0;
                for (int w : weights)
                    total += w;
                if (total == 0)
                    return -1; // 没有候选

                // 随机数生成
                std::uniform_int_distribution<long long> dist(0, total - 1);
                long long r = dist(rng_);

                // 累积查找
                long long prefix = 0;
                for (int i = 0; i < (int)weights.size(); i++)
                {
                    prefix += weights[i];
                    if (r < prefix)
                        return i;
                }
                return weights.size() - 1; // 理论上不会到这
            }

            // 构造以 match_order[M-1] 为根的生成树
            // 输入：
            //   query        - 查询图对象，假设提供 GetNeighbors(int u) -> std::vector<int>
            //   match_order  - 查询顶点的执行顺序
            //   M            - 已确认顶点的数量 (M >= 1)
            // 返回：
            //   生成树的根节点 (Tq*)，调用者需要负责释放内存
            Tq *BuildSpanningTreeTq(PatternGraph *query, const std::vector<int> &match_order, int M);

            std::pair<bool, int> checkPartialMatch(TreeNode *partial_match_node, vector<double> &threshold_vertex);
            void computeCountsPostOrder(TreeNode *node, Graph *dataGraph);
            void calculateCandidateTreeCounts(TreeNode *root);
            std::vector<int> GetRandomSubset(const std::vector<int> &candidates, int size);
            int ChooseExtendableVertex(const std::unordered_map<int, int> &M);
            double getPartialMatchRecallStratified(TreeNode *partial_match, int trails, int success);
            bool CheckEmbedding(const std::unordered_map<int, int> &sample, TreeNode *partial_match);
            double getPartialMatchRecallTree(TreeNode *partial_match, int trails);
            std::vector<int> local_candidate_size;
            std::vector<std::pair<std::vector<int>::iterator, std::vector<int>::iterator>> iterators;

            std::pair<double, double>
            recall_Estimation_treeSampling_stratified(std::vector<std::pair<TreeNode *, int>> partial_match, int sampleSize, double tp,
                                                      bool use_adaptive_sampling = false // true = 自适应采样，false = 固定比例分配
            );
            vector<int> BuildExtendableCandidates(int u, unordered_map<int, int> &sample)
            {
                local_candidate_size[u] = 0;
                iterators.clear();
                for (int q_nbr : query_->GetNeighbors(u))
                {
                    if (sample[q_nbr] == -1)
                        continue;
                    auto &candidate_neighbors = GetCandidateNeighbors(q_nbr, sample[q_nbr], u);
                    iterators.emplace_back(candidate_neighbors.begin(), candidate_neighbors.end());
                }
                std::sort(iterators.begin(), iterators.end(), [](auto &a, auto &b) -> bool { return a.second - a.first < b.second - b.first; });
                Intersection(u);
                return std::vector<int>(local_candidates[u].begin(), local_candidates[u].begin() + local_candidate_size[u]);
            }

            void Intersection(int index)
            {
                if (local_candidate_size[index] > 0)
                    return;
                int num_vectors = iterators.size();
                if (num_vectors == 1)
                {
                    while (iterators[0].first != iterators[0].second)
                    {
                        local_candidates[index][local_candidate_size[index]++] = (*iterators[0].first);
                        ++iterators[0].first;
                    }
                    return;
                }
                while (iterators[0].first != iterators[0].second)
                {
                    int target = *iterators[0].first;
                    for (int i = 1; i < num_vectors; i++)
                    {
                        while (iterators[i].first != iterators[i].second)
                        {
                            if (*iterators[i].first < target)
                            {
                                ++iterators[i].first;
                            }
                            else if (*iterators[i].first > target)
                            {
                                goto nxt_target;
                            }
                            else
                                break;
                        }
                        if (iterators[i].first == iterators[i].second)
                            return;
                    }
                    local_candidates[index][local_candidate_size[index]++] = target;
                nxt_target:
                    ++iterators[0].first;
                }
            }

            bool checkEstimationDone(double precision, double recall)
            {
                if (query_type_precision)
                {
                    if (precision >= gamma_)
                    {
                        return true;
                    }
                }
                else
                {
                    if (recall >= gamma_)
                    {
                        return true;
                    }
                }
                return false;
            }

            bool checkEstimationDone_single(double precision, double recall, double r_left, double r_right)
            {
                if (query_type_precision)
                {
                    if (precision >= gamma_ && (r_right - r_left) <= 0.05)
                    {
                        return true;
                    }
                }
                else
                {
                    if (recall >= gamma_ && (r_right - r_left) <= 0.05)
                    {
                        return true;
                    }
                }
                return false;
            }

            void getpartialMatch(int length, TreeNode *leaf, std::unordered_map<int, int> &partial_match, std::unordered_set<int> &all_cand)
            {
                TreeNode *current = leaf;
                int steps = 0;

                while (current != nullptr && steps < length)
                {
                    // printf("current->id = %d, current->cand_idx = %d\n", current->id, current->cand_idx);
                    partial_match[current->id] = current->cand_idx;
                    all_cand.insert(candidate_set_[current->id][current->cand_idx]);
                    current = current->parent;
                    steps++;
                }
            }

        private:
            bool isAmazonDataset() const
            {
                return dataset_ == "2core" || dataset_ == "amazon_single" || dataset_ == "amazon_multi";
            }

            dict CSInfo;
            DataGraph *data_;
            PatternGraph *query_;
            std::vector<std::vector<std::vector<std::vector<int>>>> candidate_neighbors;

            std::vector<std::vector<int>> candidate_set_;
            std::vector<int> neighbor_label_frequency;
            int num_candidate_vertex = 0, num_candidate_edge = 0;
            bool *in_neighbor_cs;
            bool **BitsetCS;
            bool **BitsetEdgeCS;
            int *num_visit_cs_;

            void BuildCS_original(PatternGraph *query);
            void BuildCS_baseline(PatternGraph *query, SubgraphMatchingOption opt);
            void BuildCS_optimized(PatternGraph *query, SubgraphMatchingOption opt);

            bool BuildInitialCS();

            void ConstructCS();

            bool InitRootCandidates(int root);

            bool RefineCS();

            void PrepareNeighborSafety(int cur);

            bool CheckNeighborSafety(int cur, int cand);

            bool NeighborBipartiteSafety(int cur, int cand);

            bool EdgeBipartiteSafety(int cur, int cand);

            inline bool EdgeCandidacy(int query_edge_id, int data_edge_id);

            bool TriangleSafety(int query_edge_id, int data_edge_id);

            bool FourCycleSafety(int query_edge_id, int data_edge_id);

            bool CheckSubStructures(int cur, int cand);

            bool NeighborFilter(int cur, int cand)
            {
                switch (opt.neighborhood_filter)
                {
                case NEIGHBOR_SAFETY:
                    return CheckNeighborSafety(cur, cand);
                case NEIGHBOR_BIPARTITE_SAFETY:
                    return NeighborBipartiteSafety(cur, cand);
                case EDGE_BIPARTITE_SAFETY:
                    return EdgeBipartiteSafety(cur, cand);
                }
            };

            bool StructureFilter(int query_edge_id, int data_edge_id)
            {
                switch (opt.structure_filter)
                {
                case NO_STRUCTURE_FILTER:
                    return true;
                case TRIANGLE_SAFETY:
                    return TriangleSafety(query_edge_id, data_edge_id);
                case FOURCYCLE_SAFETY:
                    return TriangleSafety(query_edge_id, data_edge_id) and FourCycleSafety(query_edge_id, data_edge_id);
                }
                return true;
            };

            void printTree(TreeNode *node, int depth = 0)
            {
                if (!node)
                    return;

                // 缩进表示层级结构
                for (int i = 0; i < depth; ++i)
                    std::cout << "  ";

                // 打印当前节点信息
                std::cout << "depth=" << depth << ", id=" << node->id << ", cand_idx=" << node->cand_idx << std::endl;

                // 递归打印所有子节点
                for (auto child : node->children)
                {
                    printTree(child, depth + 1);
                }
            };
        };

        inline CandidateSpace::CandidateSpace(DataGraph *data, SubgraphMatchingOption filter_option):rng_(std::random_device{}())
        {
            opt = filter_option;
            data_ = data;
            BitsetCS = new bool *[opt.MAX_QUERY_VERTEX];
            for (int i = 0; i < opt.MAX_QUERY_VERTEX; i++)
            {
                BitsetCS[i] = new bool[data->GetNumVertices()]();
            }
            BitsetEdgeCS = new bool *[opt.MAX_QUERY_EDGE];
            for (int i = 0; i < opt.MAX_QUERY_EDGE; i++)
            {
                BitsetEdgeCS[i] = new bool[data->GetNumEdges()]();
            }
            in_neighbor_cs = new bool[data->GetNumVertices()]();
            neighbor_label_frequency.resize(data->GetNumVertices());
            num_visit_cs_ = new int[data_->GetNumVertices()]();
            candidate_set_.resize(opt.MAX_QUERY_VERTEX);
            used_ids.assign(data->GetNumVertices(), false);
            std::fill(std::begin(partial_match_array), std::end(partial_match_array), -1);
            BPSolver.Initialize(50, data_->GetMaxDegree(), 50);
            startOracleWorker();
            std::cout << "Constructing Candidate Space: " << opt.MAX_QUERY_VERTEX << " " << opt.MAX_QUERY_EDGE << std::endl;
        }

        inline void CandidateSpace::initONNX()
        {
            
            if (isAmazonDataset()){
                getSessionForKey("00");
                getTokenizerForKey("00");

                getSessionForKey("01");
                getTokenizerForKey("01");

                getSessionForKey("10");
                getTokenizerForKey("10");

                getSessionForKey("11");
                getTokenizerForKey("11");

                getSessionForKey("20");
                getTokenizerForKey("20");

                getSessionForKey("21");
                getTokenizerForKey("21");


            }
            else{
                getSessionForKey("30");
                getTokenizerForKey("30");

                getSessionForKey("31");
                getTokenizerForKey("31");

                getSessionForKey("40");
                getTokenizerForKey("40");

                getSessionForKey("41");
                getTokenizerForKey("41");

            }
        }

        inline void CandidateSpace::getArgs(string vertex_review_path, string reviews_kcore_path,
                                            string modelLocation_path, string vertex_product_path, string product_meta_path,
                                            string image_path, int est_type, double gamma, string output_path,
                                            string root_name, string dataset)
        {
            vertex_review_path_ = vertex_review_path;
            reviews_kcore_path_ = reviews_kcore_path;
            modelLocation_path_ = modelLocation_path;
            vertex_product_path_ = vertex_product_path;
            image_path_ = image_path;
            est_type_ = est_type;
            gamma_ = gamma;
            root_name_ = root_name;
            output_path_ = output_path;
            product_kcore_path__ = product_meta_path;
            dataset_ = dataset;
            if (est_type == 0)
                query_type_precision = true;
            else
                query_type_precision = false;
        }

        inline CandidateSpace::~CandidateSpace()
        {
            for (int i = 0; i < opt.MAX_QUERY_VERTEX; i++)
            {
                delete[] BitsetCS[i];
            }
            delete[] BitsetCS;
            for (int i = 0; i < opt.MAX_QUERY_EDGE; i++)
            {
                delete[] BitsetEdgeCS[i];
            }
            // 停止 worker 并 flush 剩余任务
            stopOracleWorkerAndFlush();
            delete[] BitsetEdgeCS;
            delete[] num_visit_cs_;
            delete[] in_neighbor_cs;
        }

        inline bool CandidateSpace::exportMatchesToFile(std::vector<std::pair<TreeNode *, int>> full_match)
        {
            string output_file = output_path_;

            if (full_match.empty())
            {
                std::cout << "No matches to export." << std::endl;
                return false;
            }

            // 1. 使用缓冲写入提升 I/O 效率
            std::ofstream outfile(output_file, std::ios::out | std::ios::binary);
            if (!outfile.is_open())
            {
                std::cerr << "Error: Could not open file " << output_file << " for writing." << std::endl;
                return false;
            }

            int length = query_->GetNumVertices();

            // 2. 提前生成表头字符串，避免在循环中反复判断
            std::string header = "";
            for (int u = 0; u < length; ++u)
            {
                header += std::to_string(u) + (u == length - 1 ? "" : ",");
            }
            outfile << header << "\n";

            // 3. 核心优化：预分配一个固定大小的 vector 作为“行缓冲区”
            // 这样不需要为每个匹配创建 unordered_map，极大减少内存申请开销
            std::vector<int> row_buffer(length);

            // 增加内部缓冲区提升写入速度
            std::string line_buffer;
            line_buffer.reserve(length * 12); // 预留大概一行的字符空间

            for (auto &match_pair : full_match)
            {
                TreeNode *current = match_pair.first;

                // 4. 直接填充 row_buffer，代替创建 unordered_map
                int steps = 0;
                while (current != nullptr && steps < length)
                {
                    row_buffer[current->id] = candidate_set_[current->id][current->cand_idx];
                    current = current->parent;
                    steps++;
                }

                // 5. 快速构建行字符串
                line_buffer.clear();
                for (int i = 0; i < length; ++i)
                {
                    line_buffer += std::to_string(row_buffer[i]);
                    if (i < length - 1)
                        line_buffer += ",";
                }
                line_buffer += "\n";

                // 写入文件
                outfile.write(line_buffer.c_str(), line_buffer.size());
            }

            outfile.close();
            std::cout << "Successfully exported " << full_match.size() << " matches to " << output_file << std::endl;
            return true;
        }

        inline void CandidateSpace::BuildCS_original(PatternGraph *query)
        {
            CSInfo.clear();
            query_ = query;
            num_candidate_vertex = num_candidate_edge = 0;
            for (int i = 0; i < query_->GetNumVertices(); i++)
            {
                memset(BitsetCS[i], false, data_->GetNumVertices());
            }
            for (int i = 0; i < query_->GetNumEdges(); i++)
            {
                memset(BitsetEdgeCS[i], false, data_->GetNumEdges());
            }
            memset(num_visit_cs_, 0, data_->GetNumVertices());
            BPSolver.Reset();
            for (int i = 0; i < query_->GetNumVertices(); i++)
            {
                candidate_set_[i].clear();
            }
            Timer CStimer;
            CStimer.Start();
            BuildInitialCS();
            RefineCS();
            ConstructCS();
            CStimer.Stop();
            CSInfo["CSBuildTime"] = CStimer.GetTime();
            std::cout << "CSBuildTime: " << CStimer.GetTime() << " ms" << std::endl;
            CSInfo["#CSVertex"] = num_candidate_vertex;
            CSInfo["#CSEdge"] = num_candidate_edge;
        }

        inline void CandidateSpace::BuildCS_baseline(PatternGraph *query, SubgraphMatchingOption opt)
        {
            std::string strategy_str = (opt.match_order_strategy == MatchOrderStrategy::INFERENCE_FIRST)   ? "Inference first"
                                       : (opt.match_order_strategy == MatchOrderStrategy::STRUCTURE_FIRST) ? "Structure first"
                                                                                                           : "Inference on demand";
            cout << "Baseline method to build candidate space, strategy: " << strategy_str << endl;

            std::unordered_map<int, int> vertex_model_map;
            std::vector<double> model_mask;
            std::vector<int> match_order__;
            Timer CStimer;

            CStimer.Start();
            //load_model_assignment(modelLocation_path_, vertex_model_map, model_mask);
            if (isAmazonDataset()){
                load_model_assignment(modelLocation_path_, vertex_model_map, model_mask);
            }
            else{
                load_model_assignment_parler(modelLocation_path_, vertex_model_map, model_mask, root_name_, query_->GetNumVertices());
            }
            CStimer.Stop();
            std::cout << "Model_loc Load Time: " << CStimer.GetTime() << " ms" << std::endl;

            query_->predicateMap = vertex_model_map;
            buildMatchOrder(match_order__, query_, opt.match_order_strategy, vertex_model_map);
            match_order = match_order__;
            match_order_map.resize(query_->GetNumVertices());
            for (int i = 0; i < (int)match_order.size(); i++)
            {
                match_order_map[match_order[i]] = i;
            }
            query_->buildPreNeighbour(match_order__);
            threshold_vertex = model_mask;
            TreeNode treeNode_root = TreeNode(0, NodeType::TYPE_C, 0, 0);
            TrieRoot = &treeNode_root;

            if (opt.match_order_strategy != MatchOrderStrategy::STRUCTURE_FIRST)
            {
                CStimer.Start();
                constructScoreIndex_onnx_baseline();
                CStimer.Stop();
                std::cout << "Oracle Score Construct Time: " << CStimer.GetTime() << " ms" << std::endl;
            }

            CStimer.Start();
            baseline_subgraph_match_ArrayVersion(0, opt.match_order_strategy);
            std::fill(used_ids.begin(), used_ids.begin() + data_->GetNumVertices(), false);
            std::fill(std::begin(partial_match_array), std::end(partial_match_array), -1);
            CStimer.Stop();
            std::cout << "Baseline Subgraph Match Time: " << CStimer.GetTime() << " ms" << std::endl;
            cout << "leaf size:" << full_match_leaf.size() << endl;

            CStimer.Start();
            std::vector<std::pair<TreeNode *, int>> full_match;
            getFullMatch(full_match_leaf, full_match, opt.oracle_only, opt.match_order_strategy);
            exportMatchesToFile(full_match);
            CStimer.Stop();

            std::cout << "Get Full Match Time: " << CStimer.GetTime() << " ms" << std::endl;
            printf("full_match size = %zu\n", full_match.size());
        }

        inline void CandidateSpace::BuildCS_optimized(PatternGraph *query, SubgraphMatchingOption opt)
        {
            std::string strategy_str = (opt.match_order_strategy == MatchOrderStrategy::INFERENCE_FIRST)   ? "Inference first"
                                       : (opt.match_order_strategy == MatchOrderStrategy::STRUCTURE_FIRST) ? "Structure first"
                                                                                                           : "Inference on demand";
            cout << "Optimized method to build candidate space, strategy: " << strategy_str << endl;

            std::unordered_map<int, int> vertex_model_map;
            std::vector<double> model_mask;
            std::vector<int> match_order__;
            //load_model_assignment(modelLocation_path_, vertex_model_map, model_mask);
            for (int i = 0; i < query->GetNumVertices(); i++)
            {
                //print candidate_set size for each vertex
                cout << "Query vertex " << i << " candidate size: " << candidate_set_[i].size() << endl;
            }
            if (isAmazonDataset()){
                load_model_assignment(modelLocation_path_, vertex_model_map, model_mask);
            }
            else{
                load_model_assignment_parler(modelLocation_path_, vertex_model_map, model_mask, root_name_, query_->GetNumVertices());
            }
            query_->predicateMap = vertex_model_map;
            buildMatchOrder(match_order__, query_, opt.match_order_strategy, vertex_model_map);
            match_order = match_order__;
            match_order_map.resize(query_->GetNumVertices());
            for (int i = 0; i < (int)match_order.size(); i++)
            {
                match_order_map[match_order[i]] = i;
            }
            query_->buildPreNeighbour(match_order__);
            threshold_vertex = model_mask;
            TreeNode treeNode_root = TreeNode(0, NodeType::TYPE_C, 0, 0);
            TrieRoot = &treeNode_root;

            Timer CStimer;
            if (opt.match_order_strategy != MatchOrderStrategy::STRUCTURE_FIRST)
            {
                CStimer.Start();
                constructScoreIndex_onnx();
                CStimer.Stop();
                std::cout << "Proxy Score Construct Time: " << CStimer.GetTime() << " ms" << std::endl;
            }

            CStimer.Start();
            original_subgraph_match_ArrayVersion(0, opt.match_order_strategy);
            std::fill(used_ids.begin(), used_ids.begin() + data_->GetNumVertices(), false);
            std::fill(std::begin(partial_match_array), std::end(partial_match_array), -1);
            CStimer.Stop();
            std::cout << "Original Subgraph Match Time: " << CStimer.GetTime() << " ms" << std::endl;
            cout << "leaf size:" << full_match_leaf.size() << endl;

            // STRUCTURE_FIRST 需要先进行结构匹配，再用proxy score进行筛选
            std::vector<std::pair<TreeNode *, int>> full_match;
            if (opt.match_order_strategy == MatchOrderStrategy::STRUCTURE_FIRST)
            {
                CStimer.Start();
                auto active_candidates = collectActiveCandidates();
                constructScoreIndex_onnx_filtered(active_candidates);
                CStimer.Stop();
                std::cout << "Proxy Score Construct Time: " << CStimer.GetTime() << " ms" << std::endl;
            }
            CStimer.Start();
            threshold_vertex_new_global = threshold_vertex;
            getFullMatch(full_match_leaf, full_match, opt.oracle_only, opt.match_order_strategy);
            CStimer.Stop();
            std::cout << "Get Full Match Time: " << CStimer.GetTime() << " ms" << std::endl;
            printf("full_match size = %zu\n", full_match.size());

            //  输出结果
            //  打印theshold_vertex
            cout << "Initial threshold_vertex: ";
            for (double th : threshold_vertex)
            {
                cout << th << " ";
            }
            cout << endl;

            double Recall = 0.0;
            std::pair<double, double> Pre_;
            if (est_type_ == 0)
            {
                CStimer.Start();
                Pre_ = precision_Estimation(full_match, 1000);
                CStimer.Stop();
                std::cout << "Precision Estimation Time: " << CStimer.GetTime() << " ms" << std::endl;
                cout << "Precision: " << Pre_.first << endl;
            }
            else if (est_type_ == 1)
            {
                CStimer.Start();
                auto Rec_ = recall_Estimation_Wanderjoin(full_match, 1000, 0.0);
                CStimer.Stop();
                std::cout << "Recall Estimation Time: " << CStimer.GetTime() << " ms" << std::endl;
                cout << "Recall_wanderJoin: " << Rec_.first << endl;
                Recall = Rec_.first;
            }
            else if (est_type_ == 2)
            {
                CStimer.Start();
                auto Rec_2 = recall_Estimation_treeSampling_stratified(full_match, 1000, 0.0);
                CStimer.Stop();
                std::cout << "Recall_treeSampling Estimation Time: " << CStimer.GetTime() << " ms" << std::endl;
                cout << "Recall_treeSampling: " << Rec_2.first << endl;
                Recall = Rec_2.first;
            }

            std::unordered_set<int> models_all;
            int model_single_type = -1;
            int model_single = -1;
            for (auto &pair_q : query_->predicateMap)
            {
                models_all.insert(pair_q.second);
                model_single_type = pair_q.second;
                model_single = pair_q.first;
            }

            if (models_all.size() == 1) // only one model
            {
                double theta_left = 0.0;
                double theta_right = 1.0;
                double best_threshold = threshold_vertex[model_single];
                bool best_thes_occur = false;
                double best_acc = 0.0;
                double satisfied_threshold = 1.1;
                if (query_type_precision)
                {
                    if (Pre_.first < gamma_)
                    {
                        theta_left = threshold_vertex[model_single];
                        theta_right = 1.0;
                    }
                    else
                    {
                        theta_right = threshold_vertex[model_single];
                        theta_left = 0.0;
                        best_thes_occur = true;
                        best_acc = Pre_.first;
                    }
                }
                else
                {
                    if (Recall < gamma_)
                    {
                        theta_right = threshold_vertex[model_single];
                        theta_left = 0.0;
                    }
                    else
                    {
                        theta_left = threshold_vertex[model_single];
                        theta_right = 1.0;
                        best_thes_occur = true;
                        best_acc = Recall;
                    }
                }

                const int MAX_BAD = 3;
                int consecutive_infeasible = 0;
                while (!checkEstimationDone_single(Pre_.first, Recall, theta_left, theta_right))
                {
                    std::vector<double> threshold_vertex_new;
                    // changeThreshold_multi(Pre_.first, Rec_.first, threshold_vertex, threshold_vertex_new);
                    std::pair<vector<double>, int> ret_pair;

                    double theshold_single_old = threshold_vertex[model_single];
                    double theshold_single_new = tune_threshold_single(theshold_single_old, theta_right, gamma_, theta_left);
                    //if (consecutive_infeasible % 2 == 0){
                    //    theshold_single_new = 0.75;
                    //}
                    //else {
                    //    theshold_single_new = 0.85;
                    //}
                    // theshold_single_new = 0.0;
                    threshold_vertex_new = threshold_vertex;
                    // threshold_vertex_new[model_single] = theshold_single_new;
                    for (auto &pair_q : query_->predicateMap) // 给每个相同的model设置相同的阈值
                    {
                        int query_u = pair_q.first;
                        threshold_vertex_new[query_u] = theshold_single_new;
                    }

                    cout << "threshold_vertex_new: ";
                    for (double th : threshold_vertex_new)
                    {
                        cout << th << " ";
                    }
                    cout << endl;
                    threshold_vertex_new_global = threshold_vertex_new;
                    full_match.clear();
                    // full_match_leaf.clear();
                    precision_marginal.clear();
                    precision_marginal_frac.clear();
                    wanderJoin_recall_marginal.clear();
                    treeSampling_recall_marginal_all.clear();
                    oracle_scoremap.clear(); // 明天看看这里能不能行

                    CStimer.Start();
                    //threshold_vertex = threshold_vertex_new;
                    if (theshold_single_new > theshold_single_old)
                    {
                        progressive_subgraph_match_del(threshold_vertex_new);
                    }
                    else
                    {
                        progressive_subgraph_match_add(threshold_vertex_new, opt.match_order_strategy);
                    }
                    CStimer.Stop();
                    std::cout << "Progressive Subgraph Match Time: " << CStimer.GetTime() << " ms" << std::endl;
                    // recreate the full_match_leaf set
                    int final_u = match_order[match_order.size() - 1];
                    for (const auto &it : node_map[final_u])
                    {
                        if (it.second.second != nullptr)
                        {
                            if (it.second.second->type == NodeType::full_Match)
                            {
                                full_match_leaf.insert(it.first);
                            }
                        }
                    }
                    //threshold_vertex = threshold_vertex_new;
                    getFullMatch(full_match_leaf, full_match, opt.oracle_only, opt.match_order_strategy);
                    printf("full_match size = %zu\n", full_match.size());

                    if (est_type_ == 0)
                    {
                        CStimer.Start();
                        Pre_ = precision_Estimation(full_match, 1000);
                        CStimer.Stop();
                        std::cout << "Precision Estimation Time: " << CStimer.GetTime() << " ms" << std::endl;
                        std::cout << "Precision: " << Pre_.first << std::endl;
                    }
                    else if (est_type_ == 1)
                    {
                        CStimer.Start();
                        auto Rec_ = recall_Estimation_Wanderjoin(full_match, 1000, 0.0);
                        CStimer.Stop();
                        std::cout << "Recall Estimation Time: " << CStimer.GetTime() << " ms" << std::endl;
                        cout << "Final Recall_wanderJoin: " << Rec_.first << endl;
                        Recall = Rec_.first;
                    }
                    else if (est_type_ == 2)
                    {
                        CStimer.Start();
                        auto Rec_2 = recall_Estimation_treeSampling_stratified(full_match, 1000, 0.0);
                        CStimer.Stop();
                        std::cout << "Recall_treeSampling Estimation Time: " << CStimer.GetTime() << " ms" << std::endl;
                        cout << "Final Recall_treeSampling: " << Rec_2.first << endl;
                        Recall = Rec_2.first;
                    }

                    // 输出结果
                    cout << "Precision: " << Pre_.first << endl;
                    cout << "Recall: " << Recall << endl;

                    if (query_type_precision)
                    {
                        if ((!best_thes_occur) || (Pre_.first >= gamma_ && (satisfied_threshold > theshold_single_new || satisfied_threshold > 1)) || (Pre_.first < gamma_ && Pre_.first > best_acc))
                        {

                            if (Pre_.first >= gamma_){
                                satisfied_threshold = theshold_single_new;
                                best_threshold = theshold_single_new;
                                best_thes_occur = true;
                                best_acc = Pre_.first;
                            }
                            else{
                                if (satisfied_threshold < 1){
                                    // do nothing
                                }
                                else{
                                    best_threshold = theshold_single_new;
                                    best_thes_occur = true;
                                    best_acc = Pre_.first;
                                }
                            }
                        }
                    }
                    else
                    {
                        //if ((!best_thes_occur) || (Recall >= gamma_ && satisfied_threshold < theshold_single_new) || (Recall < gamma_ && Recall > best_acc))
                        //{
                        //    best_threshold = theshold_single_new;
                        //    best_thes_occur = true;
                        //    best_acc = Recall;
                        //}

                        if ((!best_thes_occur) || (Recall >= gamma_ && (satisfied_threshold < theshold_single_new) || satisfied_threshold > 1) || (Recall < gamma_ && Recall > best_acc))
                        {

                            if (Recall >= gamma_){
                                satisfied_threshold = theshold_single_new;
                                best_threshold = theshold_single_new;
                                best_thes_occur = true;
                                best_acc = Recall;
                            }
                            else{
                                if (satisfied_threshold < 1){ //
                                    // do nothing
                                }
                                else{
                                    best_threshold = theshold_single_new;
                                    best_thes_occur = true;
                                    best_acc = Recall;
                                }
                            }
                        }
                    }
                    
                    // 如果出现过满足precision/recall要求的阈值，并且当前区间小于0.05，则选择最佳阈值退出
                    if (best_thes_occur && ((theta_right - theta_left <= 0.05) || consecutive_infeasible >= MAX_BAD))
                    {
                        threshold_vertex[model_single] = best_threshold;
                        cout << "Best threshold" << best_threshold << endl;
                        cout << "With precision/recall: " << best_acc << endl;
                        break;
                    }

                    // 更新阈值区间
                    if (query_type_precision)
                    {
                        if (Pre_.first < gamma_)
                        {
                            theta_left = theshold_single_new;
                            theta_right = theta_right;
                            consecutive_infeasible++;
                        }
                        else
                        {
                            theta_right = theshold_single_new;
                            theta_left = theta_left;
                            best_thes_occur = true;
                            consecutive_infeasible = 0;
                        }
                    }
                    else
                    {
                        if (Recall < gamma_)
                        {
                            theta_left = theta_left;
                            theta_right = theshold_single_new;
                            consecutive_infeasible++;
                        }
                        else
                        {
                            theta_right = theta_right;
                            theta_left = theshold_single_new;
                            best_thes_occur = true;
                            consecutive_infeasible = 0;
                        }
                    }
                    threshold_vertex = threshold_vertex_new;
                }
            }
            else // multiple models
            {
                int cycle_times = 0;
                int max_cycle = 15;
                int model_count = models_all.size();

                while (!checkEstimationDone(Pre_.first, Recall))
                {
                    if (cycle_times >= max_cycle)
                    {
                        std::cout << "Reached maximum adjustment cycles." << std::endl;
                        break;
                    }
                    std::vector<double> threshold_vertex_new;
                    std::pair<vector<double>, int> ret_pair;
                    double final_accurate_score = 0.0;
                    if (est_type_ == 0)
                    {
                        ret_pair = aggregate_marginal_to_models(precision_marginal_frac, threshold_vertex, model_count);
                        final_accurate_score = Pre_.first;
                    }
                    else if (est_type_ == 1)
                    {
                        ret_pair = aggregate_marginal_to_models(wanderJoin_recall_marginal, threshold_vertex, model_count);
                        final_accurate_score = Recall;
                    }
                    else if (est_type_ == 2)
                    {
                        ret_pair = aggregate_marginal_to_models(treeSampling_recall_marginal_all, threshold_vertex, model_count);
                        final_accurate_score = Recall;
                    }
                    vector<double> r_marg = ret_pair.first;
                    int only_model = ret_pair.second;
                    if (ret_pair.second == 100)
                    {
                        break;
                    }
                    if (only_model != -1)
                    {
                        throw std::runtime_error("Error: only_model should be -1 for multiple models.");
                    }
                    else
                    {
                        threshold_vertex_new = tune_thresholds(threshold_vertex, final_accurate_score, r_marg, gamma_, 5);
                    }
                    cout << "threshold_vertex_new: ";
                    for (double th : threshold_vertex_new)
                    {
                        cout << th << " ";
                    }
                    cout << endl;
                    threshold_vertex_new_global = threshold_vertex_new;
                    full_match.clear();
                    // full_match_leaf.clear();
                    precision_marginal.clear();
                    precision_marginal_frac.clear();
                    wanderJoin_recall_marginal.clear();
                    treeSampling_recall_marginal.clear();
                    treeSampling_recall_marginal_all.clear();
                    oracle_scoremap.clear();

                    CStimer.Start();
                    if (query_type_precision)
                    {
                        progressive_subgraph_match_del(threshold_vertex_new);
                    }
                    else
                    {
                        progressive_subgraph_match_add(threshold_vertex_new, opt.match_order_strategy);
                    }
                    CStimer.Stop();
                    std::cout << "Progressive Subgraph Match Time: " << CStimer.GetTime() << " ms" << std::endl;
                    // recreate the full_match_leaf set
                    int final_u = match_order[match_order.size() - 1];
                    for (const auto &it : node_map[final_u])
                    {
                        if (it.second.second != nullptr)
                        {
                            if (it.second.second->type == NodeType::full_Match)
                            {
                                full_match_leaf.insert(it.first);
                            }
                        }
                    }
                    getFullMatch(full_match_leaf, full_match, opt.oracle_only, opt.match_order_strategy);
                    printf("full_match size = %zu\n", full_match.size());

                    if (est_type_ == 0)
                    {
                        CStimer.Start();
                        Pre_ = precision_Estimation(full_match, 1000);
                        CStimer.Stop();
                        std::cout << "Precision Estimation Time: " << CStimer.GetTime() << " ms" << std::endl;
                        std::cout << "Precision: " << Pre_.first << std::endl;
                    }
                    else if (est_type_ == 1)
                    {
                        CStimer.Start();
                        auto Rec_ = recall_Estimation_Wanderjoin(full_match, 1000, 0.0);
                        CStimer.Stop();
                        std::cout << "Recall Estimation Time: " << CStimer.GetTime() << " ms" << std::endl;
                        cout << "Final Recall_wanderJoin: " << Rec_.first << endl;
                        Recall = Rec_.first;
                    }
                    else if (est_type_ == 2)
                    {
                        CStimer.Start();
                        auto Rec_2 = recall_Estimation_treeSampling_stratified(full_match, 1000, 0.0);
                        CStimer.Stop();
                        std::cout << "Recall_treeSampling Estimation Time: " << CStimer.GetTime() << " ms" << std::endl;
                        cout << "Final Recall_treeSampling: " << Rec_2.first << endl;
                        Recall = Rec_2.first;
                    }

                    // 输出结果
                    cout << "Precision: " << Pre_.first << endl;
                    cout << "Recall: " << Recall << endl;
                    threshold_vertex = threshold_vertex_new;
                    cycle_times++;
                }
            }

            // 输出结果
            cout << "Final Precision: " << Pre_.first << endl;
            cout << "Final Recall: " << Recall << endl;

            // 导出匹配结果到文件
            getFullMatch(full_match_leaf, full_match, opt.oracle_only, opt.match_order_strategy);
            exportMatchesToFile(full_match);
        }

        inline bool CandidateSpace::BuildCS(PatternGraph *query, SubgraphMatchingOption opt)
        {
            BuildCS_original(query);

            if (opt.oracle_only)
            {
                BuildCS_baseline(query, opt);
            }
            else
            {
                BuildCS_optimized(query, opt);
            }
            return true;
        }

        // 回溯前 M 层，恢复已确认顶点的候选选择
        inline void recoverFixedAssignments(TreeNode *trie_node, int M, std::unordered_map<int, int> &fixed_assignments)
        {
            fixed_assignments.clear();
            TreeNode *cur = trie_node;
            int depth = 0;

            while (cur != nullptr && depth < M)
            {
                // cur->id 表示查询图的顶点 id
                // cur->cand_idx 表示该顶点选择的候选
                fixed_assignments[cur->id] = cur->cand_idx;

                cur = cur->parent; // 一层一层往上回溯
                depth++;
            }
        }

        inline void CandidateSpace::ConstructCS()
        {
            candidate_neighbors.clear();
            candidate_neighbors.resize(query_->GetNumVertices());
            std::vector<int> candidate_index(data_->GetNumVertices());
            for (int i = 0; i < query_->GetNumVertices(); ++i)
            {
                candidate_neighbors[i].resize(GetCandidateSetSize(i));
            }
            for (int u = 0; u < query_->GetNumVertices(); u++)
            {
                int u_label = query_->GetVertexLabel(u);
                int u_degree = query_->GetDegree(u);
                num_candidate_vertex += GetCandidateSetSize(u);
                for (int idx = 0; idx < GetCandidateSetSize(u); idx++)
                {
                    candidate_index[candidate_set_[u][idx]] = idx;
                    candidate_neighbors[u][idx].resize(query_->GetDegree(u));
                }

                for (int uc : query_->GetNeighbors(u))
                {
                    int query_edge_idx = query_->GetEdgeIndex(uc, u);
                    for (int vc_idx = 0; vc_idx < candidate_set_[uc].size(); ++vc_idx)
                    {
                        int vc = candidate_set_[uc][vc_idx];
                        for (int data_edge_idx : data_->GetIncidentEdges(vc, u_label))
                        {
                            int v = data_->GetOppositePoint(data_edge_idx);
                            if (data_->GetDegree(v) < u_degree)
                                break;
                            if (!BitsetEdgeCS[query_edge_idx][data_edge_idx])
                                continue;
                            num_candidate_edge++;
                            candidate_neighbors[u][candidate_index[v]][query_->GetAdjIdx(u, uc)].emplace_back(vc_idx);
                        }
                    }
                }
            }
        }

        inline int CandidateSpace::original_subgraph_match_ArrayVersion(int index, MatchOrderStrategy strategy, bool first_loop)
        {
            // 基础性能统计点（可选）
            int cur_u = match_order[index];
            std::vector<int> *nodelist = nullptr;
            std::vector<int> vec;

            // ---------------------------------------------------------
            // 1. 获取候选集并进行求交集过滤
            // ---------------------------------------------------------
            if (index == 0)
            {
                // 第一层直接使用该查询节点的所有候选
                nodelist = &candidate_set_[cur_u];
            }
            else if (index == match_order.size() && strategy == MatchOrderStrategy::STRUCTURE_FIRST)
            {
                int last_u = match_order[index - 1];
                //printf("proxy score: %f, threshold: %f\n", getProxyScore(last_u, partial_match_array[last_u]), threshold_vertex[last_u]);
                //printf("u: %d, partial match v: %d\n", last_u, partial_match_array[last_u]);
                addPartialMatch2Trie_ArrayVersion(index, full_match);
                total_matches_found++;
                if (total_matches_found % 10000 == 0)
                {
                    std::cout << "[Progress] Matches found: " << total_matches_found << std::endl;
                }

                return 0;
            }
            else
            {
                auto preNei = query_->getPreNeighbour(cur_u);

                // 取出第一个已匹配邻居对当前节点的约束列表
                auto cur_u_idx = query_->GetAdjIdx(preNei[0], cur_u);
                int matched_cand_idx = partial_match_array[preNei[0]];
                const auto &first_neighbors = candidate_neighbors[preNei[0]][matched_cand_idx][cur_u_idx];

                vec.clear();
                // 预分配空间减少扩容开销
                vec.reserve(first_neighbors.size());

                // 初始过滤：单射性检查 (Isomorphism Check)
                for (int cand_idx : first_neighbors)
                {
                    int cand_data_id = candidate_set_[cur_u][cand_idx];
                    // 使用数组 O(1) 检查该数据节点是否已被之前的查询节点占用
                    if (!used_ids[cand_data_id])
                    {
                        vec.push_back(cand_idx);
                    }
                }

                // 与剩下的已匹配邻居约束列表求交集
                for (size_t j = 1; j < preNei.size(); ++j)
                {
                    if (vec.empty())
                        break; // 提前剪枝

                    cur_u_idx = query_->GetAdjIdx(preNei[j], cur_u);
                    matched_cand_idx = partial_match_array[preNei[j]];
                    const auto &current_constraint = candidate_neighbors[preNei[j]][matched_cand_idx][cur_u_idx];

                    // 为了快速求交集，将当前约束列表存入临时 set (如果约束列表很大，这步仍可进一步优化)
                    std::unordered_set<int> constraint_set(current_constraint.begin(), current_constraint.end());

                    // 使用 remove_if 原地过滤 vec，避免频繁分配新 vector
                    auto it = std::remove_if(vec.begin(), vec.end(), [&](int idx) { return constraint_set.find(idx) == constraint_set.end(); });
                    vec.erase(it, vec.end());
                }
                nodelist = &vec;
            }

            // ---------------------------------------------------------
            // 2. 遍历候选节点并递归回溯
            // ---------------------------------------------------------
            int process = 0;
            for (int i = 0; i < nodelist->size(); i++)
            {
                if (first_loop && (double)i / nodelist->size() > process * 0.1)
                {
                    // 打印时间戳
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                    std::cout << std::put_time(std::localtime(&now_c), "%F %T") << " ";
                    cout << "processing node " << cur_u << ": " << (double)i / nodelist->size() * 100 << "%" << endl;
                    process++;
                }

                // 确定当前候选节点的索引
                int cand_idx = (index == 0) ? i : (*nodelist)[i];
                int cand_data_id = candidate_set_[cur_u][cand_idx];

                // --- [关键点] 修改全局状态 ---
                partial_match_array[cur_u] = cand_idx;
                used_ids[cand_data_id] = true;

                // 检查谓词条件 (Predicate Logic)
                if (strategy != MatchOrderStrategy::STRUCTURE_FIRST && query_->havePredicate(cur_u) &&
                    getProxyScore(cur_u, cand_idx) < threshold_vertex[cur_u])
                {
                    // 谓词失败，记录到 Trie 树并终止该路径
                    //std::cout << "Predicate failed for query node " << cur_u << " with candidate " << cand_data_id << std::endl;
                    //std::cout << "Proxy score: " << getProxyScore(cur_u, cand_idx) << ", Threshold: " << threshold_vertex[cur_u] << std::endl;
                    int cand_id = candidate_set_[cur_u][cand_idx];
                    int vtx = all_string_idx[getModelKey(cur_u)][cand_id];
                    std::string text = all_string_inputs[getModelKey(cur_u)][vtx];
                    //std::cout << "Candidate text: " << text << std::endl;
                    addPartialMatch2Trie_ArrayVersion(index + 1, failed_expension);
                }
                else
                {
                    // 检查是否已完成全图匹配
                    if (index + 1 == query_->GetNumVertices())
                    {
                        // 成功找到一个全匹配，记录到 Trie
                        addPartialMatch2Trie_ArrayVersion(index + 1, full_match);
                        // --- 新增：打印逻辑 ---
                        total_matches_found++;
                        if (total_matches_found % 10000 == 0)
                        {
                            std::cout << "[Progress] Matches found: " << total_matches_found << std::endl;
                        }
                    }
                    else
                    {
                        // 继续递归下一层
                        original_subgraph_match_ArrayVersion(index + 1, strategy, false);
                    }
                }

                // --- [关键点] 回溯：必须将状态改回初始值，否则会污染其他分支 ---
                used_ids[cand_data_id] = false;
                partial_match_array[cur_u] = -1;
            }

            return 0;
        }

        inline int CandidateSpace::baseline_subgraph_match_ArrayVersion(int index, MatchOrderStrategy strategy, bool first_loop)
        {
            // 基础性能统计点（可选）
            int cur_u = match_order[index];
            std::vector<int> *nodelist = nullptr;
            std::vector<int> vec;

            // ---------------------------------------------------------
            // 1. 获取候选集并进行求交集过滤
            // ---------------------------------------------------------
            if (index == 0)
            {
                // 第一层直接使用该查询节点的所有候选
                nodelist = &candidate_set_[cur_u];
            }
            else
            {
                auto preNei = query_->getPreNeighbour(cur_u);

                // 取出第一个已匹配邻居对当前节点的约束列表
                auto cur_u_idx = query_->GetAdjIdx(preNei[0], cur_u);
                int matched_cand_idx = partial_match_array[preNei[0]];
                const auto &first_neighbors = candidate_neighbors[preNei[0]][matched_cand_idx][cur_u_idx];

                vec.clear();
                // 预分配空间减少扩容开销
                vec.reserve(first_neighbors.size());

                // 初始过滤：单射性检查 (Isomorphism Check)
                for (int cand_idx : first_neighbors)
                {
                    int cand_data_id = candidate_set_[cur_u][cand_idx];
                    // 使用数组 O(1) 检查该数据节点是否已被之前的查询节点占用
                    if (!used_ids[cand_data_id])
                    {
                        vec.push_back(cand_idx);
                    }
                }

                // 与剩下的已匹配邻居约束列表求交集
                for (size_t j = 1; j < preNei.size(); ++j)
                {
                    if (vec.empty())
                        break; // 提前剪枝

                    cur_u_idx = query_->GetAdjIdx(preNei[j], cur_u);
                    matched_cand_idx = partial_match_array[preNei[j]];
                    const auto &current_constraint = candidate_neighbors[preNei[j]][matched_cand_idx][cur_u_idx];

                    // 为了快速求交集，将当前约束列表存入临时 set (如果约束列表很大，这步仍可进一步优化)
                    std::unordered_set<int> constraint_set(current_constraint.begin(), current_constraint.end());

                    // 使用 remove_if 原地过滤 vec，避免频繁分配新 vector
                    auto it = std::remove_if(vec.begin(), vec.end(), [&](int idx) { return constraint_set.find(idx) == constraint_set.end(); });
                    vec.erase(it, vec.end());
                }
                nodelist = &vec;
            }

            // collect oracle tasks
            // std::unordered_set<int> oracle_tasks;
            // std::unordered_set<int> cand_set;
            // all_string_inputs.resize(3);
            // all_string_idx.resize(3);
            // for (int i = 0; i < nodelist->size(); i++)
            // {
            //     int cand_idx = (index == 0) ? i : (*nodelist)[i];
            //     int cand_data_id = candidate_set_[cur_u][cand_idx];
            //     cand_set.insert(cand_idx);
            //     oracle_tasks.insert(cand_data_id);
            // }
            // std::vector<int> needed_vertex_ids(oracle_tasks.begin(), oracle_tasks.end());
            // if (strategy != MatchOrderStrategy::STRUCTURE_FIRST && query_->havePredicate(cur_u))
            // {
            //     int model_key = getModelKey(cur_u);
            //     std::vector<std::string> &vertex_to_text = all_string_inputs[model_key];
            //     std::unordered_map<int, int> &vertex_to_index = all_string_idx[model_key];
            //     switch (model_key)
            //     {
            //     case 0:
            //         buildCompactVectorMapping(needed_vertex_ids, vertex_to_text, vertex_to_index);
            //         break;
            //     case 1:
            //         getProductTextsForVertexIds(needed_vertex_ids, vertex_to_text, vertex_to_index, 512);
            //         break;
            //     case 2:
            //         getPicPathForVertexIds(needed_vertex_ids, vertex_to_text, vertex_to_index);
            //         break;
            //     default:
            //         throw std::runtime_error("Unknown model key in baseline_subgraph_match: " + std ::to_string(model_key));
            //     }

            //     oracle_no_more_requests.store(false);
            //     for (int v_idx : cand_set)
            //     {
            //         enqueueOracleTask(cur_u, v_idx, "1");
            //     }
            //     waitOracleComplete();
            //     oracle_no_more_requests.store(true);
            // }

            // ---------------------------------------------------------
            // 2. 遍历候选节点并递归回溯
            // ---------------------------------------------------------
            int process = 0;
            for (int i = 0; i < nodelist->size(); i++)
            {
                if (first_loop && (double)i / nodelist->size() > process * 0.1)
                {
                    // 打印时间戳
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                    std::cout << std::put_time(std::localtime(&now_c), "%F %T") << " ";
                    cout << "processing node " << cur_u << ": " << (double)i / nodelist->size() * 100 << "%" << endl;
                    process++;
                }

                // 确定当前候选节点的索引
                int cand_idx = (index == 0) ? i : (*nodelist)[i];
                int cand_data_id = candidate_set_[cur_u][cand_idx];

                // --- [关键点] 修改全局状态 ---
                partial_match_array[cur_u] = cand_idx;
                used_ids[cand_data_id] = true;

                // 检查谓词条件 (Predicate Logic)
                bool ok = true;
                if (strategy != MatchOrderStrategy::STRUCTURE_FIRST && query_->havePredicate(cur_u))
                {
                    // 根据 oracleRecord 的结果，判断谓词结果是否正确
                    if (oracleRecord.count(cur_u) && oracleRecord[cur_u].count(cand_idx))
                    {
                        if (!oracleRecord[cur_u][cand_idx])
                        {
                            // addPartialMatch2Trie_ArrayVersion(index + 1, failed_expension);
                            goto CONTINUE;
                        }
                        else
                        {
                            // ok
                            goto LOOP;
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Oracle result missing for vertex " + std::to_string(cur_u) + " candidate index " +
                                                 std::to_string(cand_idx));
                    }
                }

            LOOP:
            {
                // 检查是否已完成全图匹配
                if (index + 1 == query_->GetNumVertices())
                {
                    // 成功找到一个全匹配，记录到 Trie
                    addPartialMatch2Trie_ArrayVersion(index + 1, full_match);
                    // --- 新增：打印逻辑 ---
                    total_matches_found++;
                    if (total_matches_found % 10000 == 0)
                    {
                        std::cout << "[Progress] Matches found: " << total_matches_found << std::endl;
                    }
                }
                else
                {
                    // 继续递归下一层
                    baseline_subgraph_match_ArrayVersion(index + 1, strategy, false);
                }
            }

            // --- [关键点] 回溯：必须将状态改回初始值，否则会污染其他分支 ---
            CONTINUE:
                used_ids[cand_data_id] = false;
                partial_match_array[cur_u] = -1;
            }

            return 0;
        }

        inline bool CandidateSpace::getFullMatch(std::unordered_set<int> full_match_leaf, std::vector<std::pair<TreeNode *, int>> &full_match,
                                                 bool oracle_only, MatchOrderStrategy strategy)
        {
            full_match.clear();

            // 如果 full_match_leaf 为空，直接返回 false
            if (full_match_leaf.size() == 0)
            {
                return false;
            }

            const int last_order_idx = (int)match_order.size() - 1;
            const int cur = match_order[last_order_idx];
            all_string_inputs.resize(5);
            all_string_idx.resize(5);

            // task: map<u, set<v>>
            auto collectOracleTasks = [&](TreeNode *leaf, std::unordered_map<int, std::unordered_set<int>> &tasks)
            {
                TreeNode *node = leaf;
                for (int i = last_order_idx; i >= 0 && node != nullptr; --i)
                {
                    int u = match_order[i];
                    if (query_->havePredicate(u))
                    {
                        tasks[u].insert(node->cand_idx);
                    }
                    node = node->parent;
                }
            };

            auto enqueueOracleForMatch = [&](std::unordered_map<int, std::unordered_set<int>> &tasks)
            {
                std::vector<int> needed_vertex_ids;
                int first_model_key = -1;
                if (!isAmazonDataset()){
                    first_model_key = 3;
                }
                else{
                    first_model_key = 0;
                }
                for (int model_key = first_model_key; model_key < 5; ++model_key)
                {
                    needed_vertex_ids.clear();

                    std::vector<std::string> &vertex_to_text = all_string_inputs[model_key];
                    std::unordered_map<int, int> &vertex_to_index = all_string_idx[model_key];
                    for (const auto &task : tasks)
                    {
                        int u = task.first;
                        int real_model_key = getModelKey(u);
                        if (real_model_key != model_key)
                            continue;

                        for (int v_idx : task.second)
                        {
                            int v_id = candidate_set_[u][v_idx];
                            if (vertex_to_index.find(v_id) == vertex_to_index.end())
                            {
                                needed_vertex_ids.push_back(v_id);
                            }
                        }
                    }

                    if (needed_vertex_ids.empty())
                        continue;

                    switch (model_key)
                    {
                    case 0:
                    {
                        buildCompactVectorMapping(needed_vertex_ids, vertex_to_text, vertex_to_index);
                        break;
                    }
                    case 1:
                    {
                        getProductTextsForVertexIds(needed_vertex_ids, vertex_to_text, vertex_to_index, 512);
                        break;
                    }
                    case 2:
                    {
                        getPicPathForVertexIds(needed_vertex_ids, vertex_to_text, vertex_to_index);
                        break;
                    }
                    case 3:
                    {
                        buildCompactVectorMapping_parler(needed_vertex_ids, vertex_to_text, vertex_to_index);
                        break;
                    }
                    case 4:
                    {
                        getProductTextsForVertexIds_parler(needed_vertex_ids, vertex_to_text, vertex_to_index, 512);
                        break;
                    }
                    default:
                        throw std::runtime_error("Unknown model key in enqueueOracleForMatch: " + std ::to_string(model_key));
                    }
                }

                for (const auto &task : tasks)
                {
                    for (int v_idx : task.second)
                    {
                        enqueueOracleTask(task.first, v_idx, "1");
                    }
                }
            };

            auto findFailedOracleNode = [&](TreeNode *leaf) -> TreeNode *
            {
                TreeNode *node = leaf;
                for (int i = last_order_idx; i >= 0 && node != nullptr; --i)
                {
                    int u = match_order[i];
                    if (query_->havePredicate(u))
                    {
                        auto it_u = oracleRecord.find(u);
                        if (it_u == oracleRecord.end())
                            return node;
                        auto it_v = it_u->second.find(node->cand_idx);
                        if (it_v == it_u->second.end())
                            return node;
                        if (!it_v->second)
                            return node;
                    }
                    node = node->parent;
                }
                return nullptr;
            };

            // ===============================
            // Phase 1: STRUCTURE_FIRST + oracle_only
            // 先攒任务，再由 worker 批量推理并回填 oracleRecord。
            // ===============================
            if (strategy == MatchOrderStrategy::STRUCTURE_FIRST && oracle_only)
            {
                oracle_no_more_requests.store(false);
                {
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                    std::cout << std::put_time(std::localtime(&now_c), "%F %T") << ", before oracle task" << std::endl;
                }

                std::unordered_map<int, std::unordered_set<int>> tasks;
                for (const auto &cand_idx : full_match_leaf)
                {
                    TreeNode *treeN = node_map[cur][cand_idx].second;
                    while (treeN != nullptr)
                    {
                        collectOracleTasks(treeN, tasks);
                        treeN = treeN->candidate_ptr;
                    }
                }

                enqueueOracleForMatch(tasks);
                waitOracleComplete();

                {
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                    std::cout << std::put_time(std::localtime(&now_c), "%F %T") << ", finish oracle task" << std::endl;
                }
                oracle_no_more_requests.store(true);
            }

            // ===============================
            // Phase 2: 过滤 + 剪枝 + 收集 full_match
            // ===============================
            std::unordered_map<int, std::unordered_set<int>> failed_candidates;
            for (const auto &cand_idx : full_match_leaf)
            {
                TreeNode *treeN = node_map[cur][cand_idx].second;
                //if (cur == 0 && cand_idx == 8858){
                //    printf(" ");
                //}
                while (treeN != nullptr)
                {
                    if (strategy != MatchOrderStrategy::STRUCTURE_FIRST)
                    {
                        full_match.push_back(std::make_pair(treeN, last_order_idx));
                        treeN = treeN->candidate_ptr;
                        continue;
                    }

                    TreeNode *failed_node = oracle_only
                                                ? findFailedOracleNode(treeN)
                                                : findFailedProxyNode(treeN, threshold_vertex_new_global);
                    bool ok = failed_node == nullptr;

                    if (ok)
                    {
                        full_match.push_back(std::make_pair(treeN, last_order_idx));
                        treeN = treeN->candidate_ptr;
                    }
                    else
                    {
                        failed_candidates[failed_node->id].insert(failed_node->cand_idx);
                        treeN = treeN->candidate_ptr;
                    }
                }
            }

            for (const auto &[u, candidate_indices] : failed_candidates)
            {
                for (int failed_idx : candidate_indices)
                {
                    TreeNode *failed_chain_head = node_map[u][failed_idx].second;
                    if (failed_chain_head != nullptr)
                        TreeNode::removeRelativePath(failed_chain_head, DeleteMode::Explicit, node_map, full_match_leaf);
                }
            }

            // 成功获取所有匹配的节点
            cout << "Get full match size: " << full_match.size() << endl;
            return true;
        }

        // count计数有一个问题，就是一个节点的计数变了，它所有的父节点全都要跟着变，额外的遍历使得维护开销增大。
        inline TreeNode *CandidateSpace::addPartialMatch2Trie(std::unordered_map<int, int> &partial_match, partial_match_state state)
        {
            int cur_u = 0;
            TreeNode *new_node = nullptr;
            int index = partial_match.size() - 1;
            for (int i = partial_match.size() - 1; i >= 0; i--)
            {
                cur_u = this->match_order[i];
                TreeNode **subroot = &node_map[cur_u][partial_match[cur_u]].second;
                if ((*subroot) == nullptr && i != 0)
                { // ！！！这里按照删除处理的，后续同
                    continue;
                }
                else // 找到第一个在前缀树中存在候选的顶点，或者找到了根节点
                {
                    TreeNode *ret_node = nullptr;
                    bool res = false;
                    if ((*subroot) == nullptr && i == 0)
                    {
                        ret_node = TrieRoot;
                    }
                    else // 存在候选
                    {
                        // 找到哪个候选节点的前缀partialmatch和当前partial_match一致，返回这个节点的指针。
                        res = searchMatchTrieLeaf(partial_match, ret_node, i);
                        if (res == false)
                        {
                            // ret_node = TrieRoot;
                            continue;
                        }
                        else
                        {
                            ret_node->type = NodeType::TYPE_C;
                        }
                    }
                    TreeNode *n = ret_node;
                    ////////////////
                    for (int j = i; j >= 0; j--)
                    {
                        n->counter++;
                        if (state == GraphLib::SubgraphMatching::full_match)
                        {
                            n->full_match_counter++;
                        }
                    }
                    ///////////////////
                    TreeNode *lit = ret_node;
                    TreeNode *leaf = nullptr;

                    /*为后续顶点添加进trie*/
                    int base = i;
                    if (res == false)
                    {
                        base = 0; // 如果没有找到匹配的前缀，则从i-1开始添加
                    }
                    for (int j = base; j < partial_match.size(); j++)
                    {
                        int u = match_order[j];
                        int cand_v_idx = partial_match[u];
                        NodeType nt {};
                        // 用make_shared创建新节点
                        if (j == match_order.size() - 1 && state == GraphLib::SubgraphMatching::full_match)
                        {
                            new_node = new TreeNode(u, NodeType::full_Match);
                            new_node->cand_idx = cand_v_idx; // 设置候选集索引
                            leaf = new_node;
                            new_node->full_match_counter = 1;
                            full_match_leaf.insert(cand_v_idx);
                        }
                        else if (j == partial_match.size() - 1)
                        {
                            new_node = new TreeNode(u, NodeType::fail_Expension);
                            new_node->cand_idx = cand_v_idx; // 设置候选集索引
                            leaf = new_node;
                        }
                        else
                        {
                            new_node = new TreeNode(u, NodeType::TYPE_C);
                            new_node->cand_idx = cand_v_idx; // 设置候选集索引
                        }
                        // new_node = std::make_shared<TreeNode>(u, nt);
                        lit->counter++;
                        new_node->parent = lit; // 设置 parent 指针
                        // 添加到subroot的children中
                        lit->add_child(new_node);
                        // 向下继续，subroot 指向新创建的节点
                        lit = new_node;
                        if ((*subroot) == nullptr && i == 0)
                        {
                            node_map[u][cand_v_idx].second = lit;
                            node_map[u][cand_v_idx].second->candidate_ptr_fronter = nullptr;
                        }
                        else
                        {
                            if (node_map[u][cand_v_idx].second == nullptr)
                            {
                                node_map[u][cand_v_idx].second = lit;
                                // 这里需要设置为nullptr，因为这是一个新的节点，没有前缀树的候选指针
                                node_map[u][cand_v_idx].second->candidate_ptr_fronter = nullptr;
                            }
                            else
                            {
                                lit->candidate_ptr = node_map[u][cand_v_idx].second;
                                node_map[u][cand_v_idx].second->candidate_ptr_fronter = lit;
                                node_map[u][cand_v_idx].second = lit;
                            }
                        }
                    }
                    return new_node; // 如果不使用counter则可以直接返回。
                }
            }
            return new_node;
        }

        // 这个函数的功能：找到哪个候选节点的前缀partialmatch和当前partial_match一致，返回这个节点的指针。
        // 如果没有找到，则返回一个新的节点。
        inline bool CandidateSpace::searchMatchTrieLeaf(std::unordered_map<int, int> &partial_match, TreeNode *&ret, int lastNode_with_candidate)
        {
            int cur_u = match_order[lastNode_with_candidate];
            int cand_v_idx = partial_match[cur_u];

            ret = node_map[cur_u][cand_v_idx].second;
            TreeNode *next = ret;
            while (next != nullptr)
            {
                bool flag = true;
                TreeNode *cur_node = next;
                for (int i = lastNode_with_candidate; i >= 0; i--)
                {
                    if (!cur_node || cur_node->cand_idx != partial_match[match_order[i]])
                    {
                        flag = false;
                        break;
                    }
                    cur_node = cur_node->parent;
                }
                if (flag)
                {
                    ret = next;
                    if (ret != TrieRoot)
                    {
                        ret = ret->parent;
                    }
                    return true;
                }
                next = next->candidate_ptr;
            }

            if (node_map[cur_u][cand_v_idx].second == nullptr)
            {
                cout << "searchMatchTrieLeaf: not found, cur_u: " << cur_u << ", cand_v_idx: " << cand_v_idx << ", return nullptr" << endl;
            }
            return false;
        }

        inline int CandidateSpace::progressive_subgraph_match_add(std::vector<double> threshold_vertex_new, MatchOrderStrategy strategy)
        {
            // --- 关键：执行前彻底重置一次全局数组 ---
            // 确保没有被之前的运行（或者没回溯干净的错误）污染
            // 修正 partial_match_array (原生数组写法)
            std::fill(partial_match_array, partial_match_array + MAX_QUERY_NODES, -1);

            // 修正 used_ids (vector 标准写法)
            std::fill(used_ids.begin(), used_ids.end(), false);
            this->total_matches_found = 0; // 如果你加了计数器，这里也清零

            for (int i = 0; i < match_order.size(); i++)
            {
                int cur_u = match_order[i];
                double new_threshold = threshold_vertex_new[cur_u];
                double old_threshold = threshold_vertex[cur_u];

                auto it_low = map_Score[cur_u].lower_bound(old_threshold);
                auto it_up = map_Score[cur_u].upper_bound(new_threshold);

                for (auto it = it_low; it != it_up; ++it)
                {
                    int cand_v = it->second;
                    TreeNode *partial_chain_head = node_map[cur_u][cand_v].second;

                    while (partial_chain_head != nullptr)
                    {
                        // 1. 填充前缀
                        TreeNode *walker = partial_chain_head;
                        int order = i;

                        static std::vector<int> current_path_ids;
                        current_path_ids.clear();

                        while (walker != nullptr && walker != TrieRoot)
                        {
                            int u_in_path = match_order[order];
                            int v_idx = walker->cand_idx;
                            int v_data_id = candidate_set_[u_in_path][v_idx];

                            partial_match_array[u_in_path] = v_idx;
                            used_ids[v_data_id] = true;
                            current_path_ids.push_back(v_data_id);

                            walker = walker->parent;
                            order--;
                        }

                        // 2. 这里的入口要小心
                        // 此时 partial_match_array[match_order[0...i]] 已经填好并标记 used_ids
                        original_subgraph_match_ArrayVersion(i + 1, strategy, false);

                        // 3. --- 核心：回溯清理 ---
                        // 这里要严格清理！清理到 i 这一层
                        for (int k = 0; k <= i; ++k)
                        {
                            partial_match_array[match_order[k]] = -1;
                        }
                        for (int d_id : current_path_ids)
                        {
                            used_ids[d_id] = false;
                        }

                        partial_chain_head = partial_chain_head->candidate_ptr;
                    }
                }
            }
            return 0;
        }

        // 阈值升高结果减少
        inline int CandidateSpace::progressive_subgraph_match_del(std::vector<double> threshold_vertex_new)
        {
            for (int i = 0; i < match_order.size(); i++)
            {
                int cur_u = match_order[i];
                double new_threshold = threshold_vertex_new[cur_u];
                double old_threshold = threshold_vertex[cur_u];
                auto it_low = map_Score[cur_u].lower_bound(new_threshold); // 找到 ≤ 0.9 的第一个（即 <= 0.9 的最大值）
                auto it_up = map_Score[cur_u].upper_bound(old_threshold);  // 找到 < 0.5 的第一个（即 < 0.5 的最小值）
                for (auto &it = it_low; it != it_up; ++it)
                {
                    double cand_score = it->first;
                    int cand_v = it->second;
                    TreeNode *partial_chain_head = node_map[cur_u][cand_v].second;
                    if (partial_chain_head != nullptr)
                    { // ！！注意这里按照删除处理的
                        TreeNode::removeRelativePath(partial_chain_head, DeleteMode::Explicit, node_map, full_match_leaf);
                        //node_map[cur_u][cand_v].second = nullptr; // 删除后将指针置为nullptr
                        // printf("After removeRelativePath, node_map[%d][%d].second = %p\n", 1, 258,
                        // node_map[1][258].second);
                    }
                }
            }

            int u_last = match_order[match_order.size() - 1];
            // 重新修正 full_match_leaf

            for (auto it = full_match_leaf.begin(); it != full_match_leaf.end();)
            {
                int leaf = *it;
                if (node_map[u_last][leaf].second == nullptr || node_map[u_last][leaf].second->type == NodeType::fail_Expension)
                {
                    // printf("u_last: %d leaf: %d\n", u_last, leaf);
                    it = full_match_leaf.erase(it); // erase返回下一个迭代器
                }
                else
                {
                    ++it;
                }
            }
            return 0;
        }

        inline int CandidateSpace::constructScoreIndex_onnx()
        {
            if (isAmazonDataset()){
                buildModelInputForBatch("0",nullptr);
                buildModelInputForBatch("1",nullptr);
                buildModelInputForBatch("2",nullptr);
            }
            else{
                buildModelInputForBatch_parler("3");
                buildModelInputForBatch_parler("4");
            }
            return 0;
        }

        inline int CandidateSpace::constructScoreIndex_onnx_baseline()
        {
            if (isAmazonDataset()){
                buildModelInputForBatch_baseline("0",nullptr);
                buildModelInputForBatch_baseline("1",nullptr);
                buildModelInputForBatch_baseline("2",nullptr);
            }
            else{   
                buildModelInputForBatch_baseline("3", nullptr);
                buildModelInputForBatch_baseline("4", nullptr);
            }
            return 0;
        }

        inline std::unordered_map<int, std::unordered_set<int>> CandidateSpace::collectActiveCandidates()
        {
            std::unordered_map<int, std::unordered_set<int>> active;
            for (int order = 0; order < match_order.size(); ++order)
            {
                int u = match_order[order];
                if (!query_->havePredicate(u))
                    continue;

                int model_id = query_->predicateMap[u];
                for (auto &[cand_idx, meta] : node_map[u])
                {
                    TreeNode *chain = meta.second;
                    while (chain != nullptr)
                    {
                        active[model_id].insert(candidate_set_[u][cand_idx]);
                        chain = chain->candidate_ptr;
                    }
                }
            }
            return active;
        }

        inline int CandidateSpace::constructScoreIndex_onnx_filtered(std::unordered_map<int, std::unordered_set<int>> &active_candidates)
        {
            if (isAmazonDataset()){
                buildModelInputForBatch("0", &active_candidates[0]);
                buildModelInputForBatch("1", &active_candidates[1]);
                buildModelInputForBatch("2", &active_candidates[2]);
            }
            else{
                buildModelInputForBatch("3", &active_candidates[3]);
                buildModelInputForBatch("4", &active_candidates[4]);
            }
            return 0;
        }

        inline void GraphLib::SubgraphMatching::CandidateSpace::addoracle_scoremap_wanderjoin(int u, int v, int weight)
        {
            get<2>(oracle_scoremap[u]) += weight;
            if (oracleRecord[u].find(v) != oracleRecord[u].end())
            {
                if (/*Oracle为真*/ oracleRecord[u][v] == true) // oracle == true
                {
                    get<1>(oracle_scoremap[u]) += weight;
                    if (node_map[u][v].first > threshold_vertex[u]) // oracle == true && proxy == true
                    {
                        get<0>(oracle_scoremap[u]) += weight;
                    }
                }
            }
            else
            {
                throw std::runtime_error("Oracle record not found for given u and v.");
            }
        }

        inline double GraphLib::SubgraphMatching::CandidateSpace::getProxyScore(int u, int v)
        {
            
            return node_map[u][v].first;
        }

        inline std::pair<double, int> CandidateSpace::precision_Estimation(std::vector<std::pair<TreeNode *, int>> full_match, int sampleSize)
        {
            if (full_match.empty() || sampleSize == 0)
                return {0.0, 0};

            oracle_no_more_requests.store(false);

            // 1. 准备随机索引
            std::vector<size_t> indices(full_match.size());
            std::iota(indices.begin(), indices.end(), 0);
            //std::mt19937 gen(31);
            std::shuffle(indices.begin(), indices.end(), rng_);

            int total_needed = std::min(sampleSize, (int)full_match.size());
            int batch_size = oracle_batch_size; // 每一批处理 64 个样本
            int current_n = 0;
            int tp_count = 0;

            // 用来记录每个样本包含哪些 oracle 任务，便于后续校验
            std::vector<std::vector<std::pair<int, int>>> samples_metadata;

            for (int start = 0; start < total_needed; start += batch_size)
            {
                int end = std::min(start + batch_size, total_needed);
                int current_batch_count = end - start;

                // --- 步骤 A: 推送一波任务 ---
                for (int i = start; i < end; ++i)
                {
                    std::vector<std::pair<int, int>> single_sample_tasks;
                    TreeNode *elem = full_match[indices[i]].first;

                    // 溯源路径上的所有 Predicate
                    for (int j = (int)match_order.size() - 1; j >= 0; j--)
                    {
                        int u = match_order[j];
                        int v = elem->cand_idx;
                        if (query_->havePredicate(u))
                        {
                            single_sample_tasks.emplace_back(u, v);

                            // 异步入队
                            enqueueOracleTask(u, v, "1");
                        }
                        elem = elem->parent;
                    }
                    samples_metadata.push_back(single_sample_tasks);
                }

                // --- 步骤 B: 等待这一批推理完成 ---
                // 这里你可以通过 condition_variable 或简单的信号机制等待
                // 我们假设 waitOracleComplete() 会阻塞直到当前 Queue 清空
                waitOracleComplete();

                // --- 步骤 C: 实时计算 Precision 和 SE ---
                for (size_t i = current_n; i < samples_metadata.size(); ++i)
                {
                    bool is_tp = true;
                    for (auto &task : samples_metadata[i])
                    {
                        precision_marginal[task.first] += (oracleRecord[task.first][task.second] ? 1 : 0);
                        int v_id = candidate_set_[task.first][task.second];
                        int this_node_idx = all_string_idx[query_->predicateMap[task.first]][v_id];
                        std::string this_node_string = all_string_inputs[query_->predicateMap[task.first]][this_node_idx];
                        //std::cout << "u: " << task.first << " v_idx: " << task.second << " v_id: " << v_id << " str: " << this_node_string <<std::endl; 
                        //std::cout << "proxy[u][v]= : " << node_map[task.first][task.second].first << std::endl;
                        //if (oracleRecord[task.first].find(task.second) == oracleRecord[task.first].end())
                        //{
                        //    throw std::runtime_error("Oracle record missing for u=" + std::to_string(task.first) +
                        //                             ", v=" + std::to_string(task.second));
                        //}
                        //std::cout << "predictMap: " << query_->predicateMap[task.first] << std::endl;
                        //std::cout << "oracle[u][v]= : " << oracleRecord[task.first][task.second] << std::endl;
                        // 检查 oracleRecord (确保你的记录是线程安全的)
                        if (!oracleRecord[task.first][task.second])
                        {
                            is_tp = false;
                            // break;
                        }
                    }
                    if (is_tp)
                        tp_count++;
                }
                current_n = (int)samples_metadata.size();

                // --- 步骤 D: 统计学终止检查 ---
                double p = (double)tp_count / current_n;
                double se = std::sqrt((p * (1.0 - p)) / current_n);

                // 如果样本量超过一定阈值 (比如 100) 且误差已经很小，提前退出
                if (current_n >= 128 && se < 0.015)
                {
                    printf("Early stopping at n=%d, Precision=%.4f, SE=%.4f\n", current_n, p, se);
                    break;
                }

                // printf("Current n=%d, Precision=%.4f, SE=%.4f\n", current_n, p, se);
            }

            oracle_no_more_requests.store(true);
            double final_p = (double)tp_count / current_n;
            for (auto &pair : precision_marginal)
            {
                int u = pair.first;
                double marginal_p = (double)pair.second / current_n;
                precision_marginal_frac[u] = marginal_p;
            }
            return std::make_pair(final_p, static_cast<int>(final_p * full_match.size()));
        }

        inline TreeNode *CandidateSpace::findFailedProxyNode(TreeNode *full_match_leaf, const std::vector<double> &threshold_vertex)
        {
            for (int i = match_order.size() - 1; i >= 0; i--)
            {
                if (query_->havePredicate(match_order[i]))
                {
                    double proxy_score = getProxyScore(match_order[i], full_match_leaf->cand_idx);
                    if (proxy_score < threshold_vertex[match_order[i]])
                    {
                        return full_match_leaf;
                    }
                }
                full_match_leaf = full_match_leaf->parent;
            }
            return nullptr;
        }

        inline std::pair<double, double> CandidateSpace::recall_Estimation_Wanderjoin(std::vector<std::pair<TreeNode *, int>> partial_match,
                                                                                      int sampleSize, double tp)
        {

            oracle_no_more_requests.store(false);

            // 基础参数初始化
            long ht_estimate = 0; // 对应你原本的 ht_estimate 逻辑
            tp = 0.0;
            //static std::mt19937 gen(25);

            // 维持你原本的向量类型，用于方差函数
            std::vector<int> sampleWeight;     // w_vec
            std::vector<int> weighted_tp_list; // y_vec (a_i)
            std::vector<int> proxy_list;       // x_vec (c_i)
            int total_processed = 0;
            const int batch_size = oracle_batch_size;
            double R_hat = 0.0;
            double variance = 0.0;

            for (int u_order = 0; u_order < match_order.size(); u_order++) // 遍历匹配顺序中的每个顶点
            {
                int u = match_order[u_order];
                if (!query_->havePredicate(u)) // 如果没有谓词
                {
                    continue; // 跳过这个顶点
                }
                for (int v_idx = 0; v_idx < candidate_set_[u].size(); v_idx++)
                {

                    int v = candidate_set_[u][v_idx];
                    TreeNode *next = node_map[u][v_idx].second;
                    while (next != nullptr)
                    {
                        if (next->type == NodeType::fail_Expension)
                        {
                            partial_match.emplace_back(std::make_pair(next, u_order));
                        }                           // 将匹配的节点和对应的候选索引存入 partial_match
                        next = next->candidate_ptr; // 获取下一个候选节点
                    }
                }
            }
            printf("partial_match.size(): %d\n", partial_match.size());
            std::uniform_int_distribution<size_t> dis(0, partial_match.size() - 1);

            while (total_processed < sampleSize)
            {

                int current_batch = std::min(batch_size, sampleSize - total_processed);
                // 1. 采样与任务分发 (保持你原本的 tuple 逻辑)
                std::vector<std::unordered_map<int, int>> batch_paths;
                std::vector<std::tuple<bool, int, bool>> batch_tuples;
                for (int i = 0; i < current_batch; ++i)
                {

                    int rand_idx = dis(rng_);
                    std::unordered_map<int, int> path_nodes;
                    std::unordered_set<int> all_cand;
                    getpartialMatch(partial_match[rand_idx].second + 1, partial_match[rand_idx].first, path_nodes, all_cand);
                    int state_t = 0;
                    auto sample_ = subgraphSampling_withReplacement(

                        partial_match[rand_idx].first->id, partial_match[rand_idx].first->cand_idx, partial_match[rand_idx].second + 1,
                        partial_match.size(), path_nodes, state_t, all_cand);

                    if (get<1>(sample_) != 0)
                    {

                        batch_paths.push_back(path_nodes);
                        batch_tuples.push_back(sample_);

                        // 推送 Oracle 任务
                        for (const auto &item : path_nodes)
                        {
                            if (query_->havePredicate(item.first))
                            {
                                enqueueOracleTask(item.first, item.second, "1");
                            }
                        }
                    }
                }

                // 2. 批量推理等待
                {
                    std::scoped_lock<std::mutex> lk(oracle_mutex);
                    oracle_cv.notify_all();
                }

                waitOracleComplete();

                // 3. 结果汇总 (严格遵循你原本的 Oracle_flag 判断逻辑)
                for (size_t i = 0; i < batch_paths.size(); ++i)
                {

                    bool Oracle_flag = true;
                    auto &path = batch_paths[i];
                    int weight = get<1>(batch_tuples[i]);
                    bool proxy_res = get<2>(batch_tuples[i]);
                    for (const auto &item : path)
                    {
                        if (query_->havePredicate(item.first))
                        {
                            addoracle_scoremap_wanderjoin(item.first, item.second, weight);
                            int v_id = candidate_set_[item.first][item.second];
                            int this_node_idx = all_string_idx[query_->predicateMap[item.first]][v_id];
                            std::string this_node_string = all_string_inputs[query_->predicateMap[item.first]][this_node_idx];
                            // std::cout << "u: " << item.first << " v_idx: " << item.second << " v_id: " << v_id << " str: " << this_node_string <<
                            // std::endl; std::cout << "proxy[u][v]= : " << node_map[item.first][item.second].first << std::endl;
                            if (!oracleRecord[item.first][item.second])
                            {

                                Oracle_flag = false;
                                // break;
                            }
                        }
                    }
                    // --- 核心计算逻辑：同步 R_hat 定义 ---
                    if (Oracle_flag)
                    {
                        ht_estimate += weight; // 分母：真实存在的匹配
                        if (proxy_res)
                        {
                            tp += weight; // 分子：真实存在且 Proxy 识别成功 (TP)
                        }
                    }

                    // 填充方差向量 (这里的 y_vec 和 x_vec 必须与上面的 tp/ht_estimate 逻辑对齐)
                    sampleWeight.push_back(weight);
                    weighted_tp_list.push_back((Oracle_flag && proxy_res) ? 1 : 0); // 对应分子累加项
                    proxy_list.push_back(Oracle_flag ? 1 : 0);                      // 对应分母累加项
                }

                total_processed += current_batch;

                // 4. 增量检测：计算当前 R_hat 和 方差

                if (ht_estimate > 0 && total_processed >= 128)
                {

                    R_hat = (double)tp / ht_estimate;

                    variance = calculateVariance_Wanderjoin(sampleWeight, weighted_tp_list, proxy_list, R_hat);

                    // 如果标准误达到 1% (或你自定义的阈值)，提前终止

                    if (check_stopCondition(variance, R_hat, 0.95))
                    {

                        printf("[Wanderjoin] Converged early at n=%d, R_hat=%.4f\n", total_processed, R_hat);

                        break;
                    }
                }
            }

            oracle_no_more_requests.store(true);

            for (auto const &[u, scores] : oracle_scoremap)
            {
                // 解包 tuple: <ProxyTrue_And_OracleTrue, OracleTrue_Total, Sample_Total>
                double tp_weight = static_cast<double>(std::get<0>(scores));
                double oracle_true_weight = static_cast<double>(std::get<1>(scores));

                if (oracle_true_weight > 1e-9)
                {
                    // 计算 P(Proxy=True | Oracle=True)
                    wanderJoin_recall_marginal[u] = tp_weight / oracle_true_weight;
                }
                else
                {
                    /* 边缘情况处理：
                       如果 oracle_true_weight 为 0，说明在采样路径中，该谓词从未通过 Oracle 验证。
                       这通常意味着该谓词过滤极强，或者该分枝下根本没有真匹配。
                       此时设为 1.0 代表“在此样本空间内没有发生误删”。
                    */
                    wanderJoin_recall_marginal[u] = 1.0;
                }
            }

            return std::make_pair(R_hat, variance);
        }

        inline std::tuple<bool, int, bool> CandidateSpace::subgraphSampling_withReplacement(int failed_cur, int failed_cand, int depth, int weight,
                                                                                            std::unordered_map<int, int> &partial_match_empty,
                                                                                            int state, std::unordered_set<int> &all_cand)
        {

            // if (partial_match_empty.size() == 3)
            //{
            //     printf("subgraphSampling_withReplacement: failedNode->id = %d, failedNode->cand_idx = %d, depth = %d,
            //     weight = %d\n", failedNode->id, failedNode->cand_idx, depth, weight);
            // }

            // if (query_->havePredicate(match_order[depth]))
            //{
            //     addoracle_scoremap_wanderjoin(match_order[depth], failedNode->cand_idx);
            // }

            // 基本情况：到达匹配顺序的末尾
            if (depth == match_order.size())
            {

                bool all_success = true;
                int cur = 0;
                bool flagg = true;
                // vector<int> result_candidate_single;
                for (int i = 0; i < match_order.size(); ++i)
                {
                    int u = match_order[match_order.size() - 1 - i];
                    cur = partial_match_empty[u];
                    // printf("u = %d, cur = %d\n", u, cur);
                    if (query_->havePredicate(u))
                    {
                        /*if (!getOracle(u, cur))
                        {
                            all_success = false;
                            flagg = false; // 如果有谓词而且oracle为false，则不再向上遍历
                        }*/
                        enqueueOracleTask(u, cur, "1");
                    }
                    // addoracle_scoremap_wanderjoin(u, cur, weight);
                }
                // result_candidates.push_back(result_candidate_single);
                return std::make_tuple(all_success, weight, (state >= 0));
            }

            state = -1;
            int u_cur = failed_cur;
            int u_next = match_order[depth];
            int v_cur_cand_idx = failed_cand;

            // 检查是否有子节点可以继续
            auto preNei = query_->getPreNeighbour(u_next);
            auto cur_u_idx = query_->GetAdjIdx(preNei[0], u_next);

            std::unordered_set<int> result(candidate_neighbors[preNei[0]][partial_match_empty[preNei[0]]][cur_u_idx].begin(),
                                           candidate_neighbors[preNei[0]][partial_match_empty[preNei[0]]][cur_u_idx].end());
            for (size_t j = 1; j < preNei.size(); ++j)
            {
                cur_u_idx = query_->GetAdjIdx(preNei[j], u_next);
                std::unordered_set<int> current_set(candidate_neighbors[preNei[j]][partial_match_empty[preNei[j]]][cur_u_idx].begin(),
                                                    candidate_neighbors[preNei[j]][partial_match_empty[preNei[j]]][cur_u_idx].end());
                for (auto it = result.begin(); it != result.end();)
                {
                    int cand_v_id = candidate_set_[u_next][*it];
                    if (current_set.find(*it) == current_set.end() || all_cand.find(cand_v_id) != all_cand.end()) // 同态结果舍去，只保留同构
                    {
                        it = result.erase(it); // 如果不在 current_set 中，则删除
                    }
                    else
                    {
                        ++it;
                    }
                }
            }

            if (depth < match_order.size() && result.size() == 0) //!!
            {
                return std::make_tuple(false, 0, -1);
            }

            std::vector<int> vec(result.begin(), result.end());

            // 随机选择一个子节点继续递归
            //static std::random_device rd;
            // static std::mt19937 gen(rd());
            //static std::mt19937 gen(14);
            std::uniform_int_distribution<size_t> dis(0, vec.size() - 1);
            auto next_child_idx = dis(rng_);
            // if (vec.size() > 1)
            //{
            //     printf("right\n");
            // }
            int next = vec[next_child_idx];     // 获取下一个节点
            partial_match_empty[u_next] = next; // 将下一个节点添加到单个部分匹配中
            int v_next_cand_idx = candidate_set_[u_next][next];
            all_cand.insert(v_next_cand_idx); // 加入已匹配节点集合
            // printf("subgraphSampling_withReplacement: next = %d, depth = %d, weight = %d\n", next, depth + 1, weight
            // * vec.size()); printf("subgraphSampling_withReplacement: next->cand_idx = %d, next->id = %d\n", next,
            // u_next);
            return subgraphSampling_withReplacement(u_next, next, depth + 1, weight * vec.size(), partial_match_empty, state, all_cand);
        }

        // 使用累积概率快速加权采样
        inline int CandidateSpace::weighted_sample(const std::vector<double> &weights, std::mt19937 &rng)
        {
            std::vector<double> cdf(weights.size());
            partial_sum(weights.begin(), weights.end(), cdf.begin());
            double total = cdf.back();
            for (double &x : cdf)
                x /= total;

            uniform_real_distribution<> dist(0.0, 1.0);
            double u = dist(rng_);
            return lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin();
        }

        inline std::tuple<double, double, int, int> CandidateSpace::getSingleNodeEstimation()
        {
            double min_ratio1 = numeric_limits<double>::max();
            int min_vertex1 = -1;

            double min_ratio2 = numeric_limits<double>::max();
            int min_vertex2 = -1;

            for (const auto &pair : oracle_scoremap)
            {
                int u = pair.first;
                if (!query_->havePredicate(u))
                {
                    continue; // 只考虑有谓词的节点
                }
                double a, b, c;
                tie(a, b, c) = pair.second;

                if (b != 0)
                {
                    double ratio1 = static_cast<double>(a) / b;
                    if (ratio1 < min_ratio1)
                    {
                        min_ratio1 = ratio1;
                        min_vertex1 = pair.first;
                    }
                }

                if (c != 0)
                {
                    double ratio2 = static_cast<double>(a) / c;
                    if (ratio2 < min_ratio2)
                    {
                        min_ratio2 = ratio2;
                        min_vertex2 = pair.first;
                    }
                }
            }
            return std::make_tuple(min_ratio1, min_ratio2, min_vertex1, min_vertex2);
        }

        inline bool CandidateSpace::changeThreshold_multi(double precision, double recall, std::vector<double> &threshold,
                                                          std::vector<double> &threshold_new)
        {
            threshold_new = threshold;
            auto tuple_ = getSingleNodeEstimation();
            // 这里再仔细调节。
            if (/*查询是precision约束，recall最大化*/ true)
            {
                double min_ratio1 = std::get<0>(tuple_);
                int min_vertex1 = std::get<2>(tuple_);
                if (threshold[min_vertex1] < 0.75)
                {
                    threshold_new[min_vertex1] += 0.48; // 减小阈值
                }
                else if (threshold[min_vertex1] < 0.85)
                {
                    threshold_new[min_vertex1] += 0.05; // 增大阈值
                }
                else if (threshold[min_vertex1] < 0.95)
                {
                    threshold_new[min_vertex1] += 0.01; // 增大阈值
                }
                else
                {
                    threshold_new[min_vertex1] += 0.001; // 增大阈值
                }
            }
            else
            { /*查询是recall约束，但是precision最大化*/
                double min_ratio2 = std::get<1>(tuple_);
                int min_vertex2 = std::get<3>(tuple_);
                if (threshold[min_vertex2] > 0.75)
                {
                    threshold_new[min_vertex2] -= 0.1; // 减小阈值
                }
                else if (threshold[min_vertex2] > 0.85)
                {
                    threshold_new[min_vertex2] -= 0.05; // 增大阈值
                }
                else if (threshold[min_vertex2] > 0.95)
                {
                    threshold_new[min_vertex2] -= 0.01; // 增大阈值
                }
                else
                {
                    threshold_new[min_vertex2] -= 0.1; // 增大阈值
                }
            }
            return false;

            // 后续修改的办法：
            // 使用
        }

        inline void CandidateSpace::selectStartVertex(std::vector<int> &match_order, PatternGraph *query,
                                                      std::unordered_map<int, int> &vertex_model_map, std::unordered_set<int> &visited_vertex,
                                                      bool prefer_model)
        {
            size_t n = query->GetNumVertices();
            std::vector<int> all_vertices(n);
            for (size_t i = 0; i < n; ++i)
            {
                all_vertices[i] = i;
            }

            auto is_valid = [&](int u)
            {
                return candidate_set_[u].size() > 0;
            };

            int start = -1;
            size_t min_cand_size = std::numeric_limits<size_t>::max();

            for (int u : all_vertices)
            {
                if (!is_valid(u))
                    continue;
                size_t csize = candidate_set_[u].size();

                if (prefer_model == vertex_model_map.count(u) && csize < min_cand_size)
                {
                    start = u;
                    min_cand_size = csize;
                }
            }

            // not found:
            // 1) prefer_model=true but no model vertex found
            // 2) prefer_model=false but all model vertex
            if (start == -1)
            {
                for (int u : all_vertices)
                {
                    if (!is_valid(u))
                        continue;
                    size_t csize = candidate_set_[u].size();
                    if (csize < min_cand_size)
                    {
                        start = u;
                        min_cand_size = csize;
                    }
                }
            }

            if (start == -1)
            {
                throw std::runtime_error("No valid starting vertex found.");
            }

            match_order.push_back(start);
            visited_vertex.insert(start);
        }

        inline void CandidateSpace::selectNextVertex(std::vector<int> &match_order, PatternGraph *query,
                                                     std::unordered_map<int, int> &vertex_model_map, std::unordered_set<int> &visited_vertex,
                                                     bool prefer_model)
        {

            size_t num_query_graph_vertex = query->GetNumVertices();
            size_t num_no_predicate_vertex = num_query_graph_vertex - vertex_model_map.size();

            std::vector<int> all_vertices(num_query_graph_vertex);
            for (size_t i = 0; i < num_query_graph_vertex; ++i)
            {
                all_vertices[i] = i;
            }

            auto is_valid = [&](int u)
            {
                return candidate_set_[u].size() > 0;
            };

            while (match_order.size() < num_query_graph_vertex)
            {
                std::vector<int> candidates;

                for (int u : match_order)
                {
                    for (int nb : query->GetNeighbors(u))
                    {
                        if (visited_vertex.count(nb))
                            continue;
                        candidates.push_back(nb);
                    }
                }

                // 去重
                std::sort(candidates.begin(), candidates.end());
                candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

                if (candidates.empty())
                {
                    // 图可能不是连通图，挑一个未访问的节点继续
                    for (int u : all_vertices)
                    {
                        if (!visited_vertex.count(u))
                        {
                            match_order.push_back(u);
                            visited_vertex.insert(u);
                            break;
                        }
                    }
                    continue;
                }

                // 在 candidates 中选一个最佳点
                int best = -1;
                size_t best_size = std::numeric_limits<size_t>::max();

                for (int u : candidates)
                {
                    if (!is_valid(u))
                        continue;

                    size_t csize = candidate_set_[u].size();
                    if (prefer_model == vertex_model_map.count(u))
                    {
                        if (csize < best_size || best == -1)
                        {
                            best = u;
                            best_size = csize;
                        }
                    }
                }

                if (best == -1)
                {
                    for (int u : candidates)
                    {
                        if (!is_valid(u))
                            continue;

                        // structure_first 时，优先选择没有推理谓词的点
                        if (!prefer_model && match_order.size() < num_no_predicate_vertex && vertex_model_map.count(u))
                            continue;
                        // inference_first 时，优先选择有推理谓词的点
                        if (prefer_model && match_order.size() < vertex_model_map.size() && !vertex_model_map.count(u))
                            continue;

                        size_t csize = candidate_set_[u].size();
                        if (csize < best_size || best == -1)
                        {
                            best = u;
                            best_size = csize;
                        }
                    }
                }

                if (best == -1)
                {
                    for (int u : candidates)
                    {
                        if (!is_valid(u))
                            continue;
                        size_t csize = candidate_set_[u].size();
                        if (csize < best_size || best == -1)
                        {
                            best = u;
                            best_size = csize;
                        }
                    }
                }

                if (best == -1)
                {
                    throw std::runtime_error("No valid next vertex found.");
                }

                match_order.push_back(best);
                visited_vertex.insert(best);
            }
        }

        inline void CandidateSpace::buildMatchOrderInferenceOnDemand(std::vector<int> &match_order, PatternGraph *query,
                                                                     std::unordered_map<int, int> &vertex_model_map)
        {
            int n = query->GetNumVertices();
            std::unordered_set<int> visited;

            // 所有顶点
            std::vector<int> all_vertices(n);
            for (int i = 0; i < n; ++i)
                all_vertices[i] = i;

            // 选择起始节点
            auto is_valid = [&](int u)
            {
                return candidate_set_[u].size() > 0;
            };

            int start = -1;
            size_t min_cand_size = std::numeric_limits<size_t>::max();

            for (int u : all_vertices)
            {
                if (!is_valid(u))
                    continue;
                size_t csize = candidate_set_[u].size();

                if (vertex_model_map.count(u) && csize < min_cand_size)
                {
                    start = u;
                    min_cand_size = csize;
                }
            }

            // 如果 type==0 但找不到满足 vertex_model_map 的点，就退化选择所有点里 cand 最小的
            if (start == -1)
            {
                for (int u : all_vertices)
                {
                    if (!is_valid(u))
                        continue;
                    size_t csize = candidate_set_[u].size();
                    if (csize < min_cand_size)
                    {
                        start = u;
                        min_cand_size = csize;
                    }
                }
            }

            if (start == -1)
            {
                throw std::runtime_error("No valid starting vertex found.");
            }

            match_order.push_back(start);
            visited.insert(start);

            // 不断添加邻接节点
            while (match_order.size() < static_cast<size_t>(n))
            {
                std::vector<int> candidates;

                for (int u : match_order)
                {
                    for (int nb : query->GetNeighbors(u))
                    {
                        if (visited.count(nb))
                            continue;
                        candidates.push_back(nb);
                    }
                }

                // 去重
                std::sort(candidates.begin(), candidates.end());
                candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

                if (candidates.empty())
                {
                    // 图可能不是连通图，挑一个未访问的节点继续
                    for (int u : all_vertices)
                    {
                        if (!visited.count(u))
                        {
                            match_order.push_back(u);
                            visited.insert(u);
                            break;
                        }
                    }
                    continue;
                }

                // 在 candidates 中选一个最佳点
                int best = -1;
                size_t best_size = std::numeric_limits<size_t>::max();

                for (int u : candidates)
                {
                    if (!is_valid(u))
                        continue;
                    size_t csize = candidate_set_[u].size();

                    if (vertex_model_map.count(u))
                    {
                        if (csize < best_size || best == -1)
                        {
                            best = u;
                            best_size = csize;
                        }
                    }
                }

                if (best == -1)
                {
                    for (int u : candidates)
                    {
                        if (!is_valid(u))
                            continue;
                        size_t csize = candidate_set_[u].size();
                        if (csize < best_size || best == -1)
                        {
                            best = u;
                            best_size = csize;
                        }
                    }
                }

                if (best == -1)
                {
                    throw std::runtime_error("No valid next vertex found.");
                }

                match_order.push_back(best);
                visited.insert(best);
            }
        }

        inline void CandidateSpace::buildMatchOrder(std::vector<int> &match_order, GraphLib::SubgraphMatching::PatternGraph *query,
                                                    MatchOrderStrategy strategy, std::unordered_map<int, int> &vertex_model_map)
        {
            if (!query)
            {
                throw std::invalid_argument("queryGraph is null");
            }

            std::unordered_set<int> visited_vertex;

            switch (strategy)
            {
            case MatchOrderStrategy::STRUCTURE_FIRST:
            {
                selectStartVertex(match_order, query, vertex_model_map, visited_vertex, false);
                selectNextVertex(match_order, query, vertex_model_map, visited_vertex, false);
                return;
            }
            case MatchOrderStrategy::INFERENCE_FIRST:
            {
                selectStartVertex(match_order, query, vertex_model_map, visited_vertex, true);
                selectNextVertex(match_order, query, vertex_model_map, visited_vertex, true);
                return;
            }
            case MatchOrderStrategy::INFERENCE_ON_DEMAND:
            {
                // on demand 主要在 selectNextVertex 策略有些差别，这里先偷懒留下原代码
                buildMatchOrderInferenceOnDemand(match_order, query, vertex_model_map);
                return;
            }
            default:
                throw std::invalid_argument("Unknown MatchOrderStrategy");
            }
            return;
        }

        inline int CandidateSpace::getCandidateTreeCounts(int next_order, Tq *tq_node, int M, std::unordered_map<int, int> &fixed_assignments,
                                                          std::vector<std::vector<long long>> &num_trees, int last_cand, int last_u, int u_next)
        {
            int u = tq_node->cur; // 当前查询顶点
            // int u_next = match_order[next_order]; // 当前层要扩展的顶点
            // printf("getCandidateTreeCounts: next_order = %d, u = %d, u_next = %d, last_u = %d, last_cand = %d\n",
            // next_order, u, u_next, last_u, last_cand);

            long long total_count = 0;
            std::vector<int> candidates; // 当前顶点的候选集合

            // ========== Step1: 获取候选集合 ==========
            if (M == 0)
            {
                // === Case1: 没有超点 ===
                if (next_order == 0)
                {
                    candidates.push_back(last_cand); // 根节点只有一个候选
                }
                else
                {
                    // 父节点是单个顶点 → 从 fixed_assignments 里取父节点的候选
                    int p = last_u; //! 这里可能不对
                    int cand_p = last_cand;
                    int adj_idx = query_->GetAdjIdx(p, u_next);
                    // printf("getCandidateTreeCounts: p = %d, cand_p = %d, adj_idx = %d\n", p, cand_p, adj_idx);
                    candidates = candidate_neighbors[p][cand_p][adj_idx];
                }
            }
            else if (next_order == M && M != match_order.size())
            {
                // === Case2: 父节点在超点里 ===
                // 超点的后继节点要做多路交集
                std::vector<std::pair<std::vector<int>::const_iterator, std::vector<int>::const_iterator>> iterators;

                for (int nei_u : query_->GetNeighbors(u_next))
                {
                    auto it = fixed_assignments.find(nei_u);
                    if (it == fixed_assignments.end())
                        continue;

                    int cand_idx = it->second;
                    int u_idx = query_->GetAdjIdx(nei_u, u_next);
                    const std::vector<int> &neighs = candidate_neighbors[nei_u][cand_idx][u_idx];
                    if (!neighs.empty())
                        iterators.emplace_back(neighs.begin(), neighs.end());
                }

                if (iterators.empty())
                {
                    candidates.clear();
                }
                else if (iterators.size() == 1)
                {
                    candidates.assign(iterators[0].first, iterators[0].second);
                }
                else
                {
                    // 多路交集，小集合优先
                    std::sort(iterators.begin(), iterators.end(), [](auto &a, auto &b) { return (a.second - a.first) < (b.second - b.first); });

                    while (iterators[0].first != iterators[0].second)
                    {
                        int target = *iterators[0].first;
                        bool ok = true;

                        for (int i = 1; i < iterators.size(); i++)
                        {
                            while (iterators[i].first != iterators[i].second && *iterators[i].first < target)
                            {
                                ++iterators[i].first;
                            }
                            if (iterators[i].first == iterators[i].second)
                            {
                                ok = false;
                                break;
                            }
                            if (*iterators[i].first > target)
                            {
                                ok = false;
                                break;
                            }
                        }

                        if (ok)
                        {
                            candidates.push_back(target);
                        }

                        ++iterators[0].first; // 下一个 target
                    }
                }
            }
            else if (M == match_order.size())
            {
                // === Case3: 已经到叶子节点 ===
                total_count = 1;
            }
            else
            {
                // === Case4: 父节点是单个顶点 ===
                int p = last_u;
                int cand_p = last_cand;
                int adj_idx = query_->GetAdjIdx(p, u_next);
                candidates = candidate_neighbors[p][cand_p][adj_idx];
            }

            // ========== Step2: 遍历候选，递归 ==========
            num_trees[u].resize(candidates.size());
            if (tq_node->child_.empty())
            {
                // 叶子节点，所有候选的子树数量都是1
                for (size_t ci = 0; ci < candidates.size(); ci++)
                {
                    num_trees[u][ci] = 1;
                }
                tq_node->sub_count = candidates.size();
                return candidates.size();
            }
            for (size_t ci = 0; ci < candidates.size(); ci++)
            {
                int cand = candidates[ci];

                long long count_for_this_cand = 1;
                for (Tq *child_tq : tq_node->child_)
                {
                    count_for_this_cand *= getCandidateTreeCounts(next_order + 1, child_tq, M, fixed_assignments, num_trees, cand, u, child_tq->cur);
                }

                num_trees[u][ci] = count_for_this_cand; // 保存每个候选的子树数量
                total_count += count_for_this_cand;
            }

            tq_node->sub_count = total_count; // 保存当前节点的总计数
            return total_count;
        }

        inline bool CandidateSpace::sampleOneMatch(Tq *tq_node, std::unordered_map<int, int> &assignments,
                                                   std::unordered_map<int, int> &fixed_assignments, std::vector<int> &match_order, int M,
                                                   std::unordered_map<int, int> &output_mapping, std::vector<std::vector<long long>> &num_trees,
                                                   int last_cand, int last_u, std::unordered_set<int> visited_cands)
        {
            int u = tq_node->cur;
            visited_cands.insert(candidate_set_[last_u][last_cand]);
            // === Step1: 构造候选集合 ===
            std::vector<int> candidates;

            if (M == 0 && match_order_map[u] == 0)
            {
                // 没有超点
                // 根节点：只有一个候选
                candidates.push_back(last_cand);
            }
            else if (match_order_map[u] == M)
            {
                // 父节点是超点：求候选交集
                std::vector<int> intersection;
                bool first = true;

                for (int nei_u : query_->GetNeighbors(u))
                {
                    auto it = fixed_assignments.find(nei_u);
                    if (it == fixed_assignments.end())
                        continue;

                    int cand_idx = it->second;
                    int u_idx = query_->GetAdjIdx(nei_u, u);
                    const std::vector<int> &neighs = candidate_neighbors[nei_u][cand_idx][u_idx];

                    if (first)
                    {
                        intersection = neighs;
                        first = false;
                    }
                    else
                    {
                        std::vector<int> tmp;
                        std::set_intersection(intersection.begin(), intersection.end(), neighs.begin(), neighs.end(), std::back_inserter(tmp));
                        intersection.swap(tmp);
                    }
                }
                candidates = intersection;
            }
            else if (match_order_map[u] > M)
            {
                // 父节点是单顶点
                int p = last_u;
                int cand_p = assignments[p];
                int adj_idx = query_->GetAdjIdx(p, u);
                if (adj_idx != -1)
                {
                    candidates = candidate_neighbors[p][cand_p][adj_idx];
                }
            }

            if (candidates.empty())
                return false;

            // === Step2: 使用 num_trees 获取子树计数 ===
            std::vector<long long> sub_counts(candidates.size(), 0);
            long long total = 0;

            for (int ci = 0; ci < candidates.size(); ci++)
            {
                int cand = candidates[ci];
                sub_counts[ci] = num_trees[u][ci]; // 直接读取预计算的子树计数
                // printf("u = %d, cand = %d, sub_counts = %lld\n", u, cand, num_trees[u][ci]);
                // printf("num_trees[u].size() = %d\n", (int)num_trees[u].size());
                total += sub_counts[ci];
            }

            if (total == 0)
                return false;

            // === Step3: 加权随机选择一个候选 ===
            //static std::mt19937 rng(15);
            std::uniform_int_distribution<long long> dist(0, total - 1);
            long long r = dist(rng_);

            long long prefix = 0;
            int chosen_idx = -1;
            for (int i = 0; i < (int)candidates.size(); i++)
            {
                prefix += sub_counts[i];
                if (r < prefix)
                {
                    chosen_idx = i;
                    break;
                }
            }
            if (chosen_idx == -1)
                return false;

            // === Step4: 递归采样子树 ===
            int chosen_cand = candidates[chosen_idx];
            assignments[u] = chosen_cand;
            output_mapping[u] = chosen_cand;

            bool ok = true;
            for (Tq *child : tq_node->child_)
            {
                visited_cands.insert(candidate_set_[u][chosen_cand]);
                ok &= sampleOneMatch(child, assignments, fixed_assignments, match_order, M, output_mapping, num_trees, chosen_cand, u, visited_cands);
                if (!ok)
                    break;
            }

            return ok;
        }

        inline Tq *CandidateSpace::BuildSpanningTreeTq(PatternGraph *query, const std::vector<int> &match_order, int M)
        {
            if (M < 0 || M > (int)match_order.size())
            {
                throw std::invalid_argument("Invalid M in BuildSpanningTreeTq");
            }

            Tq *root = nullptr;
            std::unordered_map<int, Tq *> id2node;

            if (M == 0)
            {
                // === 没有超点的情况 ===
                int first_id = match_order[0];
                root = new Tq();
                root->cur = first_id;
                root->sub_count = 0;
                id2node[first_id] = root;

                // 从 1 开始构造整棵树
                for (int idx = 1; idx < (int)match_order.size(); ++idx)
                {
                    int u = match_order[idx];

                    std::vector<int> neighbors = query->GetNeighbors(u);
                    std::unordered_set<int> neigh_set(neighbors.begin(), neighbors.end());

                    Tq *parent_node = nullptr;

                    // 在 [0..idx-1] 里找第一个邻居
                    for (int j = 0; j < idx; ++j)
                    {
                        int cand = match_order[j];
                        if (id2node.count(cand) && neigh_set.find(cand) != neigh_set.end())
                        {
                            parent_node = id2node[cand];
                            break;
                        }
                    }

                    if (!parent_node)
                        parent_node = root;

                    Tq *node_u = new Tq();
                    node_u->cur = u;
                    node_u->sub_count = 0;
                    parent_node->child_.push_back(node_u);

                    id2node[u] = node_u;
                }
            }
            else
            {
                // === 原有逻辑 ===
                int supernode_id = match_order[M - 1];
                root = new Tq();
                root->cur = supernode_id;
                root->sub_count = 0;
                id2node[supernode_id] = root;

                for (int idx = M; idx < (int)match_order.size(); ++idx)
                {
                    int u = match_order[idx];

                    std::vector<int> neighbors = query->GetNeighbors(u);
                    std::unordered_set<int> neigh_set(neighbors.begin(), neighbors.end());

                    Tq *parent_node = nullptr;

                    bool found_in_confirmed = false;
                    for (int j = 0; j < M; ++j)
                    {
                        int cand = match_order[j];
                        if (neigh_set.find(cand) != neigh_set.end())
                        {
                            parent_node = root;
                            found_in_confirmed = true;
                            break;
                        }
                    }

                    if (!found_in_confirmed)
                    {
                        for (int j = M; j < idx; ++j)
                        {
                            int cand = match_order[j];
                            if (id2node.count(cand) && neigh_set.find(cand) != neigh_set.end())
                            {
                                parent_node = id2node[cand];
                                break;
                            }
                        }
                    }

                    if (!parent_node)
                        parent_node = root;

                    Tq *node_u = new Tq();
                    node_u->cur = u;
                    node_u->sub_count = 0;
                    parent_node->child_.push_back(node_u);

                    id2node[u] = node_u;
                }
            }

            return root;
        }

        inline std::pair<bool, int> CandidateSpace::checkPartialMatch(TreeNode *partial_match_node, vector<double> &threshold_vertex)
        {
            // 确定部分匹配的深度
            int depth = 0;
            TreeNode *temp = partial_match_node;
            while (temp && temp->parent)
            {
                depth++;
                temp = temp->parent;
            }

            // 从部分匹配开始，向上遍历到根节点
            TreeNode *current_node = partial_match_node;
            for (int i = depth - 1; i >= 0; i--)
            {
                int query_vertex = match_order[i];
                int data_vertex = current_node->id;

                // 如果该查询顶点有谓词，验证谓词
                if (query_->havePredicate(query_vertex) && !getOracle(query_vertex, data_vertex))
                {
                    return {false, depth}; // 谓词验证失败，返回失败结果和深度
                }

                current_node = current_node->parent;
            }

            // 所有谓词验证通过
            return {true, depth};
        }

        inline std::pair<double, double>
        CandidateSpace::recall_Estimation_treeSampling_stratified(std::vector<std::pair<TreeNode *, int>> partial_match, int sampleSize, double tp,
                                                                  bool use_adaptive_sampling)
        {
            oracle_no_more_requests.store(false);

            // ==== Step 0: 构建失败分枝列表 ====
            std::vector<std::pair<TreeNode *, int>> partial_match_failed;
            for (int u_order = 0; u_order < (int)match_order.size(); ++u_order)
            {
                int u = match_order[u_order];
                if (!query_->havePredicate(u))
                    continue;
                for (int v_idx = 0; v_idx < (int)candidate_set_[u].size(); ++v_idx)
                {
                    TreeNode *next = node_map[u][v_idx].second;
                    while (next != nullptr)
                    {
                        if (next->type == NodeType::fail_Expension)
                            partial_match_failed.emplace_back(next, u_order);
                        next = next->candidate_ptr;
                    }
                }
            }

            // ==== Step 1: 计算失败层树总量 ====
            std::vector<int> candidate_tree_counts_failed(partial_match_failed.size());
            std::vector<std::tuple<Tq *, int, std::vector<std::vector<long long>>>> candidate_tree_infos_failed;
            long long total_failed_trees = 0;

            for (size_t i = 0; i < partial_match_failed.size(); ++i)
            {
                TreeNode *failed_node = partial_match_failed[i].first;
                int failed_order = partial_match_failed[i].second;
                auto Tq_root_cur = BuildSpanningTreeTq(query_, match_order, failed_order);

                std::vector<std::vector<long long>> num_trees(query_->GetNumVertices());
                for (int u = 0; u < query_->GetNumVertices(); ++u)
                    num_trees[u].resize(GetCandidateSetSize(u));

                std::unordered_map<int, int> empty_map;
                int matched_node_num = 0;
                TreeNode *cur_node = failed_node;
                while (cur_node != TrieRoot)
                {
                    matched_node_num++;
                    cur_node = cur_node->parent;
                }
                matched_node_num--;

                candidate_tree_counts_failed[i] = getCandidateTreeCounts(matched_node_num, Tq_root_cur, matched_node_num, empty_map, num_trees,
                                                                         failed_node->cand_idx, failed_node->id, failed_node->id);
                total_failed_trees += candidate_tree_counts_failed[i];
                candidate_tree_infos_failed.emplace_back(Tq_root_cur, failed_order, std::move(num_trees));
            }

            // ==== Step 2: 边界情况硬截断 (防止崩溃的核心) ====
            long long N1 = (long long)partial_match.size();
            long long N2 = total_failed_trees;

            // 1. 没有任何数据
            if (N1 == 0 && N2 == 0)
                return {0.0, 0.0};

            // 2. 没有失败层 (Recall 必为 1.0)
            if (N2 == 0)
            {
                // 释放 Step 1 可能产生的微量内存（虽然此时 infos 为空）
                for (auto &entry : candidate_tree_infos_failed)
                    delete std::get<0>(entry);
                return {1.0, 0.0};
            }

            // 3. 没有成功层 (Recall 必为 0.0)
            if (N1 == 0)
            {
                for (auto &entry : candidate_tree_infos_failed)
                    delete std::get<0>(entry);
                return {0.0, 0.0};
            }

            // ==== Step 3: 动态采样循环 (此时 N1, N2 均 > 0) ====
            int total_sampled = 0, sampled_n1 = 0, sampled_n2 = 0;
            int TP_accum = 0, FN_accum = 0;
            double recall_estimate = 0.0;
            long double current_variance = 0.0;
            const int BATCH_SIZE = oracle_batch_size;
            //static std::mt19937 rng(60);

            while (total_sampled < sampleSize)
            {
                int batch_n1 = 0, batch_n2 = 0;

                if (total_sampled == 0)
                {
                    double ratio_n1 = (double)N1 / (N1 + N2);
                    batch_n1 = std::max(5, (int)std::round(BATCH_SIZE * ratio_n1));
                    batch_n1 = std::min(batch_n1, BATCH_SIZE - 5); // 确保给 n2 留至少 1 个样本
                    batch_n2 = BATCH_SIZE - batch_n1;
                }
                else
                {
                    double p1 = (sampled_n1 > 0) ? (double)TP_accum / sampled_n1 : 0.5;
                    double p2 = (sampled_n2 > 0) ? (double)FN_accum / sampled_n2 : 0.5;
                    auto opt = optimalAllocation(BATCH_SIZE, p1, p2, N1, N2);
                    batch_n1 = opt.first;
                    batch_n2 = opt.second;
                }

                std::vector<std::vector<std::pair<int, int>>> current_batch_tasks_success(batch_n1);
                std::vector<std::vector<std::pair<int, int>>> current_batch_tasks_failed(batch_n2);

                // 成功层采样
                for (int i = 0; i < batch_n1; ++i)
                {
                    std::uniform_int_distribution<int> dist(0, (int)N1 - 1);
                    TreeNode *cur_node = partial_match[dist(rng_)].first;
                    while (cur_node != TrieRoot)
                    {
                        if (query_->havePredicate(cur_node->id))
                        {
                            enqueueOracleTask(cur_node->id, cur_node->cand_idx, "1");
                            current_batch_tasks_success[i].emplace_back(cur_node->id, cur_node->cand_idx);
                        }
                        cur_node = cur_node->parent;
                    }
                }

                // 失败层采样
                for (int i = 0; i < batch_n2; ++i)
                {
                    int idx = weightedSample(candidate_tree_counts_failed);
                    // 此时已保证 candidate_tree_counts_failed 不为空，idx 必然有效
                    Tq *Tq_root_cur = std::get<0>(candidate_tree_infos_failed[idx]);
                    int M = std::get<1>(candidate_tree_infos_failed[idx]);
                    auto &num_trees_cache = std::get<2>(candidate_tree_infos_failed[idx]);

                    std::unordered_map<int, int> assignments, fixed_assignments, output_mapping;
                    std::unordered_set<int> visited;
                    TreeNode *cur = partial_match_failed[idx].first;
                    int temp_order = partial_match_failed[idx].second;
                    while (cur != TrieRoot)
                    {
                        if (temp_order < M)
                        {
                            fixed_assignments[cur->id] = cur->cand_idx;
                            visited.insert(candidate_set_[cur->id][cur->cand_idx]);
                        }
                        cur = cur->parent;
                        temp_order--;
                    }
                    if (sampleOneMatch(Tq_root_cur, assignments, fixed_assignments, match_order, M, output_mapping, num_trees_cache,
                                       partial_match_failed[idx].first->cand_idx, partial_match_failed[idx].first->id, visited))
                    {
                        for (auto &[u, cand] : output_mapping)
                        {
                            if (query_->havePredicate(u))
                            {
                                enqueueOracleTask(u, cand, "1");
                                current_batch_tasks_failed[i].emplace_back(u, cand);
                            }
                        }
                    }
                }

                waitOracleComplete();

                // 统计逻辑
                for (auto &tasks : current_batch_tasks_success)
                {
                    bool ok = true;
                    for (auto &p : tasks)
                    {
                        treeSampling_recall_marginal[p.first].first += (oracleRecord[p.first][p.second] ? 1 : 0);
                        if (!oracleRecord[p.first][p.second])
                        {
                            ok = false;
                            // break;
                        }
                    }
                    if (ok)
                        TP_accum++;
                    sampled_n1++;
                }
                for (auto &tasks : current_batch_tasks_failed)
                {
                    if (tasks.empty())
                        continue;
                    bool ok = true;
                    for (auto &p : tasks)
                    {
                        treeSampling_recall_marginal[p.first].second += (oracleRecord[p.first][p.second] ? 1 : 0);
                        if (!oracleRecord[p.first][p.second])
                        {
                            ok = false;
                            // break;
                        }
                    }
                    if (ok)
                        FN_accum++;
                    sampled_n2++;
                }

                total_sampled = sampled_n1 + sampled_n2;
                if (sampled_n1 > 0 && sampled_n2 > 0)
                {
                    double w1 = (double)N1 / sampled_n1, w2 = (double)N2 / sampled_n2;
                    recall_estimate = (w1 * TP_accum) / (w1 * TP_accum + w2 * FN_accum);
                    current_variance = calculateVariance_stractified(w1, w2, N1, N2, sampled_n1, sampled_n2);
                    if (total_sampled >= 128 && check_stopCondition((double)current_variance, 0.05, 0.95))
                        break;
                }
            }

            for (auto &entry : candidate_tree_infos_failed)
                delete std::get<0>(entry);

            // ==== 计算每个谓词的边缘 Recall ====
            // 逻辑对齐：Recall_u = (N1 * p1_u) / (N1 * p1_u + N2 * p2_u)
            if (sampled_n1 > 0 && sampled_n2 > 0)
            {
                double w1 = (double)N1 / sampled_n1;
                double w2 = (double)N2 / sampled_n2;

                for (auto &entry : treeSampling_recall_marginal)
                {
                    int u = entry.first;
                    long long succ_count = entry.second.first; // 建议用 long long 防止溢出
                    long long fail_count = entry.second.second;

                    double weighted_tp = w1 * succ_count;
                    double weighted_fn = w2 * fail_count;
                    double denominator = weighted_tp + weighted_fn;

                    // 防御性编程：处理分母为 0 的情况
                    if (denominator > 1e-12)
                    {
                        treeSampling_recall_marginal_all[u] = weighted_tp / denominator;
                    }
                    else
                    {
                        // 如果该谓词在采样样本中从未被观察到为 True
                        // 此时该谓词的边缘 Recall 是未定义的，通常可以设为 0.0 或保留默认
                        treeSampling_recall_marginal_all[u] = 0.0;
                    }
                }
            }

            return {recall_estimate, (double)current_variance};
        }

        inline double CandidateSpace::calculateVariance_Wanderjoin(std::vector<int> w_vec, std::vector<int> y_vec, std::vector<int> x_vec,
                                                                   double R_ht)
        {
            /*
            计算 Wanderjoin 估计的方差
            参数:
            - w_vec: 权重向量
            - y_vec: 成功样本向量
            - x_vec: 总样本向量
            - R_ht: 当前的 Wanderjoin 估计值

            返回:
            - 方差值
            */

            int n = w_vec.size();
            if (n == 0)
                return 0.0;

            long double s2 = 0.0;
            long double s1 = 0.0;
            for (int i = 0; i < n; ++i)
            {
                long double b_i = static_cast<double>(w_vec[i]);
                long double a_i = static_cast<double>(y_vec[i]);
                long double c_i = static_cast<double>(x_vec[i]);

                long double term = a_i - R_ht * c_i;
                s2 += (b_i * b_i - b_i) * (term * term);
                s1 += b_i * c_i;
            }

            if (n > 1)
                s2 /= (n - 1);
            else
                s2 = 0.0;

            long double variance = s2 / (s1 * s1);

            return variance;
        }

        inline double CandidateSpace::calculateVariance_stractified(double pi1, double pi2, int S1, int S2, int N1, int N2)
        {
            double variance = 0.0;
            double s1 = S1 * S1 * S2 * S2 * pi1 * pi2;
            double s2 = std::pow(N1 * pi1 + N2 * pi2, 4);
            double s3 = pi2 * (1 - pi1) / N1;
            double s4 = pi1 * (1 - pi2) / N2;
            variance = (s1 / s2) * (s3 + s4);

            return variance;
        }

        inline double CandidateSpace::check_stopCondition(double variance, double epsilon, double confidence)
        {
            /*
            检查停止条件是否满足
            参数:
            - variance: 当前估计的方差
            - epsilon: 允许的误差范围
            - confidence: 置信水平

            返回:
            - 是否满足停止条件
            */

            // 计算 z 值
            double z = (confidence == 0.95 ? 1.96 : confidence == 0.99 ? 2.576 : confidence == 0.90 ? 1.645 : 1.96);

            double half_width = z * std::sqrt(variance);

            return half_width <= epsilon;
        }

        inline std::pair<int, int> CandidateSpace::optimalAllocation(int total_samples, double p1, double p2, int S1, int S2)
        {
            /*
            计算最优样本分配
            参数:
            - total_samples: 总样本数
            - p1: 层1的比例
            - p2: 层2的比例
            - S1: 层1的大小
            - S2: 层2的大小

            返回:
            - 分配给层1和层2的样本数对
            */

            int min_size_everyLier = 5;
            int remaining = total_samples - 2 * min_size_everyLier;
            double sqrt_term1 = std::sqrt(p2 * (1 - p1));
            double sqrt_term2 = std::sqrt(p1 * (1 - p2));
            double sum_sqrt_terms = sqrt_term1 + sqrt_term2;

            int n1 = static_cast<int>(std::round(remaining * (sqrt_term1 / sum_sqrt_terms))) + min_size_everyLier;
            int n2 = total_samples - n1;

            return {n1, n2};
        }

        inline std::vector<double> CandidateSpace::tune_thresholds(std::vector<double> theta, double r, const std::vector<double> &r_marg,
                                                                   double gamma, double kappa)
        {
            // assert(theta.size() == r_marg.size());
            assert(kappa > 0.0);

            // Require: r < gamma
            if (r >= gamma)
                return theta;

            // i ← argmin_j r_j
            auto it = std::min_element(r_marg.begin(), r_marg.end());
            size_t i = std::distance(r_marg.begin(), it);

            int model_type = i;
            double theta_model_type = 0.0;
            for (auto &[u, model_tp_u] : query_->predicateMap)
            {
                if (!isAmazonDataset() && query_->predicateMap[u] == model_type + 3) // parler的model id是3 4，所以得加上3变成3 4
                {
                    theta_model_type = theta[u];
                    break;
                }
                else if (isAmazonDataset() && query_->predicateMap[u] == model_type) // Amazon数据集的model id是0 1 2
                {
                    theta_model_type = theta[u];
                    break;
                }
            }

            // z ← ln(θ_i / (1 - θ_i))  (logit)
            double z = std::log(theta_model_type / (1.0 - theta_model_type));

            // Δ ← κ |γ − r|
            double delta = kappa * std::abs(gamma - r);

            if (query_type_precision)
            {
                delta = 0.0 - delta;
            }

            // θ_i ← 1 / (1 + exp(-(z − Δ)))  (sigmoid)
            for (auto &[u, model_tp_u] : query_->predicateMap)
            {
                if (!isAmazonDataset() && query_->predicateMap[u] == model_type + 3)
                {
                    theta[u] = 1.0 / (1.0 + std::exp(-(z - delta)));
                    theta[u] = std::clamp(theta[u], 0.001, 0.999);
                }
                else if (isAmazonDataset() && query_->predicateMap[u] == model_type)
                {
                    theta[u] = 1.0 / (1.0 + std::exp(-(z - delta)));
                    theta[u] = std::clamp(theta[u], 0.001, 0.999);
                }
            }

            return theta;
        }

        inline double CandidateSpace::tune_threshold_single(double theta, double r_right, double gamma, double r_left)
        {
            return 0.5 * (r_left + r_right);
        }

        /**
         * @param accurate_marginal: 顶点 u 到其边缘 Recall 的映射
         * @param current_theta: 当前所有模型的阈值向量 (其 size 就是模型总数)
         * @return: 同样长度的向量，每个位置是对应 Model 的最小边缘 Recall
         */
        inline std::pair<std::vector<double>, int>
        CandidateSpace::aggregate_marginal_to_models(const std::unordered_map<int, double> &accurate_marginal,
                                                     const std::vector<double> &current_theta, int model_count)
        {
            int model_true_count = 0;
            int onlyModel = -1;
            // 初始化为 1.1 (一个大于 1.0 的值)，方便后续判断哪些模型被更新过
            std::vector<double> model_r_marg(3, 1.1);

            bool vaild_tune_exit = false;
            for (auto const &[u, r_u] : accurate_marginal)
            {
                if (query_->havePredicate(u))
                {
                    // 从你的 predicateMap 中获取该顶点对应的 model_id
                    int model_id = query_->predicateMap[u];
                    if (current_theta[u] > 0.95 or current_theta[u] < 0.05)
                    {
                        continue; // 如果该顶点的阈值已经很高了，就不考虑它对模型边缘 Recall 的贡献
                    }
                    vaild_tune_exit = true;
                    // parler的model id是3 4，所以得减去3变成0 1
                    // 注意！如果是amazon的话，这里不需要减去3，直接用model_id就行
                    if (!isAmazonDataset()){
                        model_id -= 3;
                    }
                    if (model_id >= 0)
                    {
                        // 如果是第一次遇到该模型，直接赋值；否则取最小值
                        if (model_r_marg[model_id] > 1.0)
                        {
                            model_r_marg[model_id] = r_u;
                        }
                        else
                        {
                            model_r_marg[model_id] = std::min(model_r_marg[model_id], r_u);
                        }
                    }
                }
            }

            if (!vaild_tune_exit)
            {
                return {model_r_marg, 100}; // 100是一个safenumber，我不存在一百个谓词，这是肯定的，所以返回100表示本次调参无效
            }

            // 对于本次查询中完全没出现的模型，将其 Recall 设为 1.0
            // 这样 tune_thresholds 里的 min_element 就不会选中这些“无辜”的模型
            for (int i = 0; i < model_r_marg.size(); i++)
            {
                double val = model_r_marg[i];
                if (val <= 1.0)
                {
                    model_true_count++;
                    onlyModel = i;
                }
            }

            if (model_count == 1)
            {
                return {model_r_marg, onlyModel};
            }
            return {model_r_marg, -1};
        }

        inline TreeNode *CandidateSpace::addPartialMatch2Trie_ArrayVersion(int current_size, partial_match_state state)
        {
            int last_idx = current_size - 1;
            TreeNode *new_node = nullptr;

            for (int i = last_idx; i >= 0; i--)
            {
                int cur_u = this->match_order[i];
                int cand_v_idx = partial_match_array[cur_u];

                TreeNode **subroot = &node_map[cur_u][cand_v_idx].second;

                // 逻辑 A: 跳过中间不存在的节点，除非到了第 0 层（必须从根开始）
                if ((*subroot) == nullptr && i != 0)
                {
                    continue;
                }
                else
                {
                    TreeNode *ret_node = nullptr;
                    bool res = false;

                    if ((*subroot) == nullptr && i == 0)
                    {
                        ret_node = TrieRoot;
                        res = false; // 根节点还没开始匹配第一个 cand
                    }
                    else
                    {
                        // 逻辑 B: 查找是否存在相同前缀的路径
                        res = searchMatchTrieLeaf_ArrayVersion(current_size, ret_node, i);
                        if (res == false)
                        {
                            // 如果在当前候选链表中没找到，且 i=0，说明确实得从根开始新建
                            if (i == 0)
                            {
                                ret_node = TrieRoot;
                            }
                            else
                            {
                                continue; // 向上找更短的已知前缀
                            }
                        }
                        else
                        {
                            // 找到了匹配前缀，ret_node 会被 search 设为匹配到的那个节点
                            // 注意：search 内部已经执行了 ret = ret->parent，
                            // 所以这里的 ret_node 已经是“父节点”了，逻辑正确。
                            if (cur_u == match_order[current_size - 1] && state == full_match && ret_node->type == NodeType::fail_Expension)
                            {
                                ret_node->type = NodeType::full_Match;
                                full_match_leaf.insert(cand_v_idx);
                            }
                            else{
                                ret_node->type = NodeType::TYPE_C;
                            }
                        }
                    }

                    // 逻辑 C: 更新计数器 (修正：原版逻辑是更新“已知前缀”及其所有祖先)
                    TreeNode *n = ret_node;
                    while (n != nullptr)
                    {
                        n->counter++;
                        if (state == GraphLib::SubgraphMatching::full_match)
                        {
                            n->full_match_counter++;
                        }
                        n = n->parent;
                    }

                    // 逻辑 D: 开始从断点处向下创建新节点
                    TreeNode *lit = ret_node;

                    // 重要：如果 res 是 true，说明 0~i 已经存在了，从 i+1 开始建
                    // 如果 res 是 false，说明从根开始，或者 i 这个点还没建，从 i 开始建
                    int base = (res == true) ? i + 1 : i;

                    for (int j = base; j < current_size; j++)
                    {
                        int u = match_order[j];
                        int v_idx = partial_match_array[u];

                        if (j == (int)query_->GetNumVertices() - 1 && state == full_match)
                        {
                            new_node = new TreeNode(u, NodeType::full_Match);
                            full_match_leaf.insert(v_idx);
                        }
                        else if (j == current_size - 1)
                        {
                            new_node = new TreeNode(u, NodeType::fail_Expension);
                        }
                        else
                        {
                            new_node = new TreeNode(u, NodeType::TYPE_C);
                        }

                        new_node->cand_idx = v_idx;
                        new_node->parent = lit;
                        lit->add_child(new_node);

                        // 修正：新节点创建时，其 counter 应该初始化为 1
                        new_node->counter = 1;
                        if (state == GraphLib::SubgraphMatching::full_match)
                        {
                            new_node->full_match_counter = 1;
                        }

                        lit = new_node;

                        // 维护 node_map 指针和 candidate_ptr 链表 (前插法)
                        if (node_map[u][v_idx].second == nullptr)
                        {
                            node_map[u][v_idx].second = lit;
                            lit->candidate_ptr_fronter = nullptr;
                            lit->candidate_ptr = nullptr; // 显式置空
                        }
                        else
                        {
                            lit->candidate_ptr = node_map[u][v_idx].second;
                            node_map[u][v_idx].second->candidate_ptr_fronter = lit;
                            node_map[u][v_idx].second = lit;
                        }
                    }
                    return new_node;
                }
            }
            return nullptr;
        }

        inline bool CandidateSpace::searchMatchTrieLeaf_ArrayVersion(int current_size, TreeNode *&ret, int lastNode_with_candidate)
        {
            int cur_u = match_order[lastNode_with_candidate];
            int cand_v_idx = partial_match_array[cur_u];

            // 从 node_map 获取具有相同候选值的所有 Trie 节点链表
            ret = node_map[cur_u][cand_v_idx].second;
            TreeNode *next = ret;

            while (next != nullptr)
            {
                bool flag = true;
                TreeNode *cur_node = next;

                // 向上回溯对比 0 到 lastNode_with_candidate 的完整路径
                for (int i = lastNode_with_candidate; i >= 0; i--)
                {
                    // 逻辑检查：节点不能为空，且不是根节点（根节点不存 ID），且 ID 必须匹配
                    if (cur_node == nullptr || cur_node == TrieRoot || cur_node->cand_idx != partial_match_array[match_order[i]])
                    {
                        flag = false;
                        break;
                    }
                    cur_node = cur_node->parent;
                }

                // 最终检查：回溯完 i=0 后，cur_node 必须正好指向 TrieRoot
                // 这确保了我们匹配的是完整的前缀路径，而不是某条路径的中间一截
                if (flag && cur_node == TrieRoot)
                {
                    ret = next; // 匹配成功，ret 指向深度为 lastNode_with_candidate 的那个节点
                    return true;
                }
                next = next->candidate_ptr; // 沿着相同候选值的链表继续找其他路径
            }
            return false;
        }

    } // namespace SubgraphMatching

} // namespace GraphLib
