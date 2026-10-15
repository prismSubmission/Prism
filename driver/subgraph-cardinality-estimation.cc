#include <iostream>
#include <set>
#include "Base/Metrics.h"
#include "Base/Timer.h"
#include "DataStructure/Graph.h"
#include "SpecialSubgraphs/SmallCycle.h"
#include "SubgraphMatching/DataGraph.h"
#include "SubgraphMatching/PatternGraph.h"
#include "SubgraphMatching/CandidateSpace.h"
#include "SubgraphMatching/CandidateFilter.h"
#include "SubgraphCounting/Option.h"
#include "SubgraphCounting/CardinalityEstimation.h"
#include "Base/LoadGraph.h"

using namespace std;
using namespace GraphLib;

std::set<std::string> scientific_type_results = {"#CandTree"};
std::set<std::string> double_type_results = {"Truth",           "Est",      "logQError", "CSBuildTime", "TreeCountTime", "TreeSampleTime",
                                             "GraphSampleTime", "QueryTime"};
std::set<std::string> longlong_type_results = {};
std::vector<std::string> print_order = {"#CSVertex", "#CSEdge",     "#CandTree",     "#TreeTrials",    "#TreeSuccess",    "Truth",    "Est",
                                        "logQError", "CSBuildTime", "TreeCountTime", "TreeSampleTime", "GraphSampleTime", "QueryTime"};
std::vector<dict> results;
std::string query_path;
Timer timer;
std::vector<PatternGraph *> pattern_graphs;
std::deque<std::string> query_names;
std::unordered_map<std::string, double> true_cnt;
double total_time = 0.0;

string read_ans(const std::string &dataset)
{
    std::string ans_file_name = query_path;
    cout << ans_file_name << endl;
    std::ifstream ans_in(ans_file_name);
    while (!ans_in.eof())
    {
        std::string name, t, c;
        ans_in >> name >> t >> c; // c代表真实结果数量
        if (name.empty() || c.empty())
            continue;
        string rootname = name;
        name = "../dataset/" + dataset + "/query_graph/" + name;
        true_cnt[name] = stod(c);
        query_names.push_back(name);
        return rootname;
    }
}

void read_filter_option(const std::string &opt, const std::string &filter, CardinalityEstimation::CardEstOption &option)
{
    if (opt.substr(2) == "STRUCTURE")
    {
        if (filter == "X")
            option.structure_filter = SubgraphMatching::NO_STRUCTURE_FILTER;
        else if (filter == "3")
            option.structure_filter = SubgraphMatching::TRIANGLE_SAFETY;
        else if (filter == "4")
            option.structure_filter = SubgraphMatching::FOURCYCLE_SAFETY;
    }
}

void read_match_order_strategy(const std::string &strategy, CardinalityEstimation::CardEstOption &option)
{

    if (strategy == "structure_first")
        option.match_order_strategy = SubgraphMatching::MatchOrderStrategy::STRUCTURE_FIRST;
    else if (strategy == "inference_first")
        option.match_order_strategy = SubgraphMatching::MatchOrderStrategy::INFERENCE_FIRST;
    else if (strategy == "inference_on_demand")
        option.match_order_strategy = SubgraphMatching::MatchOrderStrategy::INFERENCE_ON_DEMAND;
}

int32_t main(int argc, char *argv[])
{
    std::string dataset = "parler";
    CardinalityEstimation::CardEstOption opt;

    // 默认路径
    std::string product_kcore_path_ = "";
    std::string vertex_review_path = "";
    std::string reviews_kcore_path = "";
    std::string modelLocation_path = "";
    std::string vertex_product_path = "";
    std::string image_path = "";
    std::string output_path = "output.csv";

    int est_type = 0;
    double gamma = 0.8; // 默认 gamma 0.9

    // 解析命令行参数
    for (int i = 1; i < argc; ++i)
    {
        if (argv[i][0] == '-')
        {
            std::string flag = argv[i];

            if (flag == "-vr" && i + 1 < argc)
            {
                vertex_review_path = argv[i + 1];
                i++;
            }
            else if (flag == "-rk" && i + 1 < argc)
            {
                reviews_kcore_path = argv[i + 1];
                i++;
            }
            else if (flag == "-op" && i + 1 < argc)
            {
                output_path = argv[i + 1];
                i++;
            }
            else if (flag == "-ml" && i + 1 < argc)
            {
                modelLocation_path = argv[i + 1];
                i++;
            }
            else if (flag == "-vp" && i + 1 < argc)
            {
                vertex_product_path = argv[i + 1];
                i++;
            }
            else if ((flag == "-ip" || flag == "--image-path") && i + 1 < argc)
            {
                image_path = argv[i + 1];
                i++;
            }
            else if (flag == "-pM" && i + 1 < argc)
            {
                product_kcore_path_ = argv[i + 1];
                i++;
            }
            else if (flag == "-d" && i + 1 < argc)
            {
                dataset = argv[i + 1];
                i++;
            }
            else if (flag == "-q" && i + 1 < argc)
            {
                query_path = argv[i + 1];
                i++;
            }
            else if (flag == "-K" && i + 1 < argc)
            {
                opt.ub_initial = atoi(argv[i + 1]);
                i++;
            }
            else if (flag == "-gamma" && i + 1 < argc)
            {
                gamma = std::stod(argv[i + 1]); // 使用 stod 转换为 double
                i++;
            }
            else if (flag == "-et" && i + 1 < argc)
            {
                est_type = std::stoi(argv[i + 1]); // 或者使用 atoi(argv[i + 1])
                i++;
            }
            else if (flag == "-mos" && i + 1 < argc)
            {
                read_match_order_strategy(std::string(argv[i + 1]), opt);
                i++;
            }
            else if (flag == "-b" && i + 1 < argc)
            {
                opt.oracle_only = std::stoi(argv[i + 1]) != 0;
                i++;
            }
            else if (flag.substr(0, 2) == "--")
            {
                if (i + 1 < argc)
                {
                    read_filter_option(std::string(argv[i]), std::string(argv[i + 1]), opt);
                    i++;
                }
            }
        }
    }

    if (query_path.empty())
    {
        query_path = "../dataset/" + dataset + "/" + dataset + "_ans.txt";
    }

    std::string data_path = "../dataset/" + dataset + "/" + dataset + ".graph";
    string root_name = read_ans(dataset);
    DataGraph D;
    D.LoadLabeledGraph(data_path);
    D.Preprocess();

    // 打印使用的路径（调试用）
    std::cout << "Using paths:" << std::endl;
    std::cout << "  Vertex Review: " << vertex_review_path << std::endl;
    std::cout << "  Reviews KCore: " << reviews_kcore_path << std::endl;
    std::cout << "  Model Location: " << modelLocation_path << std::endl;
    std::cout << "  Vertex Product: " << vertex_product_path << std::endl;
    std::cout << "  Product KCore: " << product_kcore_path_ << std::endl;
    std::cout << "  Image Path: " << image_path << std::endl;

    opt.MAX_QUERY_VERTEX = 12;
    opt.MAX_QUERY_EDGE = 4;
    pattern_graphs.resize(query_names.size());
    for (int i = 0; i < query_names.size(); i++)
    {
        std::string query_name = query_names[i];
        pattern_graphs[i] = new PatternGraph();
        pattern_graphs[i]->LoadLabeledGraph(query_name);
        pattern_graphs[i]->ProcessPattern(D);
        pattern_graphs[i]->EnumerateLocalTriangles();
        pattern_graphs[i]->EnumerateLocalFourCycles();
        opt.MAX_QUERY_VERTEX = std::max(opt.MAX_QUERY_VERTEX, pattern_graphs[i]->GetNumVertices());
        opt.MAX_QUERY_EDGE = std::max(opt.MAX_QUERY_EDGE, pattern_graphs[i]->GetNumEdges());
    }

    if (opt.structure_filter >= SubgraphMatching::FOURCYCLE_SAFETY)
    {
        D.EnumerateLocalFourCycles();
    }
    if (opt.structure_filter >= SubgraphMatching::TRIANGLE_SAFETY)
    {
        D.EnumerateLocalTriangles();
    }
    CardinalityEstimation::FaSTestCardinalityEstimation estimator(&D, opt);

    for (int i = 0; i < pattern_graphs.size(); i++)
    {
        PatternGraph *P = pattern_graphs[i];
        std::string query_name = query_names[i];
        std::cout << "Start Processing " << query_name << std::endl;

        // 使用变量而不是硬编码路径
        estimator.getArgs(vertex_review_path, reviews_kcore_path, modelLocation_path, 
                          vertex_product_path, product_kcore_path_, image_path, est_type, gamma, output_path, root_name, dataset);

        estimator.initONNX();
        Timer timer_all;
        timer_all.Start();
        double est = estimator.EstimateEmbeddings(P, opt);
        timer_all.Stop();
        double query_time = timer_all.GetTime();
        printf("Query Time: %.2f ms\n", query_time);
    }

    // 清理资源
    for (auto *pg : pattern_graphs)
    {
        delete pg;
    }
    pattern_graphs.clear();

    return 0;
}
