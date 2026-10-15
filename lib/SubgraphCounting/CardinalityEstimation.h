#pragma once

#include "SubgraphCounting/Option.h"
#include "SubgraphMatching/CandidateSpace.h"
#include "SubgraphCounting/CandidateTreeSampling.h"
#include "SubgraphCounting/CandidateGraphSampling.h"
// #include "SubgraphCounting/TreeRejectionSampling.h"
/**
 * @brief Subgraph Cardinality Estimation : Given G and P, approximate the number of embeddings of P in G.
 * @date 2023-05-01
 * @ref
 */

namespace GraphLib{
using SubgraphMatching::DataGraph, SubgraphMatching::PatternGraph, SubgraphMatching::CandidateSpace;
namespace CardinalityEstimation {
    class FaSTestCardinalityEstimation {
        CandidateSpace *CS;
        DataGraph *data_;
        PatternGraph *query_;
        CardEstOption opt_;
        dict result;
        CandidateTreeSampler *TS;
        CandidateGraphSampler *GS;
    public:

        void getArgs(string vertex_review_path, string reviews_kcore_path, string modelLocation_path, string vertex_product_path,
                     string product_kcore_path_, string image_path, int est_type, double gamma, string output_path,
                     string root_name, string dataset) {
            CS->getArgs(vertex_review_path, reviews_kcore_path, modelLocation_path, vertex_product_path, product_kcore_path_,
                        image_path, est_type, gamma, output_path, root_name, dataset);
        }

        void initONNX() {
            CS->initONNX();
        }

        FaSTestCardinalityEstimation(DataGraph *data, CardEstOption opt) {
            data_ = data;
            opt_ = opt;

            CS = new CandidateSpace(data, opt);
            TS = new CandidateTreeSampler(data, opt);
            GS = new CandidateGraphSampler(data, opt);
            result.clear();
        };
        dict GetResult() {return result;}
        double EstimateEmbeddings(PatternGraph *query, CardEstOption opt) {
            result.clear();
            double query_time = 0.0;
            query_ = query;
            CS->BuildCS(query_, opt);
            for (auto &[key, value] : CS->GetCSInfo()) {
                result[key] = value;
            }
            TS->Preprocess(query, CS);

            auto ts_result = TS->Estimate();

            for (auto &[key, value] : TS->GetInfo()) {
                result[key] = value;
            }
            result["GraphSampleTime"] = 0.00;
            double est = ts_result.first;
            if (ts_result.second <= 10) {
                GS->Preprocess(query, CS);
                est = GS->Estimate(ceil((double)(opt_.ub_initial * query_->GetNumVertices()) / sqrt(ts_result.second + 1)));

                for (auto &[key, value] : GS->GetInfo()) {
                    result[key] = value;
                }
            }
            query_time = std::any_cast<double>(result["CSBuildTime"])
                    + std::any_cast<double>(result["TreeCountTime"])
                    + std::any_cast<double>(result["TreeSampleTime"])
                    + std::any_cast<double>(result["GraphSampleTime"]);
            result["QueryTime"] = query_time;
            return est;
        };
    };
}
}
