# Prism: Progressive Query Processing with Inference Predicates

This repository contains the implementation of **Prism**, a progressive framework for processing graph queries with inference predicates under precision and recall constraints.

Prism integrates subgraph matching with machine learning inference and supports progressive threshold tuning to balance efficiency and result quality.


---

## Environment

Our experiments were conducted on a server with the following configuration:

* **OS**: Ubuntu 22.04.3 LTS
* **CPU**: 2 × Intel Xeon Gold 6130 (32 physical cores, 64 threads)
* **Memory**: 503 GB RAM
* **GPU**: 4 × NVIDIA GeForce RTX 3090 (24 GB VRAM each)

**Software stack:**

* NVIDIA Driver: 550.144.03
* CUDA: 12.2
* ONNX Runtime: v1.23.2
* GLIBCXX: 3.4.30 or newer

---

## One-Stop Setup And Run

### 1. Clone The Code

```bash
git clone https://github.com/prismSubmission/Prism.git
cd Prism
```

### 2. Download Datasets And Models

Due to size limitations, datasets and models are hosted externally.

Anonymous access link:

```text
https://1drv.ms/f/c/615c075f1d5997b8/IgAaiQE-LU5bTq7-ytUg8bv7AZ0fd1XBe5hrZbZ09I-COL8?e=LrGSvr
```

Download the following resources from the shared folder:

```text
prismSubmission/
├── dataset/
│   └── all_datasets.zip
└── models/
```

Place and extract them under the repository root:

```bash
mkdir -p dataset
unzip all_datasets.zip -d dataset
```

After extraction, the dataset directory should look like:

```text
Prism/
└── dataset/
    ├── amazon_single/
    ├── amazon_multi/
    ├── parler_single/
    └── parler_multi/
```

Place the downloaded model directory directly under the repository root:

```text
Prism/
└── models/
    ├── bert-large-uncased-sst2/
    ├── clip-vit-base-patch32/
    ├── clip-vit-large-patch14/
    ├── deberta-v2-xlarge-mnli/
    ├── deberta-v3-base-mnli-fever-anli/
    ├── distilbert-base-uncased-finetuned-sst-2-english/
    ├── distilbert-base-uncasedv1-finetuned-twitter-sentiment/
    └── twitter-roberta-base-sentiment-latest/
```

The separate `query/` folder in the shared drive is not required for the packaged single-run scripts because the query graphs are already included inside `all_datasets.zip`.

### 3. Build Prism

The code was tested in our local CUDA/ONNX Runtime GPU environment. Because ONNX Runtime GPU, CUDA, OpenCV, Boost, GSL, tokenizers-cpp, and sentencepiece may be installed in different locations across systems, users may need to adjust `CMakeLists.txt` to match their local paths. We provide the expected dataset/model layout and a unified run script.

This project depends on several external libraries that are **not included in the repository**:

* ONNX Runtime (GPU)
* tokenizers-cpp
* sentencepiece
* (optional) Conda environment

CMake first searches standard system paths automatically. If some dependencies are installed in non-standard locations, specify their root directories when running CMake:

```bash
cmake -S . -B build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DTOKENIZERS_CPP_ROOT=/path/to/tokenizers-cpp \
  -DSENTENCEPIECE_ROOT=/path/to/sentencepiece \
  -DCONDA_ENV_ROOT=/path/to/conda/env
```

Then build:

```bash
cmake --build build -j
```

The generated executable should be:

```text
build/Prism
```

### 4. Run A Packaged Dataset

Use the unified single-run entry script:

```bash
bash run_prism.sh -d amazon_single
bash run_prism.sh -d amazon_multi
bash run_prism.sh -d parler_single
bash run_prism.sh -d parler_multi
```

Supported public arguments:

```bash
bash run_prism.sh -d <dataset> [-b 0|1] [-et 0|1|2] [-mos <mode>] [-op <output.csv>]
```

Here `<dataset>` must be one of:

* `amazon_single`
* `amazon_multi`
* `parler_single`
* `parler_multi`

The script hides dataset-specific metadata paths from users and internally passes the required graph, workload, model-location, text metadata, and image metadata paths to `build/Prism`.

Outputs are written to:

```text
single_run_output/<dataset>/
```

---

## Key Parameters

* `-et`:

  * `0`: Precision estimation
  * `1`: Recall estimation (Wander Join)
  * `2`: Recall estimation (Tree Sampling)

* `-b`:

  * `0`: Prism (optimized)
  * `1`: Baseline (oracle-only)

* `-mos`:

  * `inference_first`
  * `structure_first`
  * `inference_on_demand`

* `-op`:

  * Output CSV path.
  * Default: `single_run_output/<dataset>/output.csv`

* `-d`:

  * Dataset selection.
  * Supported values: `amazon_single`, `amazon_multi`, `parler_single`, `parler_multi`

---

## Project Structure

```
Prism/
├── driver/         # Entry point
├── lib/            # Core implementation
├── dataset/        # Datasets (to be downloaded)
├── models/         # ONNX models (to be downloaded)
├── run_prism.sh    # Unified single-run entry script
├── single_run_output/ # Local runtime outputs, ignored by Git
├── build/          # Compiled binaries
```

---

## Query Format

Query graphs follow the RapidMatch graph format.

### Input Format

```
t [#Vertex] [#Edge]
v [ID] [Label] [Degree]
v [ID] [Label] [Degree]
...
e [Src] [Dst] [EdgeLabel]
```

* `t`: graph header, specifying the number of vertices and edges
* `v`: vertex definition (ID, label, degree)
* `e`: edge definition (source, destination, optional label)

Edge labels are optional. If omitted, they are treated as `0`.

---

### Example

A simple path graph with labels **0–0–1–0**:

```
t 4 3
v 0 0 1
v 1 0 2
v 2 1 2
v 3 0 1
e 0 1
e 1 2
e 2 3
```

This represents a 4-node path where the third node has a different label.

---

### File Naming Convention

Example query file:

```
query_cycle_4_0.graph
```

where:

* `4` = number of vertices
* `0` = query ID

---

## Query Workload File

The `-q` argument expects a **query workload file** such as `dataset/parler/parler_ans.txt`.

In the current implementation, each line is interpreted as:

```text
<query_file_name> <time_token> <count_token>
```

Example:

```text
query_cycle_4_0.graph 1.00ms 50000
```

Notes:

* the current driver treats the first field as a **query file name** under `dataset/<dataset>/query_graph/`
* it is not currently parsed as an arbitrary relative path
* the second and third fields are retained as workload metadata / statistics tokens

---

## Datasets

This repository currently uses two datasets:

* **Amazon**: a product-and-review dataset derived from the **Home \& Kitchen** category of Amazon Reviews 2023  
  Source: https://amazon-reviews-2023.github.io/
* **Parler**: a social-network dataset  
  Source: https://zenodo.org/records/4442460

### Primitive Inference Predicates

The following table summarizes the primitive inference predicates used in our queries and their designated oracle and proxy models.

| Dataset | Vertex type | Predicate | Oracle model | Proxy model |
| --- | --- | --- | --- | --- |
| Amazon | Product | Does the product in this image appear to have a wooden/plastic/metal/fabric/glass texture? | `clip-vit-large-patch14` | `clip-vit-base-patch32` |
| Amazon | Product | Is this product kitchenware/cookware/dining-related? | `deberta-v2-xlarge-mnli` | `deberta-v3-base-mnli-fever-anli` |
| Amazon | Review | Is this review positive/negative? | `bert-large-uncased-sst2` | `distilbert-base-uncased-finetuned-sst-2-english` |
| Parler | Post | Does the author express a positive/negative opinion about Donald Trump? | `deberta-v2-xlarge-mnli` | `deberta-v3-base-mnli-fever-anli` |
| Parler | Comment | Is this comment positive/negative? | `twitter-roberta-base-sentiment-latest` | `distilbert-base-uncasedv1-finetuned-twitter-sentiment` |

Note:

* The Amazon product-category predicate and the Parler post-opinion predicate reuse the same original MNLI model pair:
  `deberta-v3-base-mnli-fever-anli` as proxy and `deberta-v2-xlarge-mnli` as oracle.
  These were used as-is without task-specific fine-tuning in this repository.

---

## Parler Dataset Layout

The Parler dataset in this repository contains both structural graph files and additional non-structural metadata used by inference predicates.

Current layout:

```text
dataset/parler/
├── parler.graph
├── parler_ans.txt
├── query_graph/
├── id_mapping.csv
├── core_nodes_config.json
└── csv_data/
    ├── comment.csv
    ├── post.csv
    └── ...
```

### Structural Inputs

The structural part consists of:

* `parler.graph`: the data graph
* `query_graph/*.graph`: query graphs in RapidMatch format
* `parler_ans.txt`: workload file indicating which query file to run

### Non-Structural Inputs For Parler

The current Parler pipeline also requires several dataset-specific metadata files and a Parler-specific model-location file. `run_prism.sh` selects these paths internally and passes them to `build/Prism`:

* `-vr`: vertex-to-comment mapping
* `-rk`: comment text metadata
* `-pM`: post text metadata
* `-ml`: Parler model-location JSON
* `-vp`: vertex-to-post mapping

Here, `core_nodes_config.json` is the Parler version of the model-location file. It is keyed by query graph name and specifies which query vertices are assigned to which inference predicate/model types. This plays the same role as the Amazon `queryWithModel/ModelLocation/...` files, but uses a JSON format.

The remaining files are used by the current codebase to support Parler-specific inference predicate processing in addition to structural subgraph matching.

### Running Parler

Examples:

```bash
bash run_prism.sh -d parler_single
bash run_prism.sh -d parler_multi
```

The script internally uses:

* dataset directories: `dataset/parler_single/` or `dataset/parler_multi/`
* workload files: `parler_single_ans.txt` or `parler_multi_ans.txt`
* match order strategy: `inference_first`
* estimation type: `-et 0`
* optimized mode: `-b 0`

Output is written under `single_run_output/<dataset>/` unless `-op` is specified.

---

## Amazon Dataset Layout

In the current repository, Amazon queries are packaged into single-predicate and multi-predicate task folders: `dataset/amazon_single/` and `dataset/amazon_multi/`.

Current metadata files used by the Amazon packaged datasets:

```text
dataset/amazon_single/
├── amazon_single.graph
├── amazon_single_ans.txt
├── vertex_review.csv
├── reviews_kcore.csv
├── meta_kcore.csv
├── vertex_product.csv
├── pic/images_kcore/
└── queryWithModel/ModelLocation/...

dataset/amazon_multi/
├── amazon_multi.graph
├── amazon_multi_ans.txt
├── vertex_review.csv
├── reviews_kcore.csv
├── meta_kcore.csv
├── vertex_product.csv
├── pic/images_kcore/
└── queryWithModel/ModelLocation/...
```

Examples:

```bash
bash run_prism.sh -d amazon_single
bash run_prism.sh -d amazon_multi
```

This script uses:

* dataset names in the current codebase: `amazon_single` or `amazon_multi`
* workload files: `amazon_single_ans.txt` or `amazon_multi_ans.txt`
* model-location files: selected internally from `queryWithModel/ModelLocation/...`

Output is written under `single_run_output/<dataset>/` unless `-op` is specified.

---

## Legacy CLI Parameters

Some command-line option names are legacy abbreviations inherited from the earlier Amazon pipeline. In the current repository, these names are reused for dataset-specific metadata inputs.

### `-vr`

Code variable: `vertex_review_path`

Current meaning:

* vertex-to-record mapping file for text-bearing review/comment vertices
* **Parler**: maps internal comment vertex IDs to original comment IDs, which are then used to retrieve comment text from `comment.csv`
* **Amazon**: maps review vertex IDs to row IDs in `reviews_kcore.csv`, which are then used to retrieve review text
* this argument is used together with `-rk`: `-vr` provides the mapping and `-rk` provides the underlying text-bearing records

### `-rk`

Code variable: `reviews_kcore_path`

Current meaning:

* **Parler**: non-structural input for the predicate `Comment / Is this comment positive/negative?`
* **Amazon**: non-structural input for the predicate `Review / Is this review positive/negative?`

### `-vp`

Code variable: `vertex_product_path`

Current meaning:

* vertex-to-record mapping file for text-bearing product/post vertices
* **Parler**: maps post vertex IDs to original post IDs, which are then used to retrieve post text from `post.csv`
* **Amazon**: maps product vertex IDs to product identifiers such as `asin` / `parentAsin`, which are then used to retrieve product text from `meta_kcore.csv`
* this argument is used together with `-pM`: `-vp` provides the mapping and `-pM` provides the underlying text-bearing records

### `-ml`

Code variable: `modelLocation_path`

Current meaning:

* query-level model / predicate assignment file
* it specifies which query vertices are associated with which inference predicate types
* unlike `-vr/-rk` and `-vp/-pM`, this file does not provide raw text content; instead, it assigns predicate/model types to query vertices

Dataset-specific formats:

* **Parler**: a JSON file keyed by query file name, e.g. `core_nodes_config.json`
* **Amazon**: a per-query CSV/log-style file specifying model assignments for the current query

Current mapping from code-level model identifiers to paper-level predicate semantics:

* **Amazon**
  * `A` / internal model id `0`: `Review / Is this review positive/negative?`
    Proxy: `distilbert-base-uncased-finetuned-sst-2-english`
    Oracle: `bert-large-uncased-sst2`
  * `B` / internal model id `1`: `Product / Is this product kitchenware/cookware/dining-related?`
    Proxy: `deberta-v3-base-mnli-fever-anli`
    Oracle: `deberta-v2-xlarge-mnli`
  * `C` / internal model id `2`: `Product / Does the product in this image appear to have a wooden/plastic/metal/fabric/glass texture?`
    Proxy: `clip-vit-base-patch32`
    Oracle: `clip-vit-large-patch14`
* **Parler**
  * JSON key `2` -> internal model id `4`: `Post / Does the author express a positive/negative opinion about Donald Trump?`
    Proxy: `deberta-v3-base-mnli-fever-anli`
    Oracle: `deberta-v2-xlarge-mnli`
  * JSON keys `3/4/5` -> internal model id `3`: `Comment / Is this comment positive/negative?`
    Proxy: `distilbert-base-uncasedv1-finetuned-twitter-sentiment`
    Oracle: `twitter-roberta-base-sentiment-latest`

### `-pM`

Code variable: `product_kcore_path_`

Current meaning:

* product/post metadata file for text-bearing product/post vertices
* **Parler**: raw post metadata file; post text is retrieved from `post.csv`
* **Amazon**: raw product metadata file; product text is constructed from fields in `meta_kcore.csv`

### `-ip` / `--image-path`

Code variable: `image_path_`

Current meaning:

* image directory for Amazon product-image predicates
* **Amazon**: directory containing product images such as `0_<parent_asin>_img1.jpg` and `0_<parent_asin>_img2.jpg`
* **Parler**: currently unused

Example:

```bash
-ip "$DATASET_DIR/pic/images_kcore"
```

If this argument is omitted, the current code falls back to a repository-relative default:

```text
dataset/<dataset>/pic/images_kcore
```

For the packaged Amazon datasets, this should normally resolve to:

```text
dataset/amazon_single/pic/images_kcore
dataset/amazon_multi/pic/images_kcore
```

More parameter descriptions will be added as the remaining dataset-specific mappings are finalized.

---

## Notes

* Queries with extremely sparse matches may be excluded from evaluation, as discussed in the paper.
* Runtime may vary depending on GPU availability and system load.
* ONNX Runtime GPU execution is required for best performance.
* The repository focuses on one primary dataset configuration; adapting to other datasets may require minor modifications.

---

## Follow-Up Tasks

The first uploaded version prioritizes making the packaged code, datasets, and models runnable for reviewers.

Possible second-version cleanup tasks:

* Clean up redundant variables, functions, and unrelated remnants inherited from the original codebase.
* Further unify the Amazon and Parler dataset formats.
* Keep the first version focused on successful execution before making reliability and maintainability improvements.

---

## Contact

Anonymous for double-blind review.
