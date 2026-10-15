#pragma once

#ifndef TREENODE_H
#define TREENODE_H
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_set>
#ifndef VERTEXID_H
#define VERTEXID_H
using VertexId = std::uint32_t;
#endif
enum class NodeType
{
    full_Match,
    fail_Expension,
    TYPE_C,
    deleted
};

enum checktype
{
    precision_,
    recall_
};

enum class DeleteMode
{
    Explicit, // 真正删除
    Implicit  // 逻辑删除（打标）
};

class TreeNode : public std::enable_shared_from_this<TreeNode>
{
public:
    VertexId id;
    int cand_idx; // 用于标识候选集中的索引
    int counter;
    int full_match_counter;
    NodeType type;

    TreeNode *parent;
    std::vector<TreeNode *> children;
    std::vector<VertexId> data_vertices;                 // 满足条件的所有数据图顶点ID
    std::unordered_map<VertexId, long long> tree_counts; // 每个数据顶点对应的候选树数量
    TreeNode *candidate_ptr;
    TreeNode *candidate_ptr_fronter;

public:
    explicit TreeNode(int id_, NodeType type_ = NodeType::TYPE_C, int counter_ = 1, int full_match_counter_ = 0)
    {
        type = type_;
        counter = counter_;
        full_match_counter = full_match_counter_;
        candidate_ptr_fronter = nullptr;
        candidate_ptr = nullptr;
        id = id_;
        parent = nullptr;
    }

    TreeNode *find(int search_id);
    bool remove_child(int child_id);
    void add_child(TreeNode *child);
    static bool removeRelativePath(TreeNode *headNode, DeleteMode mode, std::unordered_map<int, std::unordered_map<int, std::pair<double, TreeNode *>>> &node_map, std::unordered_set<int> &full_match_leaf);
    void markDeleted(TreeNode *node);
    void deepDelete(TreeNode *node, std::unordered_map<int, std::unordered_map<int, std::pair<double, TreeNode *>>> &node_map, std::unordered_set<int> &full_match_leaf);
    ~TreeNode()
    {
    }

    // 添加符合条件的数据图顶点
    void add_data_vertex(int v)
    {
        data_vertices.push_back(v);
        tree_counts[v] = 0; // 初始化计数为0
    }

    // 批量添加数据图顶点
    void add_data_vertices(const std::vector<int> &vertices)
    {
        for (int v : vertices)
        {
            add_data_vertex(v);
        }
    }

    // 获取数据顶点对应的候选树数量
    long long get_tree_count(int v) const
    {
        auto it = tree_counts.find(v);
        if (it != tree_counts.end())
        {
            return it->second;
        }
        return 0;
    }

    // 设置数据顶点对应的候选树数量
    void set_tree_count(int v, long long count)
    {
        tree_counts[v] = count;
    }

    void deleteChild(TreeNode *node, std::unordered_map<int, std::unordered_map<int, std::pair<double, TreeNode *>>> &node_map)
    {

        int cur_u = node->id;
        int cand_v = node->cand_idx;
        if (node->candidate_ptr_fronter != nullptr)
        {
            node->candidate_ptr_fronter->candidate_ptr = node->candidate_ptr; // 保留 candidate_ptr_fronter 指针
            if (node->candidate_ptr)
            {
                node->candidate_ptr->candidate_ptr_fronter = node->candidate_ptr_fronter;
            }
        }
        else
        {
            node_map[cur_u][cand_v].second = node->candidate_ptr; // 保留 candidate_ptr 指针
            if (node->candidate_ptr)
            {
                node->candidate_ptr->candidate_ptr_fronter = nullptr;
            }
        }

        while (!node->children.empty())
        {
            // 总是删除最后一个元素
            TreeNode *child = node->children.back();
            node->children.pop_back(); // 先移除，再删除

            // 递归删除子树
            deleteChild(child, node_map);
        }
        delete node;
    }
};

inline TreeNode *TreeNode::find(int search_id)
{
    if (id == search_id)
        return this;
    for (auto *child : children)
    {
        TreeNode *res = child->find(search_id);
        if (res != nullptr)
            return res;
    }
    return nullptr;
}

inline bool TreeNode::remove_child(int child_id)
{
    // if (children.find(child_id) == children.end()) {
    //     return false;  // 没有找到要删除的子节点
    // }
    // children.erase(child_id);  // 删除子节点
    // auto it = std::remove_if(children.begin(), children.end(),
    //     [&](TreeNode* child) {  // 改为原始指针
    //         return child->id == child_id;
    //     });
    // if (it != children.end()) {
    //     children.erase(it, children.end());
    //     return true;
    // }
    // TODO
    return true;
}

inline void TreeNode::add_child(TreeNode *child)
{
    child->parent = this; // 设置 parent 指针
    children.push_back(child);
}

// 主函数：选择删除模式
inline bool TreeNode::removeRelativePath(TreeNode *headNode, DeleteMode mode, std::unordered_map<int, std::unordered_map<int, std::pair<double, TreeNode *>>> &node_map, std::unordered_set<int> &full_match_leaf)
{
    if (!headNode)
        return false;

    if (mode == DeleteMode::Explicit)
    {
        headNode->deepDelete(headNode, node_map, full_match_leaf); // 调用者必须保证 headNode 指针不会被再访问！
    }
    else
    {
        headNode->markDeleted(headNode);
    }

    return true;
}

// 隐式删除：递归打标，但不释放内存
inline void TreeNode::markDeleted(TreeNode *node)
{
    if (!node)
        return;

    node->type = NodeType::deleted;

    for (auto child : node->children)
    {
        markDeleted(child);
    }

    if (node->candidate_ptr)
    {
        markDeleted(node->candidate_ptr);
    }
}

// 显式删除：递归释放所有子树 + candidate 链
inline void TreeNode::deepDelete(TreeNode *node, std::unordered_map<int, std::unordered_map<int, std::pair<double, TreeNode *>>> &node_map,
                                 std::unordered_set<int> &full_match_leaf)
{
    if (!node)
        return;
    if (node->id == 2 && (node->cand_idx == 5941 || node->cand_idx == 8334))
    {
        printf("debug: node_map[%d][%d].second = %p\n", node->id, node->cand_idx, node_map[node->id][node->cand_idx].second);
    }

    // 1. 删除纵向子树
    // for (auto child : node->children) {
    //    deepDelete(child, node_map);
    //}
    // node->children.clear();

    // 2. 删除横向 candidate 链
    if (node->candidate_ptr)
    {
        TreeNode *cand = node->candidate_ptr;

        // 沿 candidate_ptr 递归删除整条链（每个 cand 节点的子树也会被删除）
        cand->deepDelete(cand, node_map, full_match_leaf);
    }

    // 3. 将本节点标记为failed但是不删除，但删除子节点
    node->type = NodeType::fail_Expension;
    // printf("node info: id=%d, cand_idx=%d is deleted\n", node->id, node->cand_idx);

    // 4. 递归删除所有子节点（从后往前删除）
    while (!node->children.empty())
    {
        // 总是删除最后一个元素
        TreeNode *child = node->children.back();
        node->children.pop_back(); // 先移除，再删除

        // 递归删除子树
        deleteChild(child, node_map);
    }
}

// 生成树节点类
class SpanningTreeNode
{
public:
    VertexId query_vertex;                               // 查询图顶点ID
    std::vector<VertexId> data_vertices;                 // 满足条件的所有数据图顶点ID
    std::unordered_map<VertexId, long long> tree_counts; // 每个数据顶点对应的候选树数量
    std::vector<SpanningTreeNode *> children;            // 子节点指针
    SpanningTreeNode *parent;                            // 父节点指针

    SpanningTreeNode(VertexId u)
        : query_vertex(u), parent(nullptr) {}

    ~SpanningTreeNode()
    {
        // 递归删除所有子节点
        for (auto *child : children)
        {
            delete child;
        }
        children.clear();
    }

    // 添加子节点
    void add_child(SpanningTreeNode *child)
    {
        child->parent = this;
        children.push_back(child);
    }

    // 添加符合条件的数据图顶点
    void add_data_vertex(int v)
    {
        data_vertices.push_back(v);
        tree_counts[v] = 0; // 初始化计数为0
    }

    // 批量添加数据图顶点
    void add_data_vertices(const std::vector<int> &vertices)
    {
        for (int v : vertices)
        {
            add_data_vertex(v);
        }
    }

    // 获取数据顶点对应的候选树数量
    long long get_tree_count(int v) const
    {
        auto it = tree_counts.find(v);
        if (it != tree_counts.end())
        {
            return it->second;
        }
        return 0;
    }

    // 设置数据顶点对应的候选树数量
    void set_tree_count(int v, long long count)
    {
        tree_counts[v] = count;
    }
};

#endif // TREENODE_H
