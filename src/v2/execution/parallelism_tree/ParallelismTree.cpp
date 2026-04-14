/**
 * @file ParallelismTree.cpp
 * @brief Implementation of the recursive parallelism tree
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "ParallelismTree.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

namespace llaminar2
{

    // =========================================================================
    // ParallelismNode — Queries
    // =========================================================================

    std::set<int> ParallelismNode::leafRanks() const
    {
        std::set<int> ranks;
        if (isLeaf())
        {
            if (owning_rank >= 0)
            {
                ranks.insert(owning_rank);
            }
        }
        else
        {
            for (const auto &child : children)
            {
                auto child_ranks = child.leafRanks();
                ranks.insert(child_ranks.begin(), child_ranks.end());
            }
        }
        return ranks;
    }

    std::vector<const ParallelismNode *> ParallelismNode::leafDevices() const
    {
        std::vector<const ParallelismNode *> leaves;
        if (isLeaf())
        {
            leaves.push_back(this);
        }
        else
        {
            for (const auto &child : children)
            {
                auto child_leaves = child.leafDevices();
                leaves.insert(leaves.end(), child_leaves.begin(), child_leaves.end());
            }
        }
        return leaves;
    }

    bool ParallelismNode::isCrossRank() const
    {
        return leafRanks().size() > 1;
    }

    int ParallelismNode::leafCount() const
    {
        if (isLeaf())
        {
            return 1;
        }
        int count = 0;
        for (const auto &child : children)
        {
            count += child.leafCount();
        }
        return count;
    }

    std::set<DeviceType> ParallelismNode::leafDeviceTypes() const
    {
        std::set<DeviceType> types;
        if (isLeaf())
        {
            types.insert(device.device_type);
        }
        else
        {
            for (const auto &child : children)
            {
                auto child_types = child.leafDeviceTypes();
                types.insert(child_types.begin(), child_types.end());
            }
        }
        return types;
    }

    bool ParallelismNode::isMixedVendor() const
    {
        auto types = leafDeviceTypes();
        // Count how many distinct GPU types we have
        int gpu_vendor_count = 0;
        if (types.count(DeviceType::CUDA))
            gpu_vendor_count++;
        if (types.count(DeviceType::ROCm))
            gpu_vendor_count++;
        return gpu_vendor_count > 1;
    }

    bool ParallelismNode::operator==(const ParallelismNode &other) const
    {
        if (type != other.type || name != other.name)
            return false;

        if (isLeaf())
        {
            return device == other.device && owning_rank == other.owning_rank &&
                   first_layer == other.first_layer && last_layer == other.last_layer;
        }

        if (children.size() != other.children.size())
            return false;

        if (type == ParallelismNodeType::TENSOR_PARALLEL)
        {
            if (backend != other.backend)
                return false;
        }

        for (size_t i = 0; i < children.size(); ++i)
        {
            if (children[i] != other.children[i])
                return false;
        }

        return first_layer == other.first_layer && last_layer == other.last_layer;
    }

    // =========================================================================
    // ParallelismTree — Layer Assignment
    // =========================================================================

    namespace
    {

        void assignLayersRecursive(ParallelismNode &node, int first, int last, int total_layers)
        {
            node.first_layer = first;
            node.last_layer = last;
            node.has_embedding = (first == 0);
            node.has_lm_head = (last == total_layers - 1);

            if (node.isLeaf())
            {
                return;
            }

            if (node.type == ParallelismNodeType::TENSOR_PARALLEL)
            {
                // TP: all children get the same layer range
                for (auto &child : node.children)
                {
                    assignLayersRecursive(child, first, last, total_layers);
                }
            }
            else if (node.type == ParallelismNodeType::PIPELINE_PARALLEL)
            {
                // PP: split layers across children
                int num_children = static_cast<int>(node.children.size());
                int num_layers = last - first + 1;

                if (!node.tp_weights.empty() &&
                    static_cast<int>(node.tp_weights.size()) == num_children)
                {
                    // Proportional split based on weights
                    float total_weight = 0.0f;
                    for (float w : node.tp_weights)
                        total_weight += w;

                    int assigned = 0;
                    for (int i = 0; i < num_children; ++i)
                    {
                        int child_layers;
                        if (i == num_children - 1)
                        {
                            // Last child gets remainder
                            child_layers = num_layers - assigned;
                        }
                        else
                        {
                            float frac = node.tp_weights[i] / total_weight;
                            child_layers = std::max(1, static_cast<int>(frac * num_layers + 0.5f));
                            // Don't exceed remaining
                            child_layers = std::min(child_layers, num_layers - assigned - (num_children - 1 - i));
                        }

                        int child_first = first + assigned;
                        int child_last = child_first + child_layers - 1;
                        assignLayersRecursive(node.children[i], child_first, child_last, total_layers);
                        assigned += child_layers;
                    }
                }
                else
                {
                    // Equal split
                    int base_layers = num_layers / num_children;
                    int extra = num_layers % num_children;
                    int assigned = 0;

                    for (int i = 0; i < num_children; ++i)
                    {
                        int child_layers = base_layers + (i < extra ? 1 : 0);
                        int child_first = first + assigned;
                        int child_last = child_first + child_layers - 1;
                        assignLayersRecursive(node.children[i], child_first, child_last, total_layers);
                        assigned += child_layers;
                    }
                }
            }
        }

    } // anonymous namespace

    void ParallelismTree::assignLayers(int total_layers_param)
    {
        total_layers = total_layers_param;
        if (total_layers > 0)
        {
            assignLayersRecursive(root, 0, total_layers - 1, total_layers);
        }
    }

    // =========================================================================
    // ParallelismTree — Validation
    // =========================================================================

    namespace
    {

        void validateNode(const ParallelismNode &node, int world_size,
                          int total_layers, const std::string &path,
                          std::vector<std::string> &errors)
        {
            if (node.name.empty())
            {
                errors.push_back(path + ": node name is empty");
            }

            if (node.isLeaf())
            {
                // DEVICE leaf validation
                if (node.owning_rank < 0 || node.owning_rank >= world_size)
                {
                    errors.push_back(path + ": owning_rank " +
                                     std::to_string(node.owning_rank) +
                                     " out of range [0, " +
                                     std::to_string(world_size) + ")");
                }

                if (node.first_layer < 0 || node.last_layer < 0)
                {
                    errors.push_back(path + ": layer range not assigned");
                }
                else if (node.first_layer > node.last_layer)
                {
                    errors.push_back(path + ": invalid layer range [" +
                                     std::to_string(node.first_layer) + ", " +
                                     std::to_string(node.last_layer) + "]");
                }
                else if (node.last_layer >= total_layers)
                {
                    errors.push_back(path + ": last_layer " +
                                     std::to_string(node.last_layer) +
                                     " exceeds total_layers " +
                                     std::to_string(total_layers));
                }
                return;
            }

            // Interior node validation
            if (node.children.empty())
            {
                errors.push_back(path + ": " +
                                 parallelismNodeTypeName(node.type) +
                                 " node has no children");
                return;
            }

            if (node.type == ParallelismNodeType::TENSOR_PARALLEL)
            {
                // TP must have ≥ 2 children
                if (node.children.size() < 2)
                {
                    errors.push_back(path + ": TP node must have ≥ 2 children, has " +
                                     std::to_string(node.children.size()));
                }

                // All TP children must have the same layer range
                int ref_first = node.children[0].first_layer;
                int ref_last = node.children[0].last_layer;
                for (size_t i = 1; i < node.children.size(); ++i)
                {
                    if (node.children[i].first_layer != ref_first ||
                        node.children[i].last_layer != ref_last)
                    {
                        errors.push_back(
                            path + ": TP child " + std::to_string(i) +
                            " has layer range [" +
                            std::to_string(node.children[i].first_layer) + ", " +
                            std::to_string(node.children[i].last_layer) +
                            "] but child 0 has [" +
                            std::to_string(ref_first) + ", " +
                            std::to_string(ref_last) + "]");
                    }
                }

                // Validate backend-device compatibility for TP nodes
                if (node.backend != CollectiveBackendType::AUTO)
                {
                    auto device_types = node.leafDeviceTypes();
                    bool has_cuda = device_types.count(DeviceType::CUDA) > 0;
                    bool has_rocm = device_types.count(DeviceType::ROCm) > 0;
                    bool is_mixed = has_cuda && has_rocm;

                    if (node.backend == CollectiveBackendType::NCCL && has_rocm)
                    {
                        errors.push_back(
                            path + ": NCCL backend specified but TP domain contains ROCm devices; "
                                   "use HETEROGENEOUS or HOST for mixed CUDA+ROCm TP");
                    }
                    if (node.backend == CollectiveBackendType::RCCL && has_cuda)
                    {
                        errors.push_back(
                            path + ": RCCL backend specified but TP domain contains CUDA devices; "
                                   "use HETEROGENEOUS or HOST for mixed CUDA+ROCm TP");
                    }
                    if (node.backend == CollectiveBackendType::HETEROGENEOUS && !is_mixed)
                    {
                        // HETEROGENEOUS without mixed vendors
                        // is valid but suboptimal — use NCCL or RCCL for same-vendor
                        // We don't error here, but could log a recommendation.
                    }
                }
            }
            else if (node.type == ParallelismNodeType::PIPELINE_PARALLEL)
            {
                // PP children must cover parent layers contiguously
                if (node.first_layer >= 0 && node.last_layer >= 0)
                {
                    int expected_first = node.first_layer;
                    for (size_t i = 0; i < node.children.size(); ++i)
                    {
                        const auto &child = node.children[i];
                        if (child.first_layer != expected_first)
                        {
                            errors.push_back(
                                path + ": PP child " + std::to_string(i) +
                                " starts at layer " +
                                std::to_string(child.first_layer) +
                                " but expected " +
                                std::to_string(expected_first));
                        }
                        expected_first = child.last_layer + 1;
                    }

                    // Last child must end at parent's last_layer
                    if (!node.children.empty())
                    {
                        int actual_last = node.children.back().last_layer;
                        if (actual_last != node.last_layer)
                        {
                            errors.push_back(
                                path + ": PP children end at layer " +
                                std::to_string(actual_last) +
                                " but parent ends at " +
                                std::to_string(node.last_layer));
                        }
                    }
                }
            }

            // Recurse into children
            for (size_t i = 0; i < node.children.size(); ++i)
            {
                std::string child_path = path + "/" + node.children[i].name;
                if (child_path.empty())
                {
                    child_path = path + "/child_" + std::to_string(i);
                }
                validateNode(node.children[i], world_size, total_layers,
                             child_path, errors);
            }
        }

    } // anonymous namespace

    std::vector<std::string> ParallelismTree::validate() const
    {
        std::vector<std::string> errors;

        if (total_layers <= 0)
        {
            errors.push_back("total_layers must be > 0, got " +
                             std::to_string(total_layers));
        }

        if (world_size <= 0)
        {
            errors.push_back("world_size must be > 0, got " +
                             std::to_string(world_size));
        }

        // Check that at least one DEVICE leaf exists
        if (root.leafCount() == 0)
        {
            errors.push_back("tree has no DEVICE leaves");
        }

        // Validate recursively
        validateNode(root, world_size, total_layers, root.name, errors);

        return errors;
    }

    // =========================================================================
    // ParallelismTree — Display
    // =========================================================================

    namespace
    {

        void toStringRecursive(const ParallelismNode &node,
                               std::ostringstream &oss,
                               int depth)
        {
            // Indent
            for (int i = 0; i < depth; ++i)
            {
                oss << (i < depth - 1 ? "│   " : "├── ");
            }

            // Node type and name
            oss << parallelismNodeTypeName(node.type)
                << "(" << node.name << ")";

            // Layer range
            if (node.first_layer >= 0 && node.last_layer >= 0)
            {
                oss << " [layers " << node.first_layer << "-" << node.last_layer << "]";
            }

            // Extra info for leaves
            if (node.isLeaf())
            {
                oss << " rank=" << node.owning_rank
                    << " " << node.device.toString();
            }
            else if (node.type == ParallelismNodeType::TENSOR_PARALLEL)
            {
                oss << " backend=" << collectiveBackendTypeToString(node.backend);
            }

            if (node.has_embedding)
                oss << " [EMB]";
            if (node.has_lm_head)
                oss << " [LM]";

            oss << "\n";

            // Recurse
            for (const auto &child : node.children)
            {
                toStringRecursive(child, oss, depth + 1);
            }
        }

    } // anonymous namespace

    std::string ParallelismTree::toString() const
    {
        std::ostringstream oss;
        oss << "ParallelismTree (layers=" << total_layers
            << ", world_size=" << world_size
            << ", leaves=" << root.leafCount() << ")\n";
        toStringRecursive(root, oss, 0);
        return oss.str();
    }

    // =========================================================================
    // Fluent Builder Functions
    // =========================================================================

    ParallelismNode PP(std::string name, std::vector<ParallelismNode> children)
    {
        ParallelismNode node;
        node.type = ParallelismNodeType::PIPELINE_PARALLEL;
        node.name = std::move(name);
        node.children = std::move(children);
        return node;
    }

    ParallelismNode TP(std::string name,
                       std::vector<GlobalDeviceAddress> devices,
                       int owning_rank,
                       CollectiveBackendType backend)
    {
        ParallelismNode node;
        node.type = ParallelismNodeType::TENSOR_PARALLEL;
        node.name = std::move(name);
        node.backend = backend;

        // Create DEVICE leaf children for each device
        for (size_t i = 0; i < devices.size(); ++i)
        {
            ParallelismNode leaf;
            leaf.type = ParallelismNodeType::DEVICE;
            leaf.name = "dev" + std::to_string(i);
            leaf.device = std::move(devices[i]);
            leaf.owning_rank = owning_rank;
            node.children.push_back(std::move(leaf));
        }

        return node;
    }

    ParallelismNode TP(std::string name,
                       std::vector<ParallelismNode> children,
                       CollectiveBackendType backend)
    {
        ParallelismNode node;
        node.type = ParallelismNodeType::TENSOR_PARALLEL;
        node.name = std::move(name);
        node.backend = backend;
        node.children = std::move(children);
        return node;
    }

    ParallelismNode Device(GlobalDeviceAddress device, int owning_rank)
    {
        ParallelismNode node;
        node.type = ParallelismNodeType::DEVICE;
        node.name = device.toString();
        node.device = std::move(device);
        node.owning_rank = owning_rank;
        return node;
    }

} // namespace llaminar2
