/**
 * @file PCIeEnumeration.cpp
 * @brief PCIe link speed/width detection via Linux sysfs
 *
 * Reads current and max link speed/width from:
 *   /sys/bus/pci/devices/<BDF>/current_link_speed
 *   /sys/bus/pci/devices/<BDF>/current_link_width
 *   /sys/bus/pci/devices/<BDF>/max_link_speed
 *   /sys/bus/pci/devices/<BDF>/max_link_width
 *
 * For devices behind PCIe switches, walks the sysfs topology upstream
 * to find the most restrictive (bottleneck) link in the path.
 */

#include "GPUEnumeration.h"
#include "../utils/Logger.h"
#include <cstdio>
#include <cstring>
#include <climits>
#include <string>
#include <vector>

namespace llaminar2
{
    namespace pcie_enumeration
    {
        namespace
        {
            // Read a single-line sysfs file into a string, trimming whitespace
            std::string read_sysfs_string(const char *path)
            {
                FILE *f = fopen(path, "r");
                if (!f)
                    return {};
                char buf[128] = {};
                if (!fgets(buf, sizeof(buf), f))
                {
                    fclose(f);
                    return {};
                }
                fclose(f);
                // Trim trailing newline/whitespace
                size_t len = strlen(buf);
                while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' '))
                    buf[--len] = '\0';
                return std::string(buf);
            }

            // Parse "X.X GT/s" or "X GT/s PCIe" → GT/s as double
            double parse_link_speed(const std::string &s)
            {
                if (s.empty())
                    return 0;
                double speed = 0;
                if (sscanf(s.c_str(), "%lf", &speed) == 1)
                    return speed;
                return 0;
            }

            // Parse width string (just an integer like "16")
            int parse_link_width(const std::string &s)
            {
                if (s.empty())
                    return 0;
                int width = 0;
                if (sscanf(s.c_str(), "%d", &width) == 1)
                    return width;
                return 0;
            }

            // Map GT/s → PCIe generation
            int speed_to_pcie_gen(double gts)
            {
                if (gts >= 64.0)
                    return 6;
                if (gts >= 32.0)
                    return 5;
                if (gts >= 16.0)
                    return 4;
                if (gts >= 8.0)
                    return 3;
                if (gts >= 5.0)
                    return 2;
                if (gts >= 2.5)
                    return 1;
                return 0;
            }

            // Compute effective bandwidth for a link segment (GB/s)
            double effective_bandwidth(double speed_gts, int width)
            {
                if (speed_gts <= 0 || width <= 0)
                    return 0;
                int gen = speed_to_pcie_gen(speed_gts);
                double efficiency = (gen >= 3) ? (128.0 / 130.0) : 0.8;
                return speed_gts * width * efficiency / 8.0;
            }

            // Resolve the real sysfs path for a PCI BDF address
            // Returns empty string on failure
            std::string resolve_sysfs_path(const char *bdf)
            {
                char link_path[256];
                snprintf(link_path, sizeof(link_path), "/sys/bus/pci/devices/%s", bdf);
                char resolved[PATH_MAX];
                if (!realpath(link_path, resolved))
                    return {};
                return std::string(resolved);
            }

            // Extract the BDF from a sysfs directory path component
            // e.g., "/sys/devices/pci0000:85/0000:85:00.0/0000:86:00.0" → "0000:86:00.0"
            std::string extract_bdf_from_path(const std::string &dir_path)
            {
                auto pos = dir_path.rfind('/');
                if (pos == std::string::npos)
                    return {};
                return dir_path.substr(pos + 1);
            }

            // Check if a sysfs directory is a PCI device (has a class file)
            bool is_pci_device(const std::string &dir_path)
            {
                std::string class_path = dir_path + "/class";
                FILE *f = fopen(class_path.c_str(), "r");
                if (!f)
                    return false;
                fclose(f);
                return true;
            }

            // Check if a PCI device is a PCI bridge (class 0x0604xx)
            bool is_pci_bridge(const std::string &dir_path)
            {
                std::string class_path = dir_path + "/class";
                std::string cls = read_sysfs_string(class_path.c_str());
                // PCI bridge class code: 0x0604xx
                return cls.find("0x0604") == 0;
            }

            struct LinkSegment
            {
                std::string bdf;
                double speed_gts;
                int width;
                double bandwidth_gbps;
            };

            // Walk the sysfs topology upstream from the GPU endpoint,
            // collecting link speed/width at each bridge hop.
            // Returns the segments ordered from endpoint to root.
            std::vector<LinkSegment> walk_upstream_links(const std::string &device_sysfs_path)
            {
                std::vector<LinkSegment> segments;
                std::string current = device_sysfs_path;

                // Walk up directory hierarchy
                while (true)
                {
                    auto pos = current.rfind('/');
                    if (pos == std::string::npos || pos == 0)
                        break;
                    std::string parent = current.substr(0, pos);

                    // Stop at the PCI root (e.g., "pci0000:85")
                    std::string dirname = parent.substr(parent.rfind('/') + 1);
                    if (dirname.compare(0, 3, "pci") == 0)
                        break;

                    // Only check PCI devices (bridges)
                    if (!is_pci_device(parent))
                    {
                        current = parent;
                        continue;
                    }

                    // Read link info for this bridge
                    std::string speed_path = parent + "/current_link_speed";
                    std::string width_path = parent + "/current_link_width";
                    double speed = parse_link_speed(read_sysfs_string(speed_path.c_str()));
                    int width = parse_link_width(read_sysfs_string(width_path.c_str()));

                    if (speed > 0 && width > 0)
                    {
                        LinkSegment seg;
                        seg.bdf = extract_bdf_from_path(parent);
                        seg.speed_gts = speed;
                        seg.width = width;
                        seg.bandwidth_gbps = effective_bandwidth(speed, width);
                        segments.push_back(seg);
                    }

                    current = parent;
                }

                return segments;
            }
        } // anonymous namespace

        PCIeLinkInfo read_pcie_link_info(int pci_domain, int pci_bus, int pci_device)
        {
            PCIeLinkInfo info;

            char bdf[64];
            snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.0", pci_domain, pci_bus, pci_device);
            info.pci_address = bdf;

            char path[256];
            const char *base = "/sys/bus/pci/devices";

            // Current link speed (endpoint-reported)
            snprintf(path, sizeof(path), "%s/%s/current_link_speed", base, bdf);
            info.link_speed_gts = parse_link_speed(read_sysfs_string(path));

            // Current link width (endpoint-reported)
            snprintf(path, sizeof(path), "%s/%s/current_link_width", base, bdf);
            info.link_width = parse_link_width(read_sysfs_string(path));

            // Max link speed (endpoint capability)
            snprintf(path, sizeof(path), "%s/%s/max_link_speed", base, bdf);
            info.max_speed_gts = parse_link_speed(read_sysfs_string(path));

            // Max link width (endpoint capability)
            snprintf(path, sizeof(path), "%s/%s/max_link_width", base, bdf);
            info.max_width = parse_link_width(read_sysfs_string(path));

            // Walk upstream through PCI bridges to find the true bottleneck.
            // Devices behind PCIe switches (or multi-level bridge chains like AMD Mi50)
            // may report full x16 at the endpoint while an upstream bridge link is degraded.
            std::string sysfs_path = resolve_sysfs_path(bdf);
            if (!sysfs_path.empty())
            {
                auto segments = walk_upstream_links(sysfs_path);
                double endpoint_bw = effective_bandwidth(info.link_speed_gts, info.link_width);

                for (const auto &seg : segments)
                {
                    if (seg.bandwidth_gbps < endpoint_bw && seg.bandwidth_gbps > 0)
                    {
                        // An upstream bridge is more restrictive than the endpoint
                        LOG_DEBUG("PCIe bottleneck for " << bdf << " at upstream bridge "
                                                         << seg.bdf << ": Gen"
                                                         << speed_to_pcie_gen(seg.speed_gts)
                                                         << " x" << seg.width
                                                         << " (" << seg.bandwidth_gbps << " GB/s)"
                                                         << " vs endpoint Gen"
                                                         << speed_to_pcie_gen(info.link_speed_gts)
                                                         << " x" << info.link_width
                                                         << " (" << endpoint_bw << " GB/s)");

                        // Use the upstream bottleneck as the effective link
                        if (seg.bandwidth_gbps < effective_bandwidth(info.link_speed_gts, info.link_width))
                        {
                            info.link_speed_gts = seg.speed_gts;
                            info.link_width = seg.width;
                            info.bottleneck_bdf = seg.bdf;
                            endpoint_bw = seg.bandwidth_gbps;
                        }
                    }
                }
            }

            // Derive PCIe generation from effective (possibly bottleneck) speed
            info.pcie_gen = speed_to_pcie_gen(info.link_speed_gts);

            // Check for degraded link — either endpoint or upstream bottleneck
            info.degraded = (info.link_speed_gts > 0 && info.max_speed_gts > 0 &&
                             (info.link_speed_gts < info.max_speed_gts || info.link_width < info.max_width));

            return info;
        }

    } // namespace pcie_enumeration
} // namespace llaminar2
