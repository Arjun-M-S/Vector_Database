#pragma once 
#include "vecdb/brute_force.hpp"

#include <cstddef>
#include <span> 
#include <random> 
#include <cstdint> 
#include <vector> 

namespace vecdb {
    class HNSW{
        public:
            explicit HNSW(std::size_t d, std::uint32_t seed = 42, std::size_t M = 16, std::size_t efConstruction = 200);
            std::size_t insert(std::span<const float>  v);
            std::vector<Neighbor> search(std::span<const float> query , std::size_t k , std::size_t ef_search) const; 


        private: 
            std::vector<float> vectors_; 
            std::size_t n_ = 0; 
            std::size_t d_; 
            std::vector<int> levels_; 
            std::uint32_t entry_point_ = 0;
            std::vector<std::uint32_t> level_0_links_; 
            std::vector<std::uint32_t> level_0_count_; 
            std::vector<std::vector<std::vector<std::uint32_t>>> upper_links_; 
            int max_level_ = -1; 
            std::size_t M_; 
            std::size_t Mmax_;
            std::size_t Mmax0_; 
            std::size_t ef_construction_; 
            double m_L_; 
            std::mt19937 rng_; 
            std::span<const std::uint32_t> neighbors(std::uint32_t id , int level) const; 
            int random_level(); 
            std::vector<Neighbor> search_layer(std::span<const float> query, const std::vector<Neighbor>& entry_points, std::size_t ef, int layer) const;
            std::span<const float> vector_of(std::uint32_t id) const;

    };

}