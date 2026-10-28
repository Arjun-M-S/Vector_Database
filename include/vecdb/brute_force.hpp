#pragma once
#include "vecdb/dataset.hpp"


#include <span> 
#include <vector> 
#include <cstddef>
#include <compare>

namespace vecdb{ 

    // struct Neighbor{
    //     float dist;
    //     std::size_t id;
    //     auto operator<=>(const Neighbor&) const = default;
    // };
    // dist -> distance from the query vector
    // id -> id of the database vector 
    // compares distances first for ordering, falls back to id to break ties
    // Member order matters: the defaulted <=> compares fields in declaration order, so dist must come first.
    struct Neighbor{
        float dist;
        std::size_t id;
        auto operator<=>(const Neighbor&) const = default;
    };


    //returns min(k , n) vectors in ascending order of distance. Ties are broken with id. 
    //throws std::invalid_argument if query.size() != database.d
    std::vector<Neighbor> brute_force_knn(const Matrix<float> &database , std::span<const float> query, std::size_t k);    
    
}