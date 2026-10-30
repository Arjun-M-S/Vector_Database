#include "vecdb/distance.hpp"
#include <cassert>
#include <cstddef>


namespace vecdb {

float squared_l2(std::span<const float> a , std::span<const float> b){
    float sum = 0; 
    assert(a.size() == b.size()); 
    std::size_t length = a.size();
    for (std::size_t i = 0 ; i < length ; i++){ 
        float diff = a[i] - b[i]; 
        sum += diff*diff;
    }

    return sum; 
}
}