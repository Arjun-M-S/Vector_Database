#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>


namespace vecdb{
    float squared_l2(std::span<const float> a , std::span<const float> b); 
}