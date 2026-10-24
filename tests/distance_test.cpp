#include "vecdb/distance.hpp"

#include <gtest/gtest.h>
#include <vector>


TEST(Distance , TwoDimensional){
    std::vector<float> vec1 = {1.0f , 1.0f};
    std::vector<float> vec2 = {2.0f, 0.0f}; 
    float ans = vecdb::squared_l2(vec1, vec2); 
    EXPECT_FLOAT_EQ(ans , 2.0f);
}

TEST(Distance , ThreeDimensional){
    std::vector<float> vec1 = {1.0f , 2.0f , 3.0f};
    std::vector<float> vec2 = {4.0f, 6.0f , 8.0f}; 
    float ans = vecdb::squared_l2(vec1, vec2); 
    EXPECT_FLOAT_EQ(ans , 50.0f);
}

TEST(Distance , IsSymmetrical){
    std::vector<float> vec1 = {1.0f , 1.0f};
    std::vector<float> vec2 = {2.0f, 0.0f}; 
    float ans = vecdb::squared_l2(vec1, vec2); 
    float ans2 = vecdb::squared_l2(vec2, vec1); 
    EXPECT_FLOAT_EQ(ans , ans2);
}
TEST(Distance , IdenticalVectorsGiveZero){ 
    std::vector<float> vec1 = {1.0f , 1.0f};
    std::vector<float> vec2 = {1.0f, 1.0f}; 
    float ans = vecdb::squared_l2(vec1, vec2); 
    EXPECT_FLOAT_EQ(ans , 0.0f);
}