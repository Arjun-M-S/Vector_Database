#include "vecdb/brute_force.hpp"
#include "vecdb/dataset.hpp"

#include <vector> 
#include <cstddef> 
#include <gtest/gtest.h> 
#include <stdexcept>

namespace {

    std::vector<std::size_t> get_ids(const std::vector<vecdb::Neighbor> &neighbors) {
        std::vector<std::size_t> res(neighbors.size()); 
        for (std::size_t i = 0 ; i < neighbors.size() ; i ++){ 
            res[i] = neighbors[i].id; 
        }

        return res; 
    }

    vecdb::Matrix<float> toy_set_1(){ 
        vecdb::Matrix<float> db;
        db.data = {1, 1,  1, 2,  4, 1,  5, 2,  2, 5,  3, 5};
        db.n = 6;
        db.d = 2;
        return db;
    }

}

   


TEST(BruteForce , ToySet1_K4_TieAtBoundary){ 
    vecdb::Matrix<float> db = toy_set_1();
    std::vector<float> q = {4, 2};
    auto res = vecdb::brute_force_knn(db , q, 4); 
    std::vector<std::size_t> ids = get_ids(res);
    std::vector<std::size_t> expected = {2,3,1,0};
    EXPECT_EQ(ids , expected); 
}   

TEST(BruteForce , ToySet1_K0){ 
    vecdb::Matrix<float> db = toy_set_1();
    std::vector<float> q = {4, 2};
    auto res = vecdb::brute_force_knn(db , q, 0); 
    std::vector<std::size_t> ids = get_ids(res);
    std::vector<std::size_t> expected = {};
    EXPECT_EQ(ids , expected); 
}  

TEST(BruteForce , ToySet1_K1){ 
    vecdb::Matrix<float> db = toy_set_1();
    std::vector<float> q = {4, 2};
    auto res = vecdb::brute_force_knn(db , q, 1); 
    std::vector<std::size_t> ids = get_ids(res);
    std::vector<std::size_t> expected = {2};
    EXPECT_EQ(ids , expected); 
}  

TEST(BruteForce , ToySet1_K2){ 
    vecdb::Matrix<float> db = toy_set_1();
    std::vector<float> q = {4, 2};
    auto res = vecdb::brute_force_knn(db , q, 2); 
    std::vector<std::size_t> ids = get_ids(res);
    std::vector<std::size_t> expected = {2,3};
    EXPECT_EQ(ids , expected); 
}  

TEST(BruteForce , ToySet1_K3){ 
    vecdb::Matrix<float> db = toy_set_1();
    std::vector<float> q = {4, 2};
    auto res = vecdb::brute_force_knn(db , q, 3); 
    std::vector<std::size_t> ids = get_ids(res);
    std::vector<std::size_t> expected = {2,3,1};
    EXPECT_EQ(ids , expected); 
}  

TEST(BruteForce , ToySet1_K6){ 
    vecdb::Matrix<float> db = toy_set_1();
    std::vector<float> q = {4, 2};
    auto res = vecdb::brute_force_knn(db , q, 6); 
    std::vector<std::size_t> ids = get_ids(res);
    std::vector<std::size_t> expected = {2,3,1,0,5,4};
    EXPECT_EQ(ids , expected); 
}  

TEST(BruteForce , WrongQueryDimensionThrow){ 
    vecdb::Matrix<float> db = toy_set_1();
    std::vector<float> q = {4, 2 , 0};
    EXPECT_THROW(vecdb::brute_force_knn(db , q , 3) , std::invalid_argument); 
}  

TEST(BruteForce , ToySet1_KGreaterThanN){ 
    vecdb::Matrix<float> db = toy_set_1();
    std::vector<float> q = {4, 2};
    auto res = vecdb::brute_force_knn(db , q, 10); 
    std::vector<std::size_t> ids = get_ids(res);
    std::vector<std::size_t> expected = {2,3,1,0,5,4};
    EXPECT_EQ(ids , expected); 
}  

TEST(BruteForce , ToySet2_ExactCopy){ 
    vecdb::Matrix<float> db; 
    db.d = 2; 
    db.n = 3; 
    db.data = { 1, 1,   3, 3,    2, 0 };
    std::vector<float> q = {1,1};
    auto res = vecdb::brute_force_knn(db , q , 3); 
    std::vector<std::size_t> ids = get_ids(res); 
    std::vector<std::size_t> expected = {0 , 2, 1}; 
    EXPECT_EQ(ids , expected); 
}