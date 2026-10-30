#include "vecdb/brute_force.hpp"
#include "vecdb/dataset.hpp"
#include "vecdb/distance.hpp"


#include <vector> 
#include <cstddef> 
#include <gtest/gtest.h> 
#include <stdexcept>
#include <cstdint>
#include <filesystem> 

namespace
{

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

TEST(BruteForce , SiftSmall_k10){ 
    // check if siftsmall is downloaded

    std::filesystem::path dir = std::filesystem::path(VECDB_DATA_DIR)/"siftsmall"; 
    std::filesystem::path siftsmall_path = dir/"siftsmall_base.fvecs";
    std::filesystem::path siftsmall_query_path = dir/"siftsmall_query.fvecs"; 
    std::filesystem::path siftsmall_truth_path= dir/"siftsmall_groundtruth.ivecs";


    if (!std::filesystem::exists(siftsmall_path) || !std::filesystem::exists(siftsmall_query_path) || !std::filesystem::exists(siftsmall_truth_path)){ 
        GTEST_SKIP() << "run scripts/download_data.sh"; 
    }

    const auto base = vecdb::load_fvecs(siftsmall_path); 
    const auto query = vecdb::load_fvecs(siftsmall_query_path); 
    const auto truth = vecdb::load_ivecs(siftsmall_truth_path); 

    const std::size_t k = 10; 

    for (std::size_t q = 0 ; q < query.n ; q++){ 

        std::vector<std::size_t> res = get_ids(vecdb::brute_force_knn(base , query.row(q) , k)); 
        const auto expected = truth.row(q); 
        std::vector<std::size_t> expected_size_t; 
        for (std::size_t i = 0 ; i < k ; i++){ 
            expected_size_t.push_back(static_cast<std::size_t>(expected[i])); 
        }

        EXPECT_EQ(expected_size_t , res) << "query " << q; 

    }
}


TEST(BruteForce , SiftSmall_k100){ 
    // check if siftsmall is downloaded

    std::filesystem::path dir = std::filesystem::path(VECDB_DATA_DIR)/"siftsmall"; 
    std::filesystem::path siftsmall_path = dir/"siftsmall_base.fvecs";
    std::filesystem::path siftsmall_query_path = dir/"siftsmall_query.fvecs"; 
    std::filesystem::path siftsmall_truth_path= dir/"siftsmall_groundtruth.ivecs";


    if (!std::filesystem::exists(siftsmall_path) || !std::filesystem::exists(siftsmall_query_path) || !std::filesystem::exists(siftsmall_truth_path)){ 
        GTEST_SKIP() << "run scripts/download_data.sh"; 
    }

    const auto base = vecdb::load_fvecs(siftsmall_path); 
    const auto query = vecdb::load_fvecs(siftsmall_query_path); 
    const auto truth = vecdb::load_ivecs(siftsmall_truth_path); 

    const std::size_t k = 100; 
    

    for(std::size_t q = 0; q < query.n ; q++){
        const auto res = vecdb::brute_force_knn(base , query.row(q) , k); 
        const auto expected = truth.row(q); 
        ASSERT_EQ(res.size() , k) << "query " << q; 
        for (std::size_t i = 0; i < k ; i++ ){ 
            const float res_dist = res[i].dist; 
            float expected_dist = vecdb::squared_l2(base.row(static_cast<std::size_t>(expected[i])) , query.row(q)); 
            EXPECT_FLOAT_EQ(res_dist, expected_dist) << "query " << q << " pos " << i;
        }

    }   
}