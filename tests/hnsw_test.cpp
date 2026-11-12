#include "vecdb/hnsw.hpp"
#include "vecdb/dataset.hpp"

#include <gtest/gtest.h>

#include <iostream> 
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>
#include <algorithm>


#define VECDB_DATA_DIR "/home/arjun/Desktop/vector_database/data"

namespace {

std::vector<std::size_t> get_ids(const std::vector<vecdb::Neighbor>& neighbors) {
  std::vector<std::size_t> ids(neighbors.size());
  for (std::size_t i = 0; i < neighbors.size(); i++) {
    ids[i] = static_cast<std::size_t>(neighbors[i].id);
  }
  return ids;
}

// Inserts d-dimensional points (stored flat, back to back) in order, so point i gets id i.
vecdb::HNSW build_index(const std::vector<float>& points, std::size_t d) {
  vecdb::HNSW index(d);
  for (std::size_t i = 0; i < points.size(); i += d) {
    index.insert(std::span<const float>(points.data() + i, d));
  }
  return index;
}

// Same points as the brute-force toy set 1. Query (4, 2): squared distances 10, 9, 1, 1, 13, 10.
const std::vector<float> kToySet1 = {1, 1, 1, 2, 4, 1, 5, 2, 2, 5, 3, 5};

}  // namespace

// With 6 points and M = 16 every node links to every other, so HNSW is exact here:
// it must match brute force, including the tie order (smaller id first).
TEST(Hnsw, ToySet1MatchesBruteForceAtEveryK) {
  const vecdb::HNSW index = build_index(kToySet1, 2);
  const std::vector<float> q = {4, 2};

  struct Case {
    std::size_t k;
    std::vector<std::size_t> expected;
  };
  const std::vector<Case> cases = {
      {1, {2}},
      {2, {2, 3}},                    // ids 2 and 3 tie at distance 1
      {4, {2, 3, 1, 0}},              // ids 0 and 5 tie at 10 for the last slot
      {6, {2, 3, 1, 0, 5, 4}},        // k = n
      {10, {2, 3, 1, 0, 5, 4}},       // k > n: return all n
  };
  for (const Case& c : cases) {
    SCOPED_TRACE("k = " + std::to_string(c.k));
    EXPECT_EQ(get_ids(index.search(q, c.k, 50)), c.expected);
  }
}

TEST(Hnsw, ToySet2ExactCopyComesFirst) {
  const vecdb::HNSW index = build_index({1, 1, 3, 3, 2, 0}, 2);
  const std::vector<float> q = {1, 1};
  const auto result = index.search(q, 3, 50);
  EXPECT_EQ(get_ids(result), (std::vector<std::size_t>{0, 2, 1}));
  ASSERT_FALSE(result.empty());
  EXPECT_FLOAT_EQ(result[0].dist, 0.0f);
}

TEST(Hnsw, SinglePointIsFoundWithItsDistance) {
  const vecdb::HNSW index = build_index({3, 4}, 2);
  const std::vector<float> q = {0, 0};
  const auto result = index.search(q, 1, 10);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].id, 0u);
  EXPECT_FLOAT_EQ(result[0].dist, 25.0f);
}

TEST(Hnsw, EmptyIndexReturnsNothing) {
  const vecdb::HNSW index(2);
  const std::vector<float> q = {1, 1};
  EXPECT_TRUE(index.search(q, 5, 10).empty());
}

TEST(Hnsw, KZeroReturnsNothing) {
  const vecdb::HNSW index = build_index(kToySet1, 2);
  const std::vector<float> q = {4, 2};
  EXPECT_TRUE(index.search(q, 0, 10).empty());
}

// A beam narrower than k can't hold k results; search must widen it to at least k.
TEST(Hnsw, EfSearchSmallerThanKStillReturnsK) {
  const vecdb::HNSW index = build_index(kToySet1, 2);
  const std::vector<float> q = {4, 2};
  EXPECT_EQ(get_ids(index.search(q, 4, 1)), (std::vector<std::size_t>{2, 3, 1, 0}));
}

TEST(Hnsw, InsertReturnsSequentialIds) {
  vecdb::HNSW index(2);
  const std::vector<float> a = {1, 1}, b = {2, 2}, c = {3, 3};
  EXPECT_EQ(index.insert(a), 0u);
  EXPECT_EQ(index.insert(b), 1u);
  EXPECT_EQ(index.insert(c), 2u);
}

TEST(Hnsw, InsertWrongDimensionThrows) {
  vecdb::HNSW index(2);
  const std::vector<float> too_long = {1, 2, 3};
  const std::vector<float> empty;
  EXPECT_THROW(index.insert(too_long), std::invalid_argument);
  EXPECT_THROW(index.insert(empty), std::invalid_argument);
}

// A rejected insert must not use up an id: validation happens before n_++.
TEST(Hnsw, RejectedInsertDoesNotConsumeAnId) {
  vecdb::HNSW index(2);
  const std::vector<float> bad = {1, 2, 3};
  const std::vector<float> good = {1, 2};
  EXPECT_THROW(index.insert(bad), std::invalid_argument);
  EXPECT_EQ(index.insert(good), 0u);
}

TEST(Hnsw, SearchWrongDimensionThrows) {
  const vecdb::HNSW index = build_index(kToySet1, 2);
  const std::vector<float> q = {4, 2, 0};
  EXPECT_THROW(index.search(q, 3, 10), std::invalid_argument);
}

TEST(Hnsw, ConstructorRejectsBadParameters) {
  EXPECT_THROW(vecdb::HNSW(128, 42, 1), std::invalid_argument);  // M = 1: ln(1) = 0
  EXPECT_THROW(vecdb::HNSW(0), std::invalid_argument);           // d = 0
}

TEST(Hnsw , SiftSmall_k10){ 
    std::filesystem::path dir = std::filesystem::path(VECDB_DATA_DIR)/"siftsmall"; 
    std::filesystem::path siftsmall_path = dir/"siftsmall_base.fvecs";
    std::filesystem::path siftsmall_query_path = dir/"siftsmall_query.fvecs"; 
    std::filesystem::path siftsmall_truth_path= dir/"siftsmall_groundtruth.ivecs";

    if (!std::filesystem::exists(siftsmall_path) || !std::filesystem::exists(siftsmall_query_path) || !std::filesystem::exists(siftsmall_truth_path)){ 
        GTEST_SKIP() << "run scripts/download_data.sh"; 
    }

    const auto base = vecdb::load_fvecs(siftsmall_path); 
    const auto query = vecdb::load_fvecs(siftsmall_query_path);

    const std::size_t k = 10; 
    vecdb::HNSW index(base.d); 
    for (std::size_t i = 0 ; i < base.n ; i ++){ 
        index.insert(base.row(i)); 
    }


    double running_got = 0.0; 
    double running_tot = 0.0;
    int ef = 100; 
    
    for(std::size_t q = 0; q < query.n ; q++){

        const auto res = vecdb::brute_force_knn(base , query.row(q) , k); 
        const auto HNSW_res = index.search(query.row(q) , k , ef ); 
        auto res_ids = get_ids(res); 
        auto HNSW_res_ids = get_ids(HNSW_res);
        std::sort(res_ids.begin() , res_ids.end()); 
        std::sort(HNSW_res_ids.begin() , HNSW_res_ids.end());
        std::vector<size_t> intersection; 
        std::set_intersection(res_ids.begin() , res_ids.end() , HNSW_res_ids.begin() , HNSW_res_ids.end() , std::back_inserter(intersection)); 
        running_got += intersection.size(); 
        running_tot += res_ids.size();
    }
    
    const double recall = running_got / running_tot;
    std::cout << "efSearch " << ef << ": recall@10 = " << recall << '\n';
    if (ef == 10){ 
        EXPECT_GE(recall , 0.92); 
    }
    else{
      
    EXPECT_GE(recall,0.98);
    }


}