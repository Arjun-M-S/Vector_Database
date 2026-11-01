#include "vecdb/hnsw.hpp"
#include "vecdb/distance.hpp"

#include <cmath>        
#include <functional>   
#include <queue>        
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cassert> 


namespace vecdb{ 

    // Initialize
    HNSW::HNSW(std::size_t d, std::uint32_t seed, std::size_t M, std::size_t efConstruction)
            : d_(d) , M_(M) , Mmax_(M), Mmax0_(2*M), ef_construction_(std::max(efConstruction , M)), m_L_ (1/std::log(M)) , rng_ (seed) 
            
            {
                if (M < 2){ 
                    throw std::invalid_argument("M value must be greater than 1"); 
                }

                if(d < 1){ 
                    throw std::invalid_argument("vector must have at least 1 dimension"); 
                }

            }

    std::span<const float> HNSW::vector_of(std::uint32_t id) const{ 
        return {vectors_.data() + id * d_, d_ }; 
    }

    std::span<const std::uint32_t> HNSW::neighbors(std::uint32_t id , int level) const{
        if (level == 0){ 
            std::size_t used = static_cast<std::size_t>(level_0_count_[id]); 
            return {level_0_links_.data() + id * Mmax0_ , used};
        }
        else { 
            return {upper_links_[id][level - 1]}; 
        }
    }

    int HNSW::random_level() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng_);
    if (u == 0.0) {
        u = 1.0;   // log(0) is infinite; map 0 to 1 (level 0)
    }
    return static_cast<int>(std::floor(-std::log(u) * m_L_));
    }
    
    std::vector<Neighbor> HNSW::search_layer(std::span<const float> query, const std::vector<Neighbor>& entry_points, std::size_t ef, int layer) const{

        assert(ef >= 1); 
        std::vector<bool> visited(n_ , false); 
        std::priority_queue<Neighbor , std::vector<Neighbor> , std::greater<Neighbor>> candidates; // min heap 
        std::priority_queue<Neighbor> results; // max heap
        
        for (const Neighbor& entry_point : entry_points){ 
            visited[entry_point.id] = true; 
            candidates.push(entry_point); 
            results.push(entry_point); 
            while( results.size() > ef){
                results.pop(); 
            }
        }

        while (!candidates.empty()){ 
            Neighbor candidate = candidates.top(); 
            candidates.pop(); 
            if (candidate.dist > results.top().dist){ 
                break;
            }
            std::span<const std::uint32_t> candidate_neighbors = neighbors(static_cast<std::uint32_t> (candidate.id) , layer);


            for (std::uint32_t cand : candidate_neighbors){ 
                if (visited[cand]){
                    continue;
                }
                visited[cand] = true; 
                std::span<const float> neighbor_vector = vector_of(cand); 
                float dist = squared_l2(neighbor_vector , query);
                Neighbor candidate_neighbor = Neighbor {dist , cand};
                

                if (results.size() < ef || results.top().dist > dist){ 
                    results.push(candidate_neighbor);  
                    candidates.push(candidate_neighbor);
                    if (results.size() > ef){ 
                        results.pop(); 
                    }
                }
                    
            } 

            
        }
        

        std::vector<Neighbor> res(results.size()); 
        for(std::size_t i = results.size(); i > 0 ; i--){ 
            res[i-1] = results.top(); 
            results.pop();
        }
        return res; 
    }
}