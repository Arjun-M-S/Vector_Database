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


    void HNSW::add_link(std::uint32_t from , std::uint32_t to , int layer){ 
        if (layer == 0){ 
            if (level_0_count_[from] == Mmax0_){ 
                std::span<const std::uint32_t> curr = neighbors(from , 0); 
                std::vector<std::uint32_t> temp; 
                temp.insert(temp.begin() , curr.begin() , curr.end()); 
                temp.push_back(to); 
                std::vector<Neighbor> distances; 
                std::span<const float> from_vector = vector_of(from); 
                for (auto neighbor : temp){ 
                    distances.push_back(Neighbor{squared_l2(from_vector , vector_of(neighbor)) , neighbor}); 
                }

                std::sort(distances.begin() , distances.end());
                distances.pop_back(); 
                std::size_t start = from * Mmax0_; 
                for(std::size_t i = 0 ; i < Mmax0_ ; i++){ 
                    level_0_links_[start + i] = static_cast<std::uint32_t> (distances[i].id); 
                }
            }
            else{ 
                std::uint32_t count = level_0_count_[from]; 
                level_0_count_[from] += 1; 
                level_0_links_[from * Mmax0_ + static_cast<std::size_t>(count)] = to;  
            }

        }
        else{ 
            std::size_t count = upper_links_[from][layer-1].size();
            upper_links_[from][layer-1].push_back(to); 
            
            if (count + 1 > Mmax_){ 
                std::vector<Neighbor> distances;
                std::span<const float> from_vector = vector_of(from); 
                for(auto id : upper_links_[from][layer - 1]){ 
                    distances.push_back(Neighbor { squared_l2(from_vector , vector_of(id)) , id}); 
                } 
                std::sort(distances.begin() , distances.end()); 
                for(std::size_t i = 0; i < upper_links_[from][layer-1].size() ; i ++){ 
                    upper_links_[from][layer-1][i] = static_cast<std::uint32_t>(distances[i].id); 
                }
                upper_links_[from][layer-1].pop_back(); 
            } 
        }
    }


    std::size_t HNSW::insert(std::span<const float>  v){
        if (v.size() != d_){
            throw std::invalid_argument("dimensions do not match the database");  
        }

        // ====== Book_keeping ========

        if (n_ == UINT32_MAX){ 
            throw std::overflow_error("Database is full");
        }
        vectors_.insert(vectors_.end() , v.begin() , v.end());
        int level = random_level();
        levels_.push_back(level); 
        level_0_links_.resize(level_0_links_.size() + Mmax0_); 
        level_0_count_.push_back(0); 
        upper_links_.push_back(std::vector<std::vector<std::uint32_t>>(level));
        std::uint32_t id = static_cast<std::uint32_t>(n_++); 
        if (max_level_ == -1){ 
            entry_point_ = id; 
            max_level_ = level; 
            return id; 
        }

        std::vector<Neighbor> ep = { Neighbor {squared_l2(v , vector_of(entry_point_)) , entry_point_}};
        for (int i = max_level_ ; i > level; i--){ 
            ep = search_layer(v , ep , 1 , i); 
        } 

        for (int i = std::min(level,max_level_) ; i >= 0 ; i--){ 
            ep = search_layer(v , ep , ef_construction_ , i);
            std::size_t count = std::min(ep.size() , Mmax_);            
            for (std::size_t j = 0 ; j < count ; j ++){
                add_link(id , static_cast<std::uint32_t>(ep[j].id) , i); 
                add_link(static_cast<std::uint32_t>(ep[j].id) , id , i); 
            }
        }

        
        if (level > max_level_){
            entry_point_ = id; 
            max_level_ = level; 
        }

        return id; 
    }


    std::vector<Neighbor> HNSW::search(std::span<const float> query , std::size_t k , std::size_t ef_search) const{
        if (query.size() != d_){ 
            throw std::invalid_argument("Query has the wrong dimensions"); 
        }
        if (max_level_ == -1 || k == 0){ 
            return {};
        }

        std::vector <Neighbor> ep = {Neighbor {squared_l2(query , vector_of(entry_point_)) , entry_point_}}; 
        for (int layer = max_level_ ; layer > 0 ; layer--){
            ep = search_layer(query , ep , 1 , layer); 
        }

        std::vector<Neighbor> W= search_layer(query , ep , std::max(ef_search,k) , 0); 
        std::size_t count = std::min(W.size() , k); 
        
        W.resize(count); 
        return W; 


    }

}