#include "vecdb/brute_force.hpp"
#include "vecdb/distance.hpp"

#include <queue>
#include <stdexcept>

namespace vecdb{

    std::vector<Neighbor> brute_force_knn(const Matrix<float> &database , std::span<const float> query, std::size_t k){
        //check if query dimensions == vector dimensions
        if (query.size() != database.d){
            throw std::invalid_argument("query dimensions do not match the database vector dimensions");
        }

        // Handles k = 0 
        if (k == 0){ 
            return {};
        }

        std::priority_queue<Neighbor> heap; 

        for (std::size_t i = 0 ; i < database.n ; i ++){
            float dist = squared_l2(database.row(i) , query); 
            Neighbor temp = Neighbor{dist , i};
            if (((heap.size() == k) && (temp < heap.top()))){
                heap.pop();
                heap.push(temp);
            }

            else if (heap.size() < k){
                heap.push(temp); 
            }
        }

        std::vector<Neighbor> res(heap.size()); 

        for (std::size_t j = heap.size() ; j > 0 ; j--){
            res[j - 1] = heap.top();
            heap.pop();
        }

        return res;
        
    }
}