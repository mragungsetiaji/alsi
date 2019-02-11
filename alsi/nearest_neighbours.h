
#ifndef ALSI_NEAREST_NEIGHBOURS_H_
#define ALSI_NEAREST_NEIGHBOURS_H_

#include <algorithm>
#include <vector>
#include <utility>
#include <functional>

namespace implicit {

template <typename Index, typename Value>
struct TopK {
    explicit TopK(size_t K) : K(K) {}

    void operator()(Index index, Value score) {
        if ((results.size() < K) || (score > results[0].first)) {
            if (results.size() >= K) {
                std::pop_heap(results.begin(), results.end(), heap_order);
                results.pop_back();
            }

            results.push_back(std::make_pair(score, index));
            std::push_heap(results.begin(), results.end(), heap_order);
        }
    }

    size_t K;
    std::vector<std::pair<Value, Index> > results;
    std::greater<std::pair<Value, Index> > heap_order;
};

template <typename Index, typename Value>
class SparseMatrixMultiplier {
 public:
    explicit SparseMatrixMultiplier(Index item_count)
        : sums(item_count, 0), nonzeros(item_count, -1), head(-2), length(0) {
    }

    void add(Index index, Value value) {
        sums[index] += value;

        if (nonzeros[index] == -1) {
            nonzeros[index] = head;
            head = index;
            length += 1;
        }
    }

    template <typename Function>
    void foreach(Function & f) {  // NOLINT(*)
        for (int i = 0; i < length; ++i) {
            Index index = head;
            f(index, sums[index]);

            head = nonzeros[head];
            sums[index] = 0;
            nonzeros[index] = -1;
        }

        length = 0;
        head = -2;
    }

    Index nnz() const { return length; }

    std::vector<Value> sums;

 protected:
    std::vector<Index> nonzeros;
    Index head, length;
};
} 
#endif  