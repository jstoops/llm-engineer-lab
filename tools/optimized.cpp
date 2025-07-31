
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>

// LCG generator
std::mt19937 lcg_gen(42);

// Generate random numbers
std::vector<int> generate_random_numbers(int n, int min_val, int max_val) {
    std::uniform_int_distribution<int> dist(min_val, max_val);
    std::vector<int> random_numbers(n);
    for (int i = 0; i < n; ++i) {
        random_numbers[i] = dist(lcg_gen);
    }
    return random_numbers;
}

// Find maximum subarray sum
int max_subarray_sum(const std::vector<int>& random_numbers) {
    int max_sum = INT_MIN;
    int current_sum = 0;
    for (int num : random_numbers) {
        current_sum += num;
        if (current_sum > max_sum) {
            max_sum = current_sum;
        }
        if (current_sum < 0) {
            current_sum = 0;
        }
    }
    return max_sum;
}

// Total maximum subarray sum
int total_max_subarray_sum(int n, int initial_seed, int min_val, int max_val) {
    int total_sum = 0;
    std::mt19937 lcg_gen(initial_seed);
    for (int i = 0; i < 20; ++i) {
        std::vector<int> random_numbers = generate_random_numbers(n, min_val, max_val);
        total_sum += max_subarray_sum(random_numbers);
    }
    return total_sum;
}

int main() {
    int n = 10000;         // Number of random numbers
    int initial_seed = 42; // Initial seed for the LCG
    int min_val = -10;     // Minimum value of random numbers
    int max_val = 10;      // Maximum value of random numbers

    // Timing the function
    auto start_time = std::chrono::high_resolution_clock::now();
    int result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << std::chrono::duration<double>(end_time - start_time).count() << " seconds" << std::endl;

    return 0;
}
