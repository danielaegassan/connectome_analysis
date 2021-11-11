//##############################################################################
//IMPORT LIBRARIES
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <thread>
#include <functional>
#include <deque>
#include <sstream>
#include <cmath>
#include <array>

//##############################################################################
// DEFINITIONS
typedef uint64_t vertex_index_t;
typedef int64_t index_t;
typedef float value_t;

#define PARALLEL_THREADS 8



//##############################################################################
//DIRECTED GRAPH CLASS

class directed_graph_t {
public:
	// The filtration values of the vertices
	vertex_index_t number_of_vertices;
        bool transpose;

	// These are the incidences as a matrix of 64-bit masks
	std::deque<size_t> incidence_outgoing;
	size_t incidence_row_length;

	// Assume by default that the edge density will be roughly one percent
	directed_graph_t(vertex_index_t _number_of_vertices, bool _transpose=false, float density_hint = 0.01)
	    : number_of_vertices(_number_of_vertices), transpose(_transpose), incidence_row_length((_number_of_vertices >> 6) + 1) {
		incidence_outgoing.resize(incidence_row_length * _number_of_vertices, 0);
	}

	vertex_index_t vertex_number() const { return number_of_vertices; }

	void add_edge(vertex_index_t v, vertex_index_t w) {
		if(transpose){
                    vertex_index_t v_temp = v;
                    v=w;
                    w=v_temp;
                }
                const size_t ww = w >> 6;
		incidence_outgoing[v * incidence_row_length + ww] |= 1UL << ((w - (ww << 6)));
	}

	bool is_connected_by_an_edge(vertex_index_t from, vertex_index_t to) const {
		const auto t = to >> 6;
		return incidence_outgoing[incidence_row_length * from + t] & (1UL << (to - (t << 6)));
	}

	size_t get_outgoing_chunk(vertex_index_t from, size_t chunk_number) const {
		return incidence_outgoing[incidence_row_length * from + chunk_number];
	}
};

//##############################################################################
//DIRECTED FLAG COMPLEX CLASS

class directed_flag_complex_t {
public:
	const directed_graph_t& graph;
	directed_flag_complex_t(const directed_graph_t& _graph) : graph(_graph) {}

public:
	template <typename Func> void for_each_cell(Func& f, std::vector<vertex_index_t>& do_vertices, std::vector<std::vector<std::vector<vertex_index_t>>>& contain_counts, int min_dimension, int max_dimension = -1) {
		std::array<Func*, 1> fs{&f};
		for_each_cell(fs, do_vertices, contain_counts, min_dimension, max_dimension);
	}

	template <typename Func, size_t number_of_threads>
	void for_each_cell(std::array<Func*, number_of_threads>& fs, std::vector<vertex_index_t>& do_vertices, std::vector<std::vector<std::vector<vertex_index_t>>>& contain_counts, int min_dimension, int max_dimension = -1) {
		if (max_dimension == -1) max_dimension = min_dimension;
		std::thread t[number_of_threads - 1];

		for (size_t index = 0; index < number_of_threads - 1; ++index)
			t[index] = std::thread(&directed_flag_complex_t::worker_thread<Func>, this, number_of_threads, index,
			                       fs[index], min_dimension, max_dimension, std::ref(do_vertices), std::ref(contain_counts));

		// Also do work in this thread, namely the last bit
		worker_thread(number_of_threads, number_of_threads - 1, fs[number_of_threads - 1], min_dimension,
		              max_dimension, do_vertices, contain_counts);

		// Wait until all threads stopped
		for (size_t i = 0; i < number_of_threads - 1; ++i) t[i].join();
	}

private:
	template <typename Func>
	void worker_thread(int number_of_threads, int thread_id, Func* f, int min_dimension, int max_dimension,
                       std::vector<vertex_index_t>& do_vertices, std::vector<std::vector<std::vector<vertex_index_t>>>& contain_counts) {
		const size_t vertices_per_thread = graph.vertex_number() / number_of_threads;

		std::vector<vertex_index_t> first_position_vertices;
		for (size_t index = thread_id; index < do_vertices.size(); index += number_of_threads)
			first_position_vertices.push_back(do_vertices[index]);

		vertex_index_t prefix[max_dimension + 1];

		do_for_each_cell(f, min_dimension, max_dimension, first_position_vertices, prefix, 0, thread_id, do_vertices.size(), contain_counts);

		f->done();
	}

	template <typename Func>
	void do_for_each_cell(Func* f, int min_dimension, int max_dimension,
	                      const std::vector<vertex_index_t>& possible_next_vertices, vertex_index_t* prefix,
	                      unsigned short prefix_size, int thread_id, size_t number_of_vertices, std::vector<std::vector<std::vector<vertex_index_t>>>& contain_counts) {
		// As soon as we have the correct dimension, execute f
		if (prefix_size >= min_dimension + 1) { (*f)(prefix, prefix_size); }
        for(int i = 0; i < prefix_size; i++){
            while(contain_counts[thread_id][prefix[i]].size() < prefix_size){
                contain_counts[thread_id][prefix[i]].push_back(0);
            }
            contain_counts[thread_id][prefix[i]][prefix_size-1]++;
        }

		// If this is the last dimension we are interested in, exit this branch
		if (prefix_size == max_dimension + 1) return;

        for (auto vertex : possible_next_vertices) {
			// We can write the cell given by taking the current vertex as the maximal element
			prefix[prefix_size] = vertex;

			// And compute the next elements
			std::vector<vertex_index_t> new_possible_vertices;
			if (prefix_size > 0) {
				for (auto v : possible_next_vertices) {
					if (vertex != v && graph.is_connected_by_an_edge(vertex, v)) new_possible_vertices.push_back(v);
				}
			} else {
				// Get outgoing vertices of v in chunks of 64
				for (size_t offset = 0; offset < graph.incidence_row_length; offset++) {
					size_t bits = graph.get_outgoing_chunk(vertex, offset);

					size_t vertex_offset = offset << 6;
					while (bits > 0) {
						// Get the least significant non-zero bit
						int b = __builtin_ctzl(bits);

						// Unset this bit
						bits &= ~(1UL << b);

						new_possible_vertices.push_back(vertex_offset + b);
					}
				}
			}

            do_for_each_cell(f, min_dimension, max_dimension, new_possible_vertices, prefix, prefix_size + 1, thread_id, number_of_vertices, contain_counts);
		}
	}
};


//##############################################################################
//CELL COUNTER STRUCT

struct cell_counter_t {
	void done() {}
	void operator()(vertex_index_t* first_vertex, int size) {
		// Add (-1)^size to the Euler characteristic
		if (size & 1)
			ec++;
		else
			ec--;

		if (cell_counts.size() < size) { cell_counts.resize(size, 0); }
		cell_counts[size - 1]++;
	}

	int64_t euler_characteristic() const { return ec; }
	std::vector<size_t> cell_count() const { return cell_counts; }

private:
	int64_t ec = 0;
	std::vector<size_t> cell_counts;
};

//##############################################################################
//COUNT CELL FUNCTION

std::vector<std::vector<vertex_index_t>> count_cells(directed_graph_t& graph) {
	directed_flag_complex_t complex(graph);


   std::vector<vertex_index_t> do_vertices;
   for(int i = 0; i < graph.vertex_number(); i++){ do_vertices.push_back(i); }


    std::vector<std::vector<std::vector<vertex_index_t>>> contain_counts(PARALLEL_THREADS,
			                                                     std::vector<std::vector<vertex_index_t>>(graph.vertex_number(),
																														std::vector<vertex_index_t>(0)));

	std::array<cell_counter_t*, PARALLEL_THREADS> cell_counter;
	for (int i = 0; i < PARALLEL_THREADS; i++)
		cell_counter[i] = new cell_counter_t();

		complex.for_each_cell(cell_counter, do_vertices, contain_counts, 0, 10000);


    for(int i = 1; i < contain_counts.size(); i++){
        for(int j = 0; j < contain_counts[i].size(); j++){
            while(contain_counts[0][j].size() < contain_counts[i][j].size()){
                contain_counts[0][j].push_back(0);
            }
            for(int k = 0; k < contain_counts[i][j].size(); k++){
                contain_counts[0][j][k] += contain_counts[i][j][k];
            }
        }
    }

	return contain_counts[0];
}
