#include "generate_model.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(genModels, m) {

  m.doc() = "Python interface for random models";

  m.def("ER", [](int n, double p, int threads, uint64_t seed1, uint64_t seed2){
      //intialise vectors
      std::vector<std::thread> t(threads - 1);
      std::vector<std::vector<vertex_t>> row(threads);
      std::vector<std::vector<vertex_t>> col(threads);

      std::vector<std::pair<uint64_t,uint64_t>> new_seeds = hash_seeds(seed1,seed2,threads);

      //Start each thread
      for (size_t index = 0; index < threads - 1; ++index){
          t[index] = std::thread(&add_edges_ER, index, n, threads, p, std::ref(row[index]),
                                   std::ref(col[index]), new_seeds[index].first, new_seeds[index].second);
      }
      add_edges_ER(threads-1, n, threads, p, std::ref(row[threads-1]),std::ref(col[threads-1]),
                               new_seeds[threads-1].first, new_seeds[threads-1].first);

      // Wait until all threads stopped
      for (size_t i = 0; i < threads - 1; ++i) t[i].join();

      combine_threads(threads,row,col);

      py::dict output;
      output["row"] = row[0];
      output["col"] = col[0];
      return output;
  }, py::arg("n"),py::arg("p"),py::arg("threads"),py::arg("seed1")=std::numeric_limits<uint64_t>::max(),py::arg("seed2") = std::numeric_limits<uint64_t>::max());

  m.def("SBM", [](int n, std::vector<std::vector<prob_t>>& pathways, std::vector<mtype_t>& mtypes,
                        int threads, uint64_t seed1, uint64_t seed2) {
      //intialise vectors
      std::vector<std::thread> t(threads - 1);
      std::vector<std::vector<vertex_t>> row(threads);
      std::vector<std::vector<vertex_t>> col(threads);

      std::vector<std::pair<uint64_t,uint64_t>> new_seeds = hash_seeds(seed1,seed2,threads);

      //Start each thread
      for (size_t index = 0; index < threads - 1; ++index){
          t[index] = std::thread(&add_edges_SBM, index, n, threads, std::ref(row[index]),std::ref(col[index]),
                                  std::ref(pathways),std::ref(mtypes), new_seeds[index].first, new_seeds[index].second);
      }
      add_edges_SBM(threads-1, n, threads, std::ref(row[threads-1]),std::ref(col[threads-1]),
                      std::ref(pathways),std::ref(mtypes), new_seeds[threads-1].first, new_seeds[threads-1].first);

      for (size_t i = 0; i < threads - 1; ++i) t[i].join(); // Wait until all threads stopped

      combine_threads(threads,row,col);

      py::dict output;
      output["row"] = row[0];
      output["col"] = col[0];
      return output;
  }, py::arg("n"),py::arg("pathways"),py::arg("mtypes"),py::arg("threads"),
     py::arg("seed1")=std::numeric_limits<uint64_t>::max(),py::arg("seed2") = std::numeric_limits<uint64_t>::max());

    m.def("DD2", [](int n, double a, double b, std::vector<coord_t>& xyz, int threads,
                        uint64_t seed1, uint64_t seed2) {
        //intialise vectors
        std::vector<std::thread> t(threads - 1);
        std::vector<std::vector<vertex_t>> row(threads);
        std::vector<std::vector<vertex_t>> col(threads);

        std::vector<std::pair<uint64_t,uint64_t>> new_seeds = hash_seeds(seed1,seed2,threads);

        //Start each thread
    	for (size_t index = 0; index < threads - 1; ++index){
    		t[index] = std::thread(&add_edges_DD2, index, n, threads, a, b, std::ref(row[index]),std::ref(col[index]),
                                    std::ref(xyz), new_seeds[index].first, new_seeds[index].second);
    	}
    	add_edges_DD2(threads-1, n, threads, a, b, std::ref(row[threads-1]),std::ref(col[threads-1]),
                        std::ref(xyz), new_seeds[threads-1].first, new_seeds[threads-1].first);

		for (size_t i = 0; i < threads - 1; ++i) t[i].join(); // Wait until all threads stopped

        combine_threads(threads,row,col);

        py::dict output;
        output["row"] = row[0];
        output["col"] = col[0];
        return output;
    }, py::arg("n"),py::arg("a"),py::arg("b"),py::arg("xyz"),py::arg("threads"),
       py::arg("seed1")=std::numeric_limits<uint64_t>::max(),py::arg("seed2") = std::numeric_limits<uint64_t>::max());

    m.def("DD3", [](int n, double a1, double b1, double a2, double b2, std::vector<coord_t>& xyz,
                           std::vector<coeff_t>& depths, int threads, uint64_t seed1, uint64_t seed2) {
        //intialise vectors
        std::vector<std::thread> t(threads - 1);
        std::vector<std::vector<vertex_t>> row(threads);
        std::vector<std::vector<vertex_t>> col(threads);

        std::vector<std::pair<uint64_t,uint64_t>> new_seeds = hash_seeds(seed1,seed2,threads);

        //Start each thread
    	for (size_t index = 0; index < threads - 1; ++index){
    		t[index] = std::thread(&add_edges_DD3, index, n, threads, a1, b1, a2, b2, std::ref(row[index]),std::ref(col[index]),
                                    std::ref(xyz), std::ref(depths), new_seeds[index].first, new_seeds[index].second);
    	}
    	add_edges_DD3(threads-1, n, threads, a1, b1, a2, b2, std::ref(row[threads-1]),std::ref(col[threads-1]),
                        std::ref(xyz), std::ref(depths), new_seeds[threads-1].first, new_seeds[threads-1].first);

		for (size_t i = 0; i < threads - 1; ++i) t[i].join(); // Wait until all threads stopped

        combine_threads(threads,row,col);

        py::dict output;
        output["row"] = row[0];
        output["col"] = col[0];
        return output;
    }, py::arg("n"),py::arg("a1"),py::arg("b1"),py::arg("a2"),py::arg("b2"),py::arg("xyz"),py::arg("depths"),py::arg("threads"),
       py::arg("seed1")=std::numeric_limits<uint64_t>::max(),py::arg("seed2") = std::numeric_limits<uint64_t>::max());

    m.def("DD2_block_pre", [](int n, std::vector<std::pair<coeff_t,coeff_t>>& pathways,
                                     std::vector<mtype_t>& mtypes, std::vector<coord_t>& xyz,
                                     int threads, uint64_t seed1, uint64_t seed2) {
        //intialise vectors
        std::vector<std::thread> t(threads - 1);
        std::vector<std::vector<vertex_t>> row(threads);
        std::vector<std::vector<vertex_t>> col(threads);

        std::vector<std::pair<uint64_t,uint64_t>> new_seeds = hash_seeds(seed1,seed2,threads);

        //Start each thread
    	for (size_t index = 0; index < threads - 1; ++index){
    		t[index] = std::thread(&add_edges_DD2_block_pre, index, n, threads, std::ref(pathways), std::ref(row[index]),std::ref(col[index]),
                                    std::ref(xyz),std::ref(mtypes), new_seeds[index].first, new_seeds[index].second);
    	}
    	add_edges_DD2_block_pre(threads-1, n, threads, std::ref(pathways), std::ref(row[threads-1]),std::ref(col[threads-1]),
                        std::ref(xyz),std::ref(mtypes), new_seeds[threads-1].first, new_seeds[threads-1].first);

		for (size_t i = 0; i < threads - 1; ++i) t[i].join(); // Wait until all threads stopped

        combine_threads(threads,row,col);

        py::dict output;
        output["row"] = row[0];
        output["col"] = col[0];
        return output;
    }, py::arg("n"),py::arg("pathways"),py::arg("mtypes"),py::arg("xyz"),py::arg("threads"),
       py::arg("seed1")=std::numeric_limits<uint64_t>::max(),py::arg("seed2") = std::numeric_limits<uint64_t>::max());

    m.def("DD2_block", [](int n, std::vector<std::vector<std::pair<coeff_t,coeff_t>>>& pathways,
                                 std::vector<mtype_t>& mtypes, std::vector<coord_t>& xyz, int threads,
                                 uint64_t seed1, uint64_t seed2) {
        //intialise vectors
        std::vector<std::thread> t(threads - 1);
        std::vector<std::vector<vertex_t>> row(threads);
        std::vector<std::vector<vertex_t>> col(threads);

        std::vector<std::pair<uint64_t,uint64_t>> new_seeds = hash_seeds(seed1,seed2,threads);

        //Start each thread
    	for (size_t index = 0; index < threads - 1; ++index){
    		t[index] = std::thread(&add_edges_DD2_block, index, n, threads, std::ref(pathways), std::ref(row[index]),std::ref(col[index]),
                                    std::ref(xyz),std::ref(mtypes), new_seeds[index].first, new_seeds[index].second);
    	}
    	add_edges_DD2_block(threads-1, n, threads, std::ref(pathways), std::ref(row[threads-1]),std::ref(col[threads-1]),
                        std::ref(xyz),std::ref(mtypes), new_seeds[threads-1].first, new_seeds[threads-1].first);

		for (size_t i = 0; i < threads - 1; ++i) t[i].join(); // Wait until all threads stopped

        combine_threads(threads,row,col);

        py::dict output;
        output["row"] = row[0];
        output["col"] = col[0];
        return output;
    }, py::arg("n"),py::arg("pathways"),py::arg("mtypes"),py::arg("xyz"),py::arg("threads"),
       py::arg("seed1")=std::numeric_limits<uint64_t>::max(),py::arg("seed2") = std::numeric_limits<uint64_t>::max());
}
