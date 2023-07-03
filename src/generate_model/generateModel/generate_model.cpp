#include <vector>
#include <thread>
#include <cstring>
#include <iostream>
#include <numeric>
#include <complex>
#include <chrono>
#include <cassert>
#include <regex>
#include <math.h>
#include <limits>
// #include <mutex>

typedef uint32_t vertex_t;
typedef std::array<double,3> coord_t;
typedef double prob_t;
typedef double coeff_t;
typedef uint8_t mtype_t;

// std::mutex mu;

//************************************pcg*************************************//
//Fast random number generator taken from https://github.com/wjakob/pcg32/pcg.h


class pcg32 {
	uint64_t state;
	uint64_t inc;
public:
	void seed(uint64_t initstate, uint64_t initseq) {
		this->state = 0U;
		this->inc = (initseq << 1U) | 1U;
		next();
		this->state += initstate;
		next();
	}

    //constructor with seed inputted
	pcg32(uint64_t sd1, uint64_t sd2) {
		seed(sd1,sd2);
	}

#pragma warning (push)
#pragma warning (disable : 4146) //pragmas disable error on intended "unsigned out of range" behavior
#pragma warning (disable : 4244)
	//Generate a uniformly distributed 32-bit random number
	uint32_t next()	{
		uint64_t oldstate = this->state;
		this->state = oldstate * 6364136223846793005ULL + this->inc;
		uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
		uint32_t rot = oldstate >> 59u;
		return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
	}


	float next_float() {
		return (next() >> 8) * (1.f / ((uint32_t) 1 << 24));
	}
#pragma warning (pop)
};

std::vector<std::pair<uint64_t,uint64_t>> hash_seeds(uint64_t seed1, uint64_t seed2, int threads){
    std::vector<std::pair<uint64_t,uint64_t>> the_seeds;
    for(int i = 0; i < threads; i++){
        uint64_t new_seed1;
        uint64_t new_seed2;
	    if (seed1 == std::numeric_limits<uint64_t>::max()){
	        new_seed1 = std::chrono::high_resolution_clock::now().time_since_epoch().count()+i*712;
	    } else {
	        new_seed1 = seed1+i*2354;
	    }

	    if (seed2 == std::numeric_limits<uint64_t>::max()){
	        std::hash<std::thread::id> pcg_hasher;
	        new_seed2 = 2 * (uint64_t) pcg_hasher(std::this_thread::get_id()) + 1 + i+413;
	    } else {
	 	   seed2 = seed2+i*5223;
	    }
        the_seeds.push_back(std::make_pair(new_seed1, new_seed2));
    }
    return the_seeds;
}


//**********************************cnpy**************************************//
//numpy save function taken from cnpy


template<typename T> std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
    //write in little endian
    for(size_t byte = 0; byte < sizeof(T); byte++) {
        char val = *((char*)&rhs+byte);
        lhs.push_back(val);
    }
    return lhs;
}
template<> std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);
template<> std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);
template<> std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs) {
    lhs.insert(lhs.end(),rhs.begin(),rhs.end());
    return lhs;
}
template<> std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs) {
    //write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for(size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

char BigEndianTest() {
    int x = 1;
    return (((char *)&x)[0]) ? '<' : '>';
}

char map_type(const std::type_info& t)
{
    if(t == typeid(float) ) return 'f';
    if(t == typeid(double) ) return 'f';
    if(t == typeid(long double) ) return 'f';

    if(t == typeid(int) ) return 'i';
    if(t == typeid(char) ) return 'i';
    if(t == typeid(short) ) return 'i';
    if(t == typeid(long) ) return 'i';
    if(t == typeid(long long) ) return 'i';

    if(t == typeid(unsigned char) ) return 'u';
    if(t == typeid(unsigned short) ) return 'u';
    if(t == typeid(unsigned long) ) return 'u';
    if(t == typeid(unsigned long long) ) return 'u';
    if(t == typeid(unsigned int) ) return 'u';

    if(t == typeid(bool) ) return 'b';

    if(t == typeid(std::complex<float>) ) return 'c';
    if(t == typeid(std::complex<double>) ) return 'c';
    if(t == typeid(std::complex<long double>) ) return 'c';

    else return '?';
}

template<typename T> std::vector<char> create_npy_header(const std::vector<size_t>& shape) {

    std::vector<char> dict;
    dict += "{'descr': '";
    dict += BigEndianTest();
    dict += map_type(typeid(T));
    dict += std::to_string(sizeof(T));
    dict += "', 'fortran_order': False, 'shape': (";
    dict += std::to_string(shape[0]);
    for(size_t i = 1;i < shape.size();i++) {
        dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if(shape.size() == 1) dict += ",";
    dict += "), }";
    //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(),remainder,' ');
    dict.back() = '\n';

    std::vector<char> header;
    header += (char) 0x93;
    header += "NUMPY";
    header += (char) 0x01; //major version of numpy format
    header += (char) 0x00; //minor version of numpy format
    header += (uint16_t) dict.size();
    header.insert(header.end(),dict.begin(),dict.end());

    return header;
}

template<typename T> void npy_save(std::string fname, const T* data, const std::vector<size_t> shape) {
    FILE* fp = fopen(fname.c_str(),"wb");

    std::vector<char> header = create_npy_header<T>(shape);
    size_t nels = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<size_t>());

    fseek(fp,0,SEEK_SET);
    fwrite(&header[0],sizeof(char),header.size(),fp);
    fseek(fp,0,SEEK_END);
    fwrite(data,sizeof(T),nels,fp);
    fclose(fp);
}

struct NpyArray {
    NpyArray(const std::vector<size_t>& _shape, size_t _word_size, bool _fortran_order, std::string _dtype) :
        shape(_shape), word_size(_word_size), fortran_order(_fortran_order), dtype(_dtype)
    {
        num_vals = 1;
        for(size_t i = 0;i < shape.size();i++) num_vals *= shape[i];
        data_holder = std::shared_ptr<std::vector<char>>(
            new std::vector<char>(num_vals * word_size));
    }

    template<typename T>
    T* data() {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template<typename T>
    const T* data() const {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template<typename T>
    std::vector<T> as_vec() const {
        const T* p = data<T>();
        return std::vector<T>(p, p+num_vals);
    }

    size_t num_bytes() const {
        return data_holder->size();
    }

    std::shared_ptr<std::vector<char>> data_holder;
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    size_t num_vals;
    std::string dtype;
};


void parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order, std::string& type) {
    char buffer[256];
    size_t res = fread(buffer,sizeof(char),11,fp);
    if(res != 11)
        throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer,256,fp);
    assert(header[header.size()-1] == '\n');

    size_t loc1, loc2;

    //fortran order
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1,4) == "True" ? true : false);

    //shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: '(' or ')'");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1+1,loc2-loc1-1);
    while(std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    //endian, word size, data type
    //byte order code | stands for not applicable.
    //not sure when this applies except for byte array
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'descr'");
    loc1 += 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    type = header[loc1+2];
    //char type = header[loc1+1];
    //assert(type == map_type(T));

    std::string str_ws = header.substr(loc1+2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0,loc2).c_str());
}


NpyArray npy_load(std::string fname) {

    FILE* fp = fopen(fname.c_str(), "rb");

    if(!fp) throw std::runtime_error("npy_load: Unable to open file "+fname);

    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    std::string dtype;
    parse_npy_header(fp,word_size,shape,fortran_order,dtype);

    NpyArray arr(shape, word_size, fortran_order, dtype);
    size_t nread = fread(arr.data<char>(),1,arr.num_bytes(),fp);
    if(nread != arr.num_bytes())
        throw std::runtime_error("load_the_npy_file: failed fread");

    fclose(fp);
    return arr;
}

void check_dtype8(NpyArray file){
    if (file.dtype != "1") { throw std::runtime_error("ERROR: mtypes must be 8 bit int."); }
}



//********************************Erdos-Renyi*********************************//

void add_edges_ER(int index, int n, int threads, prob_t p,
               std::vector<vertex_t>& this_row, std::vector<vertex_t>& this_col,
               uint64_t seed1, uint64_t seed2){
    pcg32 rng(seed1,seed2);
    for(int i = index; i < n; i=i+threads){
		for (int j = 0; j < i; j++) {
			if (rng.next_float() < p) {
				this_row.push_back(i);
                this_col.push_back(j);
			}
        }
        for (int j = i+1; j < n; j++) {
			if (rng.next_float() < p) {
				this_row.push_back(i);
                this_col.push_back(j);
			}
        }
    }
}




//**************************Stochastic Block Model****************************//

void add_edges_SBM(int index, int n, int threads,
               std::vector<vertex_t>& this_row, std::vector<vertex_t>& this_col,
               std::vector<std::vector<prob_t>>& pathways, std::vector<mtype_t>& neuron_info,
               uint64_t seed1, uint64_t seed2){

    pcg32 rng(seed1,seed2);
	for(int i = index; i < n; i=i+threads){
		for (int j = 0; j < i; j++) {
			if (rng.next_float() < pathways[neuron_info[i]][neuron_info[j]]) {
				this_row.push_back(i);
	            this_col.push_back(j);
			}
	    }
	    for (int j = i+1; j < n; j++) {
			if (rng.next_float() < pathways[neuron_info[i]][neuron_info[j]]) {
				this_row.push_back(i);
	            this_col.push_back(j);
			}
	    }
	}
}



//************************Distance Dependent 2nd Order************************//

coeff_t distance(int i, int j, std::vector<coord_t>& xyz){
    coeff_t a = xyz[i][0]-xyz[j][0];
    coeff_t b = xyz[i][1]-xyz[j][1];
    coeff_t c = xyz[i][2]-xyz[j][2];
    return sqrt(a*a+b*b+c*c);
}

coeff_t model_DD2(int i, int j, coeff_t a, coeff_t b, std::vector<coord_t>& xyz){
    a*exp(-b*distance(i,j,xyz));
    return a*exp(-b*distance(i,j,xyz));
}

void add_edges_DD2(int index, int n, int threads, coeff_t a, coeff_t b,
               std::vector<vertex_t>& this_row, std::vector<vertex_t>& this_col,
               std::vector<coord_t>& xyz, uint64_t seed1, uint64_t seed2){
    pcg32 rng(seed1,seed2);
    for(int i = index; i < n; i=i+threads){
		for (int j = 0; j < i; j++) {
			if (rng.next_float() < model_DD2(i,j,a,b,xyz)) {
				this_row.push_back(i);
                this_col.push_back(j);
			}
        }
        for (int j = i+1; j < n; j++) {
			if (rng.next_float() < model_DD2(i,j,a,b,xyz)) {
				this_row.push_back(i);
                this_col.push_back(j);
			}
        }
    }
}

//************************Distance Dependent 3rd Order************************//

coeff_t model_DD3(int i, int j, coeff_t a1, coeff_t b1, coeff_t a2, coeff_t b2, std::vector<coord_t>& xyz, std::vector<coeff_t>& depth){
    coeff_t dz = depth[i]-depth[j];
    coeff_t x = distance(i,j,xyz);
    if (dz < 0) { return a1*exp(-b1*x); }
    if (dz > 0) { return a2*exp(-b2*x); }
    return (a1*a2*exp(x*-(b1+b2)))/2;
}

void add_edges_DD3(int index, int n, int threads, coeff_t a1, coeff_t b1, coeff_t a2, coeff_t b2,
               std::vector<vertex_t>& this_row, std::vector<vertex_t>& this_col, std::vector<coord_t>& xyz,
               std::vector<coeff_t>& depth, uint64_t seed1, uint64_t seed2){
    pcg32 rng(seed1,seed2);
    for(int i = index; i < n; i=i+threads){
		for (int j = 0; j < i; j++) {
			if (rng.next_float() < model_DD3(i,j,a1,b1,a2,b2,xyz,depth)) {
				this_row.push_back(i);
                this_col.push_back(j);
			}
        }
        for (int j = i+1; j < n; j++) {
			if (rng.next_float() < model_DD3(i,j,a1,b1,a2,b2,xyz,depth)) {
				this_row.push_back(i);
                this_col.push_back(j);
			}
        }
    }
}
//************************Distance Dependent Stochastic Block Model 1************************//

coeff_t model_DD2_block_pre(int i, int j, std::vector<std::pair<coeff_t,coeff_t>>& pathways, std::vector<coord_t>& xyz, std::vector<mtype_t>& neuron_info){
    return pathways[neuron_info[i]].first*exp(-pathways[neuron_info[i]].second*distance(i,j,xyz));
}

void add_edges_DD2_block_pre(int index, int n, int threads, std::vector<std::pair<coeff_t,coeff_t>>& pathways,
               std::vector<vertex_t>& this_row, std::vector<vertex_t>& this_col, std::vector<coord_t>& xyz,
               std::vector<mtype_t>& neuron_info, uint64_t seed1, uint64_t seed2){
    pcg32 rng(seed1,seed2);
    for(int i = index; i < n; i=i+threads){
		for (int j = 0; j < i; j++) {
			if (rng.next_float() < model_DD2_block_pre(i,j,pathways,xyz,neuron_info)) {
				this_row.push_back(i);
                this_col.push_back(j);
			}
        }
        for (int j = i+1; j < n; j++) {
			if (rng.next_float() < model_DD2_block_pre(i,j,pathways,xyz,neuron_info)) {
				this_row.push_back(i);
                this_col.push_back(j);
			}
        }
    }
}

//************************Distance Dependent Stochastic Block Model 2************************//

coeff_t model_DD2_block(int i, int j, std::vector<std::vector<std::pair<coeff_t,coeff_t>>>& pathways,
                                   std::vector<coord_t>& xyz, std::vector<mtype_t>& neuron_info){
    return pathways[neuron_info[i]][neuron_info[j]].first*exp(-pathways[neuron_info[i]][neuron_info[j]].second*distance(i,j,xyz));
}

void add_edges_DD2_block(int index, int n, int threads, std::vector<std::vector<std::pair<coeff_t,coeff_t>>>& pathways,
               std::vector<vertex_t>& this_row, std::vector<vertex_t>& this_col, std::vector<coord_t>& xyz,
               std::vector<mtype_t>& neuron_info, uint64_t seed1, uint64_t seed2){
    pcg32 rng(seed1,seed2);
    for(int i = index; i < n; i=i+threads){
		for (int j = 0; j < i; j++) {
			if (rng.next_float() < model_DD2_block(i,j,pathways,xyz,neuron_info)) {
				this_row.push_back(i);
                this_col.push_back(j);
			}
        }
        for (int j = i+1; j < n; j++) {
			if (rng.next_float() < model_DD2_block(i,j,pathways,xyz,neuron_info)) {
				this_row.push_back(i);
                this_col.push_back(j);
			}
        }
    }
}
//**********************************main**************************************//

void combine_threads(int threads, std::vector<std::vector<vertex_t>>& row, std::vector<std::vector<vertex_t>>& col){
    //Combine output from threads
    for (int i = 1; i < threads; i++){
        std::move( row[i].begin(), row[i].end(), std::back_inserter(row[0]));
        row[i].clear();
        std::move( col[i].begin(), col[i].end(), std::back_inserter(col[0]));
        col[i].clear();
    }
}
