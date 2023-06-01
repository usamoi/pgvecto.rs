#include "hnswlib.h"
#include "rust/cxx.h"
#define fn auto
#define let auto

class Binding {
public:
    hnswlib::L2Space *space;
    hnswlib::HierarchicalNSW<float> *algo;
    Binding(size_t dim, size_t max_elements) {
        space = new hnswlib::L2Space(dim);
        algo = new hnswlib::HierarchicalNSW<float>(space, max_elements, 64, 500);
    }
    Binding(size_t dim, const std::string &location) {
        space = new hnswlib::L2Space(dim);
        algo = new hnswlib::HierarchicalNSW<float>(space, location);
    }
    ~Binding() {
        delete algo;
        delete space;
    }
};

fn binding_new(size_t dim)->Binding * {
    let self = new Binding(dim, 2097152);
    return self;
}

fn binding_delete(Binding *self) {
    delete self;
}

fn binding_save(Binding *self, const uint8_t *location) {
    std::string s((const char *)location);
    self->algo->saveIndex(s);
}

fn binding_load(size_t dim, const uint8_t *location)->Binding * {
    std::string s((const char *)location);
    let self = new Binding(dim, s);
    return self;
}

fn binding_insert(Binding *self, const uint8_t *data, size_t label) {
    self->algo->addPoint(data, label);
}

fn binding_search(Binding *self, const uint8_t *data, size_t k) -> rust::Vec<size_t> {
    let cxx = self->algo->searchKnnCloserFirst(data, k);
    let rust = rust::Vec<size_t>();
    for (let x : cxx) {
        rust.push_back(x.second);
    }
    return rust;
}
