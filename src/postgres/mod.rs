mod datatype;
mod gucs;
mod hooks;
mod index;

pub use gucs::BGWORKER_PORT;
pub use gucs::OPENAI_API_KEY_GUC;
pub use gucs::SEARCH_K;

pub unsafe fn init() {
    self::gucs::init();
    self::hooks::init();
    self::index::init();
}
