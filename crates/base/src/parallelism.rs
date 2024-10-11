use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub use rayon::iter::ParallelIterator;

pub trait Parallelism: Send + Sync {
    fn check(&self);

    #[allow(clippy::wrong_self_convention)]
    fn into_par_iter<I: rayon::iter::IntoParallelIterator>(&self, x: I) -> I::Iter;
}

struct ParallelismCheckPanic;

pub struct RayonParallelism {
    stop: Arc<AtomicBool>,
}

impl RayonParallelism {
    pub fn scoped<R>(
        num_threads: usize,
        stop: Arc<AtomicBool>,
        f: impl FnOnce(&Self) -> R,
    ) -> Result<Option<R>, Box<dyn Any + Send>> {
        match std::panic::catch_unwind(AssertUnwindSafe(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .panic_handler(|e| {
                    if e.downcast_ref::<ParallelismCheckPanic>().is_some() {
                        return;
                    }
                    log::error!("Asynchronous task panickied.");
                })
                .build_scoped(
                    |thread| thread.run(),
                    |_| {
                        let pool = Self { stop: stop.clone() };
                        f(&pool)
                    },
                )
        })) {
            Ok(Ok(r)) => Ok(Some(r)),
            Ok(Err(e)) => Err(Box::new(e)),
            Err(e) if e.downcast_ref::<ParallelismCheckPanic>().is_some() => Ok(None),
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}

impl Parallelism for RayonParallelism {
    fn check(&self) {
        use std::sync::atomic::Ordering;
        if self.stop.load(Ordering::Relaxed) {
            std::panic::panic_any(ParallelismCheckPanic);
        }
    }

    fn into_par_iter<I: rayon::iter::IntoParallelIterator>(&self, x: I) -> I::Iter {
        x.into_par_iter()
    }
}
