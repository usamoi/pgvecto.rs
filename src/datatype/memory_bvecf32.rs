use base::vector::*;
use pgrx::pg_sys::Datum;
use pgrx::pg_sys::Oid;
use pgrx::pgrx_sql_entity_graph::metadata::ArgumentError;
use pgrx::pgrx_sql_entity_graph::metadata::Returns;
use pgrx::pgrx_sql_entity_graph::metadata::ReturnsError;
use pgrx::pgrx_sql_entity_graph::metadata::SqlMapping;
use pgrx::pgrx_sql_entity_graph::metadata::SqlTranslatable;
use pgrx::FromDatum;
use pgrx::IntoDatum;
use pgrx::UnboxDatum;
use std::alloc::Layout;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr::NonNull;

pub const HEADER_MAGIC: u16 = 3;

#[repr(C, align(8))]
pub struct BVecf32Header {
    varlena: u32,
    dims: u16,
    magic: u16,
    phantom: [u64; 0],
}

impl BVecf32Header {
    fn varlena(size: usize) -> u32 {
        (size << 2) as u32
    }
    fn layout(len: usize) -> Layout {
        u16::try_from(len).expect("Vector is too large.");
        let layout_alpha = Layout::new::<BVecf32Header>();
        let layout_beta = Layout::array::<u64>((len as u32).div_ceil(BVECF32_WIDTH) as _).unwrap();
        let layout = layout_alpha.extend(layout_beta).unwrap().0;
        layout.pad_to_align()
    }
    pub fn dims(&self) -> u32 {
        self.dims as u32
    }
    pub fn data(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.phantom.as_ptr(),
                (self.dims as u32).div_ceil(BVECF32_WIDTH) as usize,
            )
        }
    }
    pub fn as_borrowed(&self) -> BVecf32Borrowed<'_> {
        unsafe { BVecf32Borrowed::new_unchecked(self.dims(), self.data()) }
    }
}

pub enum BVecf32Input<'a> {
    Owned(BVecf32Output),
    Borrowed(&'a BVecf32Header),
}

impl<'a> BVecf32Input<'a> {
    pub unsafe fn new(p: NonNull<BVecf32Header>) -> Self {
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast()).unwrap()
        };
        if p != q {
            BVecf32Input::Owned(BVecf32Output(q))
        } else {
            unsafe { BVecf32Input::Borrowed(p.as_ref()) }
        }
    }
}

impl Deref for BVecf32Input<'_> {
    type Target = BVecf32Header;

    fn deref(&self) -> &Self::Target {
        match self {
            BVecf32Input::Owned(x) => x,
            BVecf32Input::Borrowed(x) => x,
        }
    }
}

pub struct BVecf32Output(NonNull<BVecf32Header>);

impl BVecf32Output {
    pub fn new(vector: BVecf32Borrowed<'_>) -> BVecf32Output {
        unsafe {
            let dims = vector.dims();
            let internal_dims = dims as u16;
            let layout = BVecf32Header::layout(dims as usize);
            let ptr = pgrx::pg_sys::palloc(layout.size()) as *mut BVecf32Header;
            ptr.cast::<u8>().add(layout.size() - 8).write_bytes(0, 8);
            std::ptr::addr_of_mut!((*ptr).varlena).write(BVecf32Header::varlena(layout.size()));
            std::ptr::addr_of_mut!((*ptr).magic).write(HEADER_MAGIC);
            std::ptr::addr_of_mut!((*ptr).dims).write(internal_dims);
            std::ptr::copy_nonoverlapping(
                vector.data().as_ptr(),
                (*ptr).phantom.as_mut_ptr(),
                dims.div_ceil(BVECF32_WIDTH) as usize,
            );
            BVecf32Output(NonNull::new(ptr).unwrap())
        }
    }
    pub fn into_raw(self) -> *mut BVecf32Header {
        let result = self.0.as_ptr();
        std::mem::forget(self);
        result
    }
}

impl Deref for BVecf32Output {
    type Target = BVecf32Header;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl DerefMut for BVecf32Output {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut() }
    }
}

impl Drop for BVecf32Output {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::pfree(self.0.as_ptr() as _);
        }
    }
}

impl<'a> FromDatum for BVecf32Input<'a> {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let ptr = NonNull::new(datum.cast_mut_ptr::<BVecf32Header>()).unwrap();
            unsafe { Some(BVecf32Input::new(ptr)) }
        }
    }
}

impl IntoDatum for BVecf32Output {
    fn into_datum(self) -> Option<Datum> {
        Some(Datum::from(self.into_raw() as *mut ()))
    }

    fn type_oid() -> Oid {
        let namespace =
            pgrx::pg_catalog::PgNamespace::search_namespacename(crate::SCHEMA_C_STR).unwrap();
        let namespace = namespace.get().expect("pgvecto.rs is not installed.");
        let t = pgrx::pg_catalog::PgType::search_typenamensp(c"bvector", namespace.oid()).unwrap();
        let t = t.get().expect("pg_catalog is broken.");
        t.oid()
    }
}

impl FromDatum for BVecf32Output {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let p = NonNull::new(datum.cast_mut_ptr::<BVecf32Header>())?;
            let q =
                unsafe { NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast())? };
            if p != q {
                Some(BVecf32Output(q))
            } else {
                let header = p.as_ptr();
                let vector = unsafe { (*header).as_borrowed() };
                Some(BVecf32Output::new(vector))
            }
        }
    }
}

unsafe impl UnboxDatum for BVecf32Output {
    type As<'src> = BVecf32Output;
    #[inline]
    unsafe fn unbox<'src>(d: pgrx::Datum<'src>) -> Self::As<'src>
    where
        Self: 'src,
    {
        let p = NonNull::new(d.sans_lifetime().cast_mut_ptr::<BVecf32Header>()).unwrap();
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast()).unwrap()
        };
        if p != q {
            BVecf32Output(q)
        } else {
            let header = p.as_ptr();
            let vector = unsafe { (*header).as_borrowed() };
            BVecf32Output::new(vector)
        }
    }
}

unsafe impl SqlTranslatable for BVecf32Input<'_> {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("bvector")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("bvector"))))
    }
}

unsafe impl SqlTranslatable for BVecf32Output {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("bvector")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("bvector"))))
    }
}

unsafe impl pgrx::callconv::BoxRet for BVecf32Output {
    unsafe fn box_in_fcinfo(self, fcinfo: pgrx::pg_sys::FunctionCallInfo) -> Datum {
        self.into_datum()
            .unwrap_or_else(|| unsafe { pgrx::fcinfo::pg_return_null(fcinfo) })
    }
}
