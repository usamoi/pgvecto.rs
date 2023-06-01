use std::ptr::null_mut;
use std::sync::mpsc::SyncSender;

use super::datatype::Vector;
use crate::bgworker::export::build;
use crate::bgworker::export::bulkdelete;
use crate::bgworker::export::insert;
use crate::bgworker::export::search;
use crate::bgworker::CommonOptions;
use crate::bgworker::Options;
use crate::postgres::datatype::VectorTypmod;
use crate::postgres::HeapPointer;
use crate::postgres::Scalar;
use pg_sys::Datum;
use pgrx::prelude::*;

const PROBES: usize = 64;

struct ScanState {
    data: Vec<HeapPointer>,
}

#[pgrx::pg_extern(sql = "
    CREATE OR REPLACE FUNCTION pgvectors_hnsw_amhandler(internal) RETURNS index_am_handler
    PARALLEL SAFE IMMUTABLE STRICT LANGUAGE c AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';
    CREATE ACCESS METHOD pgvectors_hnsw TYPE INDEX HANDLER pgvectors_hnsw_amhandler;
    COMMENT ON ACCESS METHOD pgvectors_hnsw IS 'HNSW index access method';
", requires = ["vector"])]
fn pgvectors_hnsw_amhandler(
    _fcinfo: pg_sys::FunctionCallInfo,
) -> pgrx::PgBox<pgrx::pg_sys::IndexAmRoutine> {
    let mut am_routine = unsafe {
        pgrx::PgBox::<pgrx::pg_sys::IndexAmRoutine>::alloc_node(
            pgrx::pg_sys::NodeTag_T_IndexAmRoutine,
        )
    };

    am_routine.amstrategies = 1;
    am_routine.amsupport = 1;
    am_routine.amoptsprocnum = 0;

    am_routine.amcanorder = false;
    am_routine.amcanorderbyop = true;
    am_routine.amcanbackward = false;
    am_routine.amcanunique = false;
    am_routine.amcanmulticol = false;
    am_routine.amoptionalkey = true;
    am_routine.amsearcharray = false;
    am_routine.amsearchnulls = false;
    am_routine.amstorage = false;
    am_routine.amclusterable = false;
    am_routine.ampredlocks = false;
    am_routine.amcaninclude = false;
    am_routine.amusemaintenanceworkmem = false;
    am_routine.amkeytype = pgrx::pg_sys::InvalidOid;

    am_routine.amvalidate = Some(amvalidate);
    am_routine.amoptions = Some(amoptions);
    am_routine.amcostestimate = Some(amcostestimate);

    am_routine.ambuild = Some(ambuild);
    am_routine.ambuildempty = Some(ambuildempty);
    am_routine.aminsert = Some(aminsert);

    am_routine.ambeginscan = Some(ambeginscan);
    am_routine.amrescan = Some(amrescan);
    am_routine.amgettuple = Some(amgettuple);
    am_routine.amendscan = Some(amendscan);

    am_routine.ambulkdelete = Some(ambulkdelete);
    am_routine.amvacuumcleanup = Some(amvacuumcleanup);

    am_routine.into_pg_boxed()
}

#[pg_guard]
extern "C" fn amvalidate(_op_class_oid: pg_sys::Oid) -> bool {
    true
}

#[pg_guard]
extern "C" fn amoptions(_rel_options: pg_sys::Datum, _validate: bool) -> *mut pg_sys::bytea {
    unimplemented!()
}

#[pg_guard]
unsafe extern "C" fn amcostestimate(
    _root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    if (*path).indexorderbys.is_null() {
        *index_startup_cost = f64::MAX;
        *index_total_cost = f64::MAX;
        *index_selectivity = 0.0;
        *index_correlation = 0.0;
        *index_pages = 0.0;
        return;
    }
    *index_startup_cost = 0.0;
    *index_total_cost = 0.0;
    *index_selectivity = 1.0;
    *index_correlation = 1.0;
    *index_pages = 0.0;
}

#[pg_guard]
unsafe extern "C" fn ambuild(
    heap_relation: pg_sys::Relation,
    index_relation: pg_sys::Relation,
    _index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let typmod = VectorTypmod::parse_from_i32(
        (*(*(*index_relation).rd_att).attrs.as_ptr().add(0)).type_mod(),
    )
    .unwrap();
    let dims = typmod
        .dimensions()
        .expect("Column does not have dimensions.");
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<Scalar>, HeapPointer)>(1024);
    let thread = std::thread::spawn({
        let id = (*index_relation).rd_id;
        let options = Options {
            common: CommonOptions { dims },
            algo: "".to_string(),
        };
        move || {
            build(id, options, rx.into_iter());
        }
    });
    struct BuildState {
        sender: Option<SyncSender<(Vec<Scalar>, HeapPointer)>>,
    }
    let mut state = BuildState { sender: Some(tx) };
    #[pgrx::pg_guard]
    unsafe extern "C" fn callback(
        _index_relation: pgrx::pg_sys::Relation,
        ctid: pgrx::pg_sys::ItemPointer,
        values: *mut pgrx::pg_sys::Datum,
        is_null: *mut bool,
        _tuple_is_alive: bool,
        state: *mut std::os::raw::c_void,
    ) {
        let pgvector = <&Vector>::from_datum(*values.add(0), *is_null.add(0)).unwrap();
        (&mut *(state as *mut BuildState))
            .sender
            .as_mut()
            .unwrap()
            .send((pgvector.to_vec(), HeapPointer::from_sys(*ctid)))
            .unwrap();
    }
    let index_info = pgrx::pg_sys::BuildIndexInfo(index_relation);
    let ntuples = (*(*heap_relation).rd_tableam)
        .index_build_range_scan
        .unwrap()(
        heap_relation,
        index_relation,
        index_info,
        true,
        false,
        true,
        0,
        pgrx::pg_sys::InvalidBlockNumber,
        Some(callback),
        (&mut state) as *mut _ as _,
        null_mut(),
    );
    state.sender = None;
    thread.join().unwrap();
    let mut result = PgBox::<pg_sys::IndexBuildResult>::alloc0();
    result.heap_tuples = ntuples;
    result.index_tuples = 0.0;
    result.into_pg()
}

#[pg_guard]
unsafe extern "C" fn ambuildempty(index_relation: pg_sys::Relation) {
    let typmod = VectorTypmod::parse_from_i32(
        (*(*(*index_relation).rd_att).attrs.as_ptr().add(0)).type_mod(),
    )
    .unwrap();
    let dims = typmod
        .dimensions()
        .expect("Column does not have dimensions.");
    let options = Options {
        common: CommonOptions { dims },
        algo: "".to_string(),
    };
    build((*index_relation).rd_id, options, std::iter::empty());
}

#[pg_guard]
unsafe extern "C" fn aminsert(
    index_relation: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    is_null: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap_relation: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    let pgvector = <&Vector>::from_datum(*values.add(0), *is_null.add(0)).unwrap();
    let vector = pgvector.data();
    let p = HeapPointer::from_sys(*heap_tid);
    insert((*index_relation).rd_id, (vector.to_vec(), p));
    true
}

#[pg_guard]
unsafe extern "C" fn ambeginscan(
    index_relation: pg_sys::Relation,
    n_keys: std::os::raw::c_int,
    n_order_bys: std::os::raw::c_int,
) -> pg_sys::IndexScanDesc {
    let mut scan: PgBox<pg_sys::IndexScanDescData> = PgBox::from_pg(pg_sys::RelationGetIndexScan(
        index_relation,
        n_keys,
        n_order_bys,
    ));

    let state = Option::<ScanState>::None;

    scan.opaque = pgrx::PgMemoryContexts::CurrentMemoryContext.leak_and_drop_on_delete(state)
        as pgrx::void_mut_ptr;

    scan.into_pg()
}

#[pg_guard]
unsafe extern "C" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    keys: pg_sys::ScanKey,
    n_keys: std::os::raw::c_int,
    orderbys: pg_sys::ScanKey,
    n_orderbys: std::os::raw::c_int,
) {
    let oid = (*(*scan).indexRelation).rd_id;
    if n_orderbys > 0 {
        let orderbys = std::slice::from_raw_parts_mut(orderbys, n_orderbys as usize);
        std::ptr::copy(orderbys.as_ptr(), (*scan).orderByData, orderbys.len());
    }
    if n_keys > 0 {
        let keys = std::slice::from_raw_parts_mut(keys, n_keys as usize);
        std::ptr::copy(keys.as_ptr(), (*scan).keyData, keys.len());
    }
    if (*scan).numberOfOrderBys > 0 {
        use pgrx::pg_sys::{palloc, palloc0};
        let size_datum = std::mem::size_of::<Datum>();
        let size_bool = std::mem::size_of::<bool>();
        let orderbyvals = palloc0(size_datum * (*scan).numberOfOrderBys as usize) as *mut Datum;
        let orderbynulls = palloc(size_bool * (*scan).numberOfOrderBys as usize) as *mut bool;
        orderbynulls.write_bytes(1, (*scan).numberOfOrderBys as usize);
        (*scan).xs_orderbyvals = orderbyvals;
        (*scan).xs_orderbynulls = orderbynulls;
    }
    assert!(n_orderbys == 1, "Not supported.");
    let state = &mut *((*scan).opaque as *mut Option<ScanState>);
    let scan_vector = (*orderbys.add(0)).sk_argument;
    let dt_vector = <&Vector>::from_datum(scan_vector, false).unwrap();
    let vector = dt_vector.data();
    *state = Some(ScanState {
        data: {
            let mut data = search(oid, (vector.to_vec(), PROBES));
            data.reverse();
            data
        },
    });
}

#[pg_guard]
unsafe extern "C" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection,
) -> bool {
    (*scan).xs_recheck = false;
    (*scan).xs_recheckorderby = false;
    let state = &mut *((*scan).opaque as *mut Option<ScanState>);
    if let Some(x) = state.as_mut().unwrap().data.pop() {
        (*scan).xs_heaptid = x.into_sys();
        true
    } else {
        false
    }
}

#[pg_guard]
extern "C" fn amendscan(_scan: pg_sys::IndexScanDesc) {}

#[pg_guard]
unsafe extern "C" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    _stats: *mut pg_sys::IndexBulkDeleteResult,
    _callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::os::raw::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let items = callback_state as *mut pgrx::pg_sys::VacDeadItems;
    let deletes = (*items)
        .items
        .as_slice((*items).max_items as usize)
        .iter()
        .copied()
        .map(HeapPointer::from_sys)
        .collect::<Vec<HeapPointer>>();
    bulkdelete((*(*info).index).rd_id, deletes);
    let result = pgrx::PgBox::<pgrx::pg_sys::IndexBulkDeleteResult>::alloc0();
    result.into_pg()
}

#[pg_guard]
unsafe extern "C" fn amvacuumcleanup(
    _info: *mut pg_sys::IndexVacuumInfo,
    _stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let result = pgrx::PgBox::<pgrx::pg_sys::IndexBulkDeleteResult>::alloc0();
    result.into_pg()
}
