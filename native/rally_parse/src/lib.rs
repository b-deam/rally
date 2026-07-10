// Licensed to Elasticsearch B.V. under one or more contributor
// license agreements. See the NOTICE file distributed with
// this work for additional information regarding copyright
// ownership. Elasticsearch B.V. licenses this file to you under
// the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Prototype native replacement for `esrally.driver.runner.parse`.
//!
//! Implements the *hot* subset of the Python contract: extracting scalar
//! `props` (dotted paths) and determining whether the arrays named in `lists`
//! are empty. It is a **streaming, early-exit** scanner: it stops as soon as all
//! requested props/lists have been found and skips subtrees it does not need
//! (e.g. a large `hits.hits` array), mirroring the early-termination behaviour
//! that makes the ijson implementation viable on large responses.
//!
//! The rarer `objects` / `stop_after` / `with_cluster_details` features are left
//! to the pure-Python fallback so behaviour is preserved.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde_json::Value;

#[inline]
fn is_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

#[inline]
fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && is_ws(b[i]) {
        i += 1;
    }
    i
}

/// `b[i]` must be the opening quote. Returns the index just past the closing quote.
fn skip_string(b: &[u8], i: usize) -> Option<usize> {
    let mut i = i + 1;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2, // skip escaped char (incl. \uXXXX: the digits are harmless)
            b'"' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

/// Skip any JSON value starting at `i` (leading whitespace tolerated).
/// Returns the index just past the value.
fn skip_value(b: &[u8], i: usize) -> Option<usize> {
    let mut i = skip_ws(b, i);
    match *b.get(i)? {
        b'"' => skip_string(b, i),
        b'{' | b'[' => {
            let mut depth = 0usize;
            while i < b.len() {
                match b[i] {
                    b'"' => {
                        i = skip_string(b, i)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            None
        }
        _ => {
            while i < b.len() && !matches!(b[i], b',' | b'}' | b']') && !is_ws(b[i]) {
                i += 1;
            }
            Some(i)
        }
    }
}

/// True if `prefix` is a strict ancestor of some target path (i.e. some target
/// is `"{prefix}.something"`), meaning we must descend into it.
fn is_ancestor(prefix: &str, targets: &[&str]) -> bool {
    let pb = prefix.as_bytes();
    targets.iter().any(|t| {
        let tb = t.as_bytes();
        tb.len() > pb.len() && &tb[..pb.len()] == pb && tb[pb.len()] == b'.'
    })
}

fn scalar_to_py(py: Python<'_>, v: &Value) -> Option<PyObject> {
    match v {
        Value::Null => Some(py.None()),
        Value::Bool(x) => Some(x.into_py(py)),
        Value::Number(n) => {
            if let Some(x) = n.as_i64() {
                Some(x.into_py(py))
            } else if let Some(x) = n.as_u64() {
                Some(x.into_py(py))
            } else {
                Some(n.as_f64().unwrap_or(f64::NAN).into_py(py))
            }
        }
        Value::String(s) => Some(s.into_py(py)),
        Value::Array(_) | Value::Object(_) => None,
    }
}

struct Scanner<'a> {
    b: &'a [u8],
    props: &'a [&'a str],
    lists: &'a [&'a str],
    found_props: usize,
    found_lists: usize,
}

impl<'a> Scanner<'a> {
    fn done(&self) -> bool {
        self.found_props == self.props.len() && self.found_lists == self.lists.len()
    }

    /// Walk an object whose `{` is at `b[i]`. Captures matching scalar props and
    /// list emptiness into `out`. Returns index just past the closing `}`.
    fn walk_object(&mut self, py: Python<'_>, out: &Bound<'_, PyDict>, i: usize, prefix: &str) -> Option<usize> {
        let b = self.b;
        let mut i = i + 1; // past '{'
        loop {
            if self.done() {
                return Some(i);
            }
            i = skip_ws(b, i);
            match *b.get(i)? {
                b'}' => return Some(i + 1),
                b',' => {
                    i += 1;
                    continue;
                }
                b'"' => {}
                _ => return None,
            }
            // key
            let key_end = skip_string(b, i)?;
            let key: String = serde_json::from_slice(&b[i..key_end]).ok()?;
            i = skip_ws(b, key_end);
            if *b.get(i)? != b':' {
                return None;
            }
            i = skip_ws(b, i + 1);

            let child = if prefix.is_empty() {
                key
            } else {
                format!("{prefix}.{key}")
            };

            match *b.get(i)? {
                b'{' => {
                    if is_ancestor(&child, self.props) || is_ancestor(&child, self.lists) {
                        i = self.walk_object(py, out, i, &child)?;
                    } else {
                        i = skip_value(b, i)?;
                    }
                }
                b'[' => {
                    if self.lists.contains(&child.as_str()) {
                        let j = skip_ws(b, i + 1);
                        let empty = *b.get(j)? == b']';
                        out.set_item(&child, empty).ok()?;
                        self.found_lists += 1;
                        // Everything requested is found: bail out *before* scanning the
                        // (potentially huge) array body. This is what keeps the native
                        // parser O(bytes-before-last-field) rather than O(response-size).
                        if self.done() {
                            return Some(i);
                        }
                    }
                    i = skip_value(b, i)?;
                }
                _ => {
                    let val_start = i;
                    i = skip_value(b, i)?;
                    if self.props.contains(&child.as_str()) {
                        if let Ok(v) = serde_json::from_slice::<Value>(&b[val_start..i]) {
                            if let Some(obj) = scalar_to_py(py, &v) {
                                out.set_item(&child, obj).ok()?;
                                self.found_props += 1;
                                if self.done() {
                                    return Some(i);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Native equivalent of the supported subset of `runner.parse`.
#[pyfunction]
#[pyo3(signature = (data, props, lists=None))]
fn parse(py: Python<'_>, data: &[u8], props: Vec<String>, lists: Option<Vec<String>>) -> PyResult<Py<PyDict>> {
    let out = PyDict::new_bound(py);

    let prop_refs: Vec<&str> = props.iter().map(|s| s.as_str()).collect();
    let list_vec = lists.unwrap_or_default();
    let list_refs: Vec<&str> = list_vec.iter().map(|s| s.as_str()).collect();

    let mut scanner = Scanner {
        b: data,
        props: &prop_refs,
        lists: &list_refs,
        found_props: 0,
        found_lists: 0,
    };

    let start = skip_ws(data, 0);
    if data.get(start) == Some(&b'{') {
        // Ignore parse errors: like the Python impl, partial results are fine.
        let _ = scanner.walk_object(py, &out, start, "");
    }

    Ok(out.unbind())
}

// ---------------------------------------------------------------------------
// Native equivalent of BulkIndex.detailed_stats (the hotspot identified by the
// driver profile: `_utf8len` + the per-item accounting loop + client json.loads).
// ---------------------------------------------------------------------------

struct OpAgg {
    name: String,
    item_count: u64,
    /// (result value -> count), insertion-ordered to match the Python Counter.
    results: Vec<(String, u64)>,
}

struct ShardAgg {
    key: String,
    item_count: u64,
    total: i64,
    successful: i64,
    failed: i64,
}

fn error_status_summary(error_details: &[(i64, Option<String>)]) -> String {
    let mut counts: Vec<(i64, u64)> = Vec::new();
    for (status, _) in error_details {
        if let Some(c) = counts.iter_mut().find(|(s, _)| s == status) {
            c.1 += 1;
        } else {
            counts.push((*status, 1));
        }
    }
    counts.sort_by_key(|(s, _)| *s);
    counts
        .iter()
        .map(|(s, c)| format!("{c}x{s}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn build_error_description(error_details: &mut Vec<(i64, Option<String>)>) -> String {
    error_details.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.as_deref().unwrap_or("").cmp(b.1.as_deref().unwrap_or("")))
    });
    let mut descs = Vec::new();
    let mut truncated = false;
    for (count, (status, reason)) in error_details.iter().enumerate() {
        if count < 5 {
            match reason {
                Some(r) => descs.push(format!("HTTP status: {status}, message: {r}")),
                None => descs.push(format!("HTTP status: {status}")),
            }
        } else {
            truncated = true;
            break;
        }
    }
    let mut description = descs.join(" | ");
    if truncated {
        description.push_str(" | TRUNCATED ");
        description.push_str(&error_status_summary(error_details));
    }
    description
}

/// Parse a JSON integer at/after `i`. Returns `(value, index just past it)`.
fn parse_i64(b: &[u8], i: usize) -> Option<(i64, usize)> {
    let start = skip_ws(b, i);
    let mut j = start;
    if b.get(j) == Some(&b'-') {
        j += 1;
    }
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    if j == start {
        return None;
    }
    let s = std::str::from_utf8(&b[start..j]).ok()?;
    Some((s.parse().ok()?, j))
}

/// `b[i]` (after ws) must be a JSON string. Returns `(raw content bytes between the
/// quotes, index just past the closing quote)`. The content is returned *verbatim*
/// (escapes are not decoded); this is exact for the values we read (op names,
/// `result`) which never contain escapes, and best-effort for `error.reason`.
fn scan_str(b: &[u8], i: usize) -> Option<(&[u8], usize)> {
    let i = skip_ws(b, i);
    if *b.get(i)? != b'"' {
        return None;
    }
    let end = skip_string(b, i)?;
    Some((&b[i + 1..end - 1], end))
}

/// Expect the next non-ws byte to be `c`; return the index just past it.
fn expect(b: &[u8], i: usize, c: u8) -> Option<usize> {
    let i = skip_ws(b, i);
    if *b.get(i)? == c {
        Some(i + 1)
    } else {
        None
    }
}

#[derive(Default)]
struct BulkAgg {
    ops: Vec<OpAgg>,
    shards: Vec<ShardAgg>,
    bulk_error_count: u64,
    bulk_success_count: u64,
    max_doc_status: i64,
    error_429: Vec<i64>,
    error_details: Vec<(i64, Option<String>)>,
    took: Option<i64>,
    ingest_took: Option<i64>,
}

fn get_or_add_op(agg: &mut BulkAgg, op_name: &[u8]) -> usize {
    if let Some(idx) = agg.ops.iter().position(|o| o.name.as_bytes() == op_name) {
        idx
    } else {
        agg.ops.push(OpAgg {
            name: String::from_utf8_lossy(op_name).into_owned(),
            item_count: 0,
            results: Vec::new(),
        });
        agg.ops.len() - 1
    }
}

/// Scan a `_shards` object, returning `(total, successful, failed, next_index)`.
fn scan_shards(b: &[u8], i: usize) -> Option<(i64, i64, i64, usize)> {
    let mut i = expect(b, i, b'{')?;
    let (mut total, mut successful, mut failed) = (0i64, 0i64, 0i64);
    loop {
        i = skip_ws(b, i);
        match *b.get(i)? {
            b'}' => {
                i += 1;
                break;
            }
            b',' => {
                i += 1;
                continue;
            }
            b'"' => {}
            _ => return None,
        }
        let (key, ni) = scan_str(b, i)?;
        let ci = expect(b, ni, b':')?;
        match key {
            b"total" => {
                let (v, n) = parse_i64(b, ci)?;
                total = v;
                i = n;
            }
            b"successful" => {
                let (v, n) = parse_i64(b, ci)?;
                successful = v;
                i = n;
            }
            b"failed" => {
                let (v, n) = parse_i64(b, ci)?;
                failed = v;
                i = n;
            }
            _ => i = skip_value(b, ci)?,
        }
    }
    Some((total, successful, failed, i))
}

/// Find a top-level `"reason"` string inside the error object whose `{` is at `b[i]`.
fn find_reason(b: &[u8], i: usize) -> Option<String> {
    let mut i = i + 1; // past '{'
    loop {
        i = skip_ws(b, i);
        match *b.get(i)? {
            b'}' => return None,
            b',' => {
                i += 1;
                continue;
            }
            b'"' => {}
            _ => return None,
        }
        let (key, ni) = scan_str(b, i)?;
        let ci = expect(b, ni, b':')?;
        if key == b"reason" {
            // Cold error path: decode the reason string exactly (JSON escapes,
            // surrogate pairs) via serde so it matches Python's json.loads output.
            let s = skip_ws(b, ci);
            if *b.get(s)? == b'"' {
                let end = skip_string(b, s)?;
                if let Ok(decoded) = serde_json::from_slice::<String>(&b[s..end]) {
                    return Some(decoded);
                }
            }
            return None;
        }
        i = skip_value(b, ci)?;
    }
}

/// Scan one bulk item's data object (its `{` follows `i` after ws) and fold it into
/// `agg`, mirroring the Python per-item accounting. `idx` is the item's position.
fn scan_item_data(b: &[u8], i: usize, agg: &mut BulkAgg, op_name: &[u8], idx: i64) -> Option<usize> {
    let mut i = expect(b, i, b'{')?;
    let op_idx = get_or_add_op(agg, op_name);
    agg.ops[op_idx].item_count += 1;

    let mut status: i64 = 0;
    let mut failed_shards: i64 = 0;
    // None => no "error" key; Some(reason) => "error" present (reason may be None).
    let mut error_reason: Option<Option<String>> = None;

    loop {
        i = skip_ws(b, i);
        match *b.get(i)? {
            b'}' => {
                i += 1;
                break;
            }
            b',' => {
                i += 1;
                continue;
            }
            b'"' => {}
            _ => return None,
        }
        let (key, ni) = scan_str(b, i)?;
        let ci = expect(b, ni, b':')?;
        match key {
            b"result" => {
                let (val, n) = scan_str(b, ci)?;
                i = n;
                let results = &mut agg.ops[op_idx].results;
                if let Some(r) = results.iter_mut().find(|(k, _)| k.as_bytes() == val) {
                    r.1 += 1;
                } else {
                    results.push((String::from_utf8_lossy(val).into_owned(), 1));
                }
            }
            b"status" => {
                let (v, n) = parse_i64(b, ci)?;
                status = v;
                i = n;
            }
            b"_shards" => {
                let (t, s, fl, n) = scan_shards(b, ci)?;
                failed_shards = fl;
                let key = format!("{t}-{s}-{fl}");
                if let Some(se) = agg.shards.iter_mut().find(|x| x.key == key) {
                    se.item_count += 1;
                } else {
                    agg.shards.push(ShardAgg {
                        key,
                        item_count: 1,
                        total: t,
                        successful: s,
                        failed: fl,
                    });
                }
                i = n;
            }
            b"error" => {
                let j = skip_ws(b, ci);
                if *b.get(j)? == b'{' {
                    error_reason = Some(find_reason(b, j));
                    i = skip_value(b, j)?;
                } else {
                    error_reason = Some(None);
                    i = skip_value(b, j)?;
                }
            }
            _ => i = skip_value(b, ci)?,
        }
    }

    if status > 299 || failed_shards > 0 {
        agg.bulk_error_count += 1;
        if status > agg.max_doc_status {
            agg.max_doc_status = status;
        }
        let reason = error_reason.unwrap_or(None);
        let detail = (status, reason);
        if !agg.error_details.contains(&detail) {
            agg.error_details.push(detail);
        }
        if status == 429 {
            agg.error_429.push(idx);
        }
    } else {
        agg.bulk_success_count += 1;
    }
    Some(i)
}

/// Scan the `items` array (its `[` follows `i` after ws), folding each item into `agg`.
fn scan_items(b: &[u8], i: usize, agg: &mut BulkAgg) -> Option<usize> {
    let mut i = expect(b, i, b'[')?;
    let mut idx: i64 = 0;
    loop {
        i = skip_ws(b, i);
        match *b.get(i)? {
            b']' => return Some(i + 1),
            b',' => {
                i += 1;
                continue;
            }
            b'{' => {}
            _ => return None,
        }
        i += 1; // past item '{'
        let (op_name, ni) = scan_str(b, i)?; // first key = op name
        i = scan_item_data(b, expect(b, ni, b':')?, agg, op_name, idx)?;
        // Skip any trailing keys in the item object (ES items carry exactly one).
        loop {
            i = skip_ws(b, i);
            match *b.get(i)? {
                b'}' => {
                    i += 1;
                    break;
                }
                b',' => {
                    let (_k, nk) = scan_str(b, i + 1)?;
                    i = skip_value(b, expect(b, nk, b':')?)?;
                }
                _ => return None,
            }
        }
        idx += 1;
    }
}

/// Streaming byte-scan of a `_bulk` response: extracts `took`/`ingest_took` and the
/// per-item aggregates without allocating a serde_json DOM for the whole response.
fn scan_bulk_response(b: &[u8]) -> Option<BulkAgg> {
    let mut agg = BulkAgg {
        max_doc_status: -1,
        ..Default::default()
    };
    let mut i = expect(b, 0, b'{')?;
    loop {
        i = skip_ws(b, i);
        match *b.get(i)? {
            b'}' => return Some(agg),
            b',' => {
                i += 1;
                continue;
            }
            b'"' => {}
            _ => return None,
        }
        let (key, ni) = scan_str(b, i)?;
        let ci = expect(b, ni, b':')?;
        match key {
            b"took" => {
                let (v, n) = parse_i64(b, ci)?;
                agg.took = Some(v);
                i = n;
            }
            b"ingest_took" => {
                let (v, n) = parse_i64(b, ci)?;
                agg.ingest_took = Some(v);
                i = n;
            }
            b"items" => i = scan_items(b, ci, &mut agg)?,
            _ => i = skip_value(b, ci)?,
        }
    }
}

/// Compute the detailed per-bulk statistics natively from the raw response body
/// and the raw request body. Returns a dict of the fields needed to build a
/// Python `BulkStats` (everything except `request_status`, which the caller
/// supplies from `response.meta.status`).
#[pyfunction]
#[pyo3(signature = (response, body, with_action_metadata))]
fn bulk_detailed_stats(py: Python<'_>, response: &[u8], body: &[u8], with_action_metadata: bool) -> PyResult<Py<PyDict>> {
    // 1) Request/document byte sizes (replaces the `_utf8len` loop).
    let mut bulk_request_size_bytes: u64 = 0;
    let mut total_document_size_bytes: u64 = 0;
    for (line_number, seg) in body.split(|&c| c == b'\n').enumerate() {
        let line_size = seg.len() as u64;
        if with_action_metadata {
            if line_number % 2 == 1 {
                total_document_size_bytes += line_size;
            }
        } else {
            total_document_size_bytes += line_size;
        }
        bulk_request_size_bytes += line_size;
    }

    // 2) Streaming byte-scan of the response + per-item accounting (replaces
    //    json.loads + the Python `for item in response["items"]` loop). Avoids
    //    allocating a serde_json DOM for the whole (multi-MB) response.
    let agg = scan_bulk_response(response).ok_or_else(|| PyValueError::new_err("invalid bulk response JSON"))?;
    let took = agg.took;
    let ingest_took = agg.ingest_took;
    let ops = agg.ops;
    let shards = agg.shards;
    let bulk_error_count = agg.bulk_error_count;
    let bulk_success_count = agg.bulk_success_count;
    let max_doc_status = agg.max_doc_status;
    let error_429 = agg.error_429;
    let mut error_details = agg.error_details;

    let error_description = if bulk_error_count > 0 {
        Some(build_error_description(&mut error_details))
    } else {
        None
    };

    let out = PyDict::new_bound(py);
    out.set_item("success_count", bulk_success_count)?;
    out.set_item("error_count", bulk_error_count)?;
    out.set_item("took", took)?;
    out.set_item("max_doc_status", max_doc_status)?;
    out.set_item("error_description", error_description)?;
    out.set_item("error_429_indices", error_429)?;

    let ops_dict = PyDict::new_bound(py);
    for op in &ops {
        let d = PyDict::new_bound(py);
        d.set_item("item-count", op.item_count)?;
        for (k, v) in &op.results {
            d.set_item(k, v)?;
        }
        ops_dict.set_item(&op.name, d)?;
    }
    out.set_item("ops", ops_dict)?;

    let hist = PyList::empty_bound(py);
    for s in &shards {
        let entry = PyDict::new_bound(py);
        entry.set_item("item-count", s.item_count)?;
        let sh = PyDict::new_bound(py);
        sh.set_item("total", s.total)?;
        sh.set_item("successful", s.successful)?;
        sh.set_item("failed", s.failed)?;
        entry.set_item("shards", sh)?;
        hist.append(entry)?;
    }
    out.set_item("shards_histogram", hist)?;

    out.set_item("bulk_request_size_bytes", bulk_request_size_bytes)?;
    out.set_item("total_document_size_bytes", total_document_size_bytes)?;
    out.set_item("ingest_took", ingest_took)?;

    Ok(out.unbind())
}

// ---------------------------------------------------------------------------
// Native equivalent of the constant-metadata bulk read + assembly path
// (MmapSource.readlines + MetadataIndexDataReader._read_bulk_fast), the next
// hotspot after detailed_stats. Reads up to `num_docs` newline-terminated lines
// directly from a memory-mapped buffer (zero-copy) and returns the assembled
// bulk request body (metadata line prepended to each doc), avoiding the creation
// of one Python bytes object per line.
// ---------------------------------------------------------------------------

/// Returns (docs_read, new_offset, body) where `body` is
/// `metadata + line` concatenated for each line read.
#[pyfunction]
#[pyo3(signature = (buffer, start, num_docs, metadata))]
fn assemble_bulk(
    py: Python<'_>,
    buffer: PyBuffer<u8>,
    start: usize,
    num_docs: usize,
    metadata: &[u8],
) -> PyResult<(usize, usize, Py<PyBytes>)> {
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err("buffer is not C-contiguous"));
    }
    let len = buffer.len_bytes();
    // SAFETY: the source is a read-only mmap that outlives this call and is not
    // mutated concurrently (each client owns its own file source). We only read.
    let data: &[u8] = unsafe { std::slice::from_raw_parts(buffer.buf_ptr() as *const u8, len) };

    let start = start.min(len);
    let mut pos = start;
    let mut docs = 0usize;
    let mut out: Vec<u8> = Vec::with_capacity(num_docs.saturating_mul(metadata.len() + 64));

    while docs < num_docs && pos < len {
        let line_end = match memchr::memchr(b'\n', &data[pos..]) {
            Some(i) => pos + i + 1, // include the trailing newline, matching mmap.readline()
            None => len,
        };
        out.extend_from_slice(metadata);
        out.extend_from_slice(&data[pos..line_end]);
        pos = line_end;
        docs += 1;
    }

    Ok((docs, pos, PyBytes::new_bound(py, &out).unbind()))
}

// ---------------------------------------------------------------------------
// Native equivalent of the general (non-constant metadata) bulk read + assembly
// path (MetadataIndexDataReader._read_bulk_regular). Unlike the fast path the
// action/meta-data line varies per document, so the caller supplies the already
// generated per-doc metadata (Python owns the id/conflict RNG). This reads the
// documents directly from the memory-mapped buffer and interleaves them with the
// supplied metadata, wrapping `update` actions in `{"doc": ...}` exactly like the
// Python implementation, while avoiding one Python bytes object per line.
// ---------------------------------------------------------------------------

/// Strip leading/trailing ASCII whitespace, matching Python `bytes.strip()`
/// (the set is `\t\n\x0b\x0c\r ` and space).
fn strip_ascii_ws(mut s: &[u8]) -> &[u8] {
    while let Some((&c, rest)) = s.split_first() {
        if matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
            s = rest;
        } else {
            break;
        }
    }
    while let Some((&c, rest)) = s.split_last() {
        if matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
            s = rest;
        } else {
            break;
        }
    }
    s
}

/// Returns (docs_read, new_offset, body). `metadata` has one entry per document
/// to read: `None` appends the doc unchanged; `Some((is_update, meta))` prepends
/// `meta` and, when `is_update`, replaces the doc with `{"doc":<stripped>}\n`.
#[pyfunction]
#[pyo3(signature = (buffer, start, metadata))]
fn assemble_bulk_regular(
    py: Python<'_>,
    buffer: PyBuffer<u8>,
    start: usize,
    metadata: Vec<Option<(bool, Vec<u8>)>>,
) -> PyResult<(usize, usize, Py<PyBytes>)> {
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err("buffer is not C-contiguous"));
    }
    let len = buffer.len_bytes();
    // SAFETY: the source is a read-only mmap that outlives this call and is not
    // mutated concurrently (each client owns its own file source). We only read.
    let data: &[u8] = unsafe { std::slice::from_raw_parts(buffer.buf_ptr() as *const u8, len) };

    let start = start.min(len);
    let mut pos = start;
    let mut docs = 0usize;
    let mut out: Vec<u8> = Vec::with_capacity(metadata.len().saturating_mul(96));

    for meta in metadata.iter() {
        if pos >= len {
            break;
        }
        let line_end = match memchr::memchr(b'\n', &data[pos..]) {
            Some(i) => pos + i + 1, // include the trailing newline, matching mmap.readline()
            None => len,
        };
        let line = &data[pos..line_end];
        match meta {
            None => out.extend_from_slice(line),
            Some((is_update, meta_bytes)) => {
                out.extend_from_slice(meta_bytes);
                if *is_update {
                    out.extend_from_slice(b"{\"doc\":");
                    out.extend_from_slice(strip_ascii_ws(line));
                    out.extend_from_slice(b"}\n");
                } else {
                    out.extend_from_slice(line);
                }
            }
        }
        pos = line_end;
        docs += 1;
    }

    Ok((docs, pos, PyBytes::new_bound(py, &out).unbind()))
}

#[pymodule]
fn rally_parse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(bulk_detailed_stats, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_bulk, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_bulk_regular, m)?)?;
    Ok(())
}
