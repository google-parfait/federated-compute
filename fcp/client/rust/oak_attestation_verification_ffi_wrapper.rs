#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use oak_proto_rust::oak::attestation::v1::{
    AttestationResults, Endorsements, Evidence, ReferenceValues,
};
use prost::Message;

/// Represents a boxed Rust slice containing a serialized [AttestationResults]
/// proto held in a contiguous buffer owned by this struct, to be returned over
/// the FFI boundary.
///
/// Every instance of this struct returned over an FFI boundary should be
/// released using the
/// `fcp_rs_oak_attestation_verification_free_attestation_result`.
#[repr(C)]
pub struct SerializedAttestationResults {
    /// A pointer to the serialized data (if `size > 0`), or a null pointer (if
    /// `size == 0`).
    pub data: *mut u8,
    /// The size of the buffer.
    pub size: usize,
}

#[no_mangle]
pub unsafe extern "C" fn fcp_rs_oak_attestation_verification_verify_attestation<'a>(
    now_utc_millis: i64,
    serialized_evidence: *const u8,
    serialized_evidence_size: usize,
    serialized_endorsements: *const u8,
    serialized_endorsements_size: usize,
    serialized_reference_values: *const u8,
    serialized_reference_values_size: usize,
) -> SerializedAttestationResults {
    let evidence = Evidence::decode(core::slice::from_raw_parts(
        serialized_evidence,
        serialized_evidence_size,
    ))
    .unwrap();
    let endorsements = Endorsements::decode(core::slice::from_raw_parts(
        serialized_endorsements,
        serialized_endorsements_size,
    ))
    .unwrap();
    let reference_values = ReferenceValues::decode(core::slice::from_raw_parts(
        serialized_reference_values,
        serialized_reference_values_size,
    ))
    .unwrap();

    let result: AttestationResults = oak_attestation_verification::verifier::to_attestation_results(
        &oak_attestation_verification::verifier::verify(
            now_utc_millis,
            &evidence,
            &endorsements,
            &reference_values,
        ),
    );

    let serialized_result: Box<[u8]> = result.encode_to_vec().into_boxed_slice();
    // If the serialized result is an empty slice, then just return a null buffer
    // and 0 size. This ensures that we'll always be returning a valid pointer
    // pointing to a non-empty buffer below.
    if serialized_result.is_empty() {
        return SerializedAttestationResults { data: core::ptr::null_mut(), size: 0 };
    }

    let static_serialized_result: &'static mut [u8] = Box::<[u8]>::leak(serialized_result);
    SerializedAttestationResults {
        data: static_serialized_result.as_mut_ptr(),
        size: static_serialized_result.len(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn fcp_rs_oak_attestation_verification_free_attestation_result<'a>(
    result: SerializedAttestationResults,
) {
    // Turn the pointer & size back into a boxed slice and drop it, to ensure the
    // memory is released, but only if the slice is non-empty (since we
    // only return a valid pointer if there was any data to return in the first
    // place).
    if result.size > 0 {
        assert!(!result.data.is_null());
        drop(Box::from_raw(core::slice::from_raw_parts_mut(result.data, result.size)))
    }
}
