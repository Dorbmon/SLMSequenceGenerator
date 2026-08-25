#![no_std]

use core::panic::PanicInfo;
use core::slice;

const ASSIGNMENT_INFINITY: f64 = 1.0e30;
const CORE_ABI_VERSION: u32 = 2;

#[link(wasm_import_module = "env")]
extern "C" {
    fn sin(value: f64) -> f64;
    fn cos(value: f64) -> f64;
    fn atan2(y: f64, x: f64) -> f64;
}

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}

#[no_mangle]
pub extern "C" fn core_abi_version() -> u32 {
    CORE_ABI_VERSION
}

/// Transform one packed complex vector in place.
///
/// `scratch_real` and `scratch_imag` must each have room for `length` f64
/// values. They are only read for a non-power-of-two DFT.
///
/// # Safety
///
/// All pointers must be aligned, non-null, non-overlapping regions containing
/// at least `length` f64 values in this module's linear memory.
#[no_mangle]
pub unsafe extern "C" fn core_fft_1d(
    real_pointer: *mut f64,
    imag_pointer: *mut f64,
    scratch_real_pointer: *mut f64,
    scratch_imag_pointer: *mut f64,
    length: u32,
    inverse: u32,
) -> i32 {
    let length = length as usize;
    if length == 0 {
        return 0;
    }
    if real_pointer.is_null()
        || imag_pointer.is_null()
        || scratch_real_pointer.is_null()
        || scratch_imag_pointer.is_null()
    {
        return -1;
    }

    let real = slice::from_raw_parts_mut(real_pointer, length);
    let imag = slice::from_raw_parts_mut(imag_pointer, length);
    let scratch_real = slice::from_raw_parts_mut(scratch_real_pointer, length);
    let scratch_imag = slice::from_raw_parts_mut(scratch_imag_pointer, length);
    fft_1d(real, imag, scratch_real, scratch_imag, inverse != 0);
    0
}

/// Transform a packed row-major complex field in place. All four work buffers
/// must contain at least `max(width, height)` f64 values.
///
/// # Safety
///
/// The field pointers must be aligned, non-null, non-overlapping regions of
/// `width * height` f64 values. The four work pointers must likewise be
/// non-overlapping regions of at least `max(width, height)` f64 values.
#[no_mangle]
pub unsafe extern "C" fn core_fft_2d(
    real_pointer: *mut f64,
    imag_pointer: *mut f64,
    width: u32,
    height: u32,
    work_real_pointer: *mut f64,
    work_imag_pointer: *mut f64,
    scratch_real_pointer: *mut f64,
    scratch_imag_pointer: *mut f64,
    inverse: u32,
) -> i32 {
    let width = width as usize;
    let height = height as usize;
    let Some(length) = width.checked_mul(height) else {
        return -1;
    };
    if width == 0 || height == 0 {
        return -1;
    }
    if real_pointer.is_null()
        || imag_pointer.is_null()
        || work_real_pointer.is_null()
        || work_imag_pointer.is_null()
        || scratch_real_pointer.is_null()
        || scratch_imag_pointer.is_null()
    {
        return -1;
    }

    let work_length = if width > height { width } else { height };
    let real = slice::from_raw_parts_mut(real_pointer, length);
    let imag = slice::from_raw_parts_mut(imag_pointer, length);
    let work_real = slice::from_raw_parts_mut(work_real_pointer, work_length);
    let work_imag = slice::from_raw_parts_mut(work_imag_pointer, work_length);
    let scratch_real = slice::from_raw_parts_mut(scratch_real_pointer, work_length);
    let scratch_imag = slice::from_raw_parts_mut(scratch_imag_pointer, work_length);
    let inverse = inverse != 0;

    for y in 0..height {
        let offset = y * width;
        work_real[..width].copy_from_slice(&real[offset..offset + width]);
        work_imag[..width].copy_from_slice(&imag[offset..offset + width]);
        fft_1d(
            &mut work_real[..width],
            &mut work_imag[..width],
            &mut scratch_real[..width],
            &mut scratch_imag[..width],
            inverse,
        );
        real[offset..offset + width].copy_from_slice(&work_real[..width]);
        imag[offset..offset + width].copy_from_slice(&work_imag[..width]);
    }

    for x in 0..width {
        for y in 0..height {
            let index = y * width + x;
            work_real[y] = real[index];
            work_imag[y] = imag[index];
        }
        fft_1d(
            &mut work_real[..height],
            &mut work_imag[..height],
            &mut scratch_real[..height],
            &mut scratch_imag[..height],
            inverse,
        );
        for y in 0..height {
            let index = y * width + x;
            real[index] = work_real[y];
            imag[index] = work_imag[y];
        }
    }
    0
}

/// Evaluate the unnormalised two-dimensional discrete Fourier transform at
/// arbitrary (including fractional) target frequencies.
///
/// The target coordinates are expressed in FFT-bin units. Integer coordinates
/// therefore agree exactly with `core_fft_2d`, while fractional coordinates are
/// evaluated directly instead of interpolating neighbouring FFT bins.
///
/// # Safety
///
/// The field pointers must each address `width * height` f64 values. Target and
/// output pointers must each address `target_count` f64 values. Regions must be
/// aligned and the two output regions must not overlap any input region.
#[no_mangle]
pub unsafe extern "C" fn core_nudft_sample_targets(
    field_real_pointer: *const f64,
    field_imag_pointer: *const f64,
    width: u32,
    height: u32,
    target_x_pointer: *const f64,
    target_y_pointer: *const f64,
    target_count: u32,
    output_real_pointer: *mut f64,
    output_imag_pointer: *mut f64,
) -> i32 {
    let width = width as usize;
    let height = height as usize;
    let target_count = target_count as usize;
    let Some(pixel_count) = width.checked_mul(height) else {
        return -1;
    };
    if width == 0 || height == 0 {
        return -1;
    }
    if target_count == 0 {
        return 0;
    }
    if field_real_pointer.is_null()
        || field_imag_pointer.is_null()
        || target_x_pointer.is_null()
        || target_y_pointer.is_null()
        || output_real_pointer.is_null()
        || output_imag_pointer.is_null()
    {
        return -1;
    }

    let field_real = slice::from_raw_parts(field_real_pointer, pixel_count);
    let field_imag = slice::from_raw_parts(field_imag_pointer, pixel_count);
    let target_x = slice::from_raw_parts(target_x_pointer, target_count);
    let target_y = slice::from_raw_parts(target_y_pointer, target_count);
    let output_real = slice::from_raw_parts_mut(output_real_pointer, target_count);
    let output_imag = slice::from_raw_parts_mut(output_imag_pointer, target_count);
    let tau = 2.0 * core::f64::consts::PI;

    for target in 0..target_count {
        let frequency_x = principal_frequency(target_x[target], width);
        let frequency_y = principal_frequency(target_y[target], height);
        if !frequency_x.is_finite() || !frequency_y.is_finite() {
            return -2;
        }
        let angle_x = -tau * frequency_x / width as f64;
        let angle_y = -tau * frequency_y / height as f64;
        let root_x_real = cos(angle_x);
        let root_x_imag = sin(angle_x);
        let root_y_real = cos(angle_y);
        let root_y_imag = sin(angle_y);
        let mut row_real = 1.0;
        let mut row_imag = 0.0;
        let mut sum_real = 0.0;
        let mut sum_imag = 0.0;

        for y in 0..height {
            let mut factor_real = row_real;
            let mut factor_imag = row_imag;
            let row_offset = y * width;
            for x in 0..width {
                let index = row_offset + x;
                sum_real += field_real[index] * factor_real - field_imag[index] * factor_imag;
                sum_imag += field_real[index] * factor_imag + field_imag[index] * factor_real;
                let next_real = factor_real * root_x_real - factor_imag * root_x_imag;
                factor_imag = factor_real * root_x_imag + factor_imag * root_x_real;
                factor_real = next_real;
            }
            let next_row_real = row_real * root_y_real - row_imag * root_y_imag;
            row_imag = row_real * root_y_imag + row_imag * root_y_real;
            row_real = next_row_real;
        }
        output_real[target] = sum_real;
        output_imag[target] = sum_imag;
    }
    0
}

/// Apply the exact adjoint trap-domain transform and retain only its phase.
/// `target_amplitude` is the desired field amplitude after WGS weighting and
/// `target_phase` is the requested complex-field phase at each target.
///
/// When the coherent sum is below `cancellation_threshold`, the prior phase is
/// retained unless `deterministic_fallback` is non-zero, in which case the
/// seeded deterministic initial phase is used.
///
/// # Safety
///
/// Target pointers must each address `target_count` f64 values. `phase`,
/// `scratch_real`, and `scratch_imag` must each address `width * height` f64
/// values and must not overlap.
#[no_mangle]
pub unsafe extern "C" fn core_nudft_synthesize_phase(
    target_x_pointer: *const f64,
    target_y_pointer: *const f64,
    target_amplitude_pointer: *const f64,
    target_phase_pointer: *const f64,
    target_count: u32,
    width: u32,
    height: u32,
    phase_pointer: *mut f64,
    scratch_real_pointer: *mut f64,
    scratch_imag_pointer: *mut f64,
    cancellation_threshold: f64,
    deterministic_seed: u32,
    deterministic_fallback: u32,
) -> i32 {
    let width = width as usize;
    let height = height as usize;
    let target_count = target_count as usize;
    let Some(pixel_count) = width.checked_mul(height) else {
        return -1;
    };
    if width == 0 || height == 0 || phase_pointer.is_null() {
        return -1;
    }
    if scratch_real_pointer.is_null() || scratch_imag_pointer.is_null() {
        return -1;
    }
    if target_count > 0
        && (target_x_pointer.is_null()
            || target_y_pointer.is_null()
            || target_amplitude_pointer.is_null()
            || target_phase_pointer.is_null())
    {
        return -1;
    }

    let phase = slice::from_raw_parts_mut(phase_pointer, pixel_count);
    let scratch_real = slice::from_raw_parts_mut(scratch_real_pointer, pixel_count);
    let scratch_imag = slice::from_raw_parts_mut(scratch_imag_pointer, pixel_count);
    scratch_real.fill(0.0);
    scratch_imag.fill(0.0);
    if target_count == 0 {
        if deterministic_fallback != 0 {
            phase.fill(0.0);
        }
        return 0;
    }

    let target_x = slice::from_raw_parts(target_x_pointer, target_count);
    let target_y = slice::from_raw_parts(target_y_pointer, target_count);
    let target_amplitude = slice::from_raw_parts(target_amplitude_pointer, target_count);
    let target_phase = slice::from_raw_parts(target_phase_pointer, target_count);
    let tau = 2.0 * core::f64::consts::PI;

    for target in 0..target_count {
        let frequency_x = principal_frequency(target_x[target], width);
        let frequency_y = principal_frequency(target_y[target], height);
        let amplitude = target_amplitude[target];
        let initial_phase = target_phase[target];
        if !frequency_x.is_finite()
            || !frequency_y.is_finite()
            || !amplitude.is_finite()
            || !initial_phase.is_finite()
        {
            return -2;
        }
        let angle_x = tau * frequency_x / width as f64;
        let angle_y = tau * frequency_y / height as f64;
        let root_x_real = cos(angle_x);
        let root_x_imag = sin(angle_x);
        let root_y_real = cos(angle_y);
        let root_y_imag = sin(angle_y);
        let mut row_real = cos(initial_phase) * amplitude;
        let mut row_imag = sin(initial_phase) * amplitude;

        for y in 0..height {
            let mut factor_real = row_real;
            let mut factor_imag = row_imag;
            let row_offset = y * width;
            for x in 0..width {
                let index = row_offset + x;
                scratch_real[index] += factor_real;
                scratch_imag[index] += factor_imag;
                let next_real = factor_real * root_x_real - factor_imag * root_x_imag;
                factor_imag = factor_real * root_x_imag + factor_imag * root_x_real;
                factor_real = next_real;
            }
            let next_row_real = row_real * root_y_real - row_imag * root_y_imag;
            row_imag = row_real * root_y_imag + row_imag * root_y_real;
            row_real = next_row_real;
        }
    }

    let threshold_squared = cancellation_threshold.max(0.0) * cancellation_threshold.max(0.0);
    for index in 0..pixel_count {
        let magnitude_squared =
            scratch_real[index] * scratch_real[index] + scratch_imag[index] * scratch_imag[index];
        if magnitude_squared > threshold_squared {
            phase[index] = atan2(scratch_imag[index], scratch_real[index]);
        } else if deterministic_fallback != 0 {
            phase[index] = deterministic_phase(index as u32, deterministic_seed);
        }
    }
    0
}

fn principal_frequency(position: f64, extent: usize) -> f64 {
    let period = extent as f64;
    let mut frequency = position % period;
    if frequency > period * 0.5 {
        frequency -= period;
    } else if frequency < -period * 0.5 {
        frequency += period;
    }
    frequency
}

fn deterministic_phase(index: u32, seed: u32) -> f64 {
    let mut value = (index ^ seed).wrapping_add(1);
    value = (value ^ (value >> 16)).wrapping_mul(0x7feb352d);
    value = (value ^ (value >> 15)).wrapping_mul(0x846ca68b);
    value ^= value >> 16;
    ((value >> 8) as f64 / 16_777_216.0) * (2.0 * core::f64::consts::PI) - core::f64::consts::PI
}

#[no_mangle]
pub extern "C" fn core_hungarian_workspace_bytes(rows: u32, columns: u32) -> u32 {
    hungarian_workspace_bytes(rows as usize, columns as usize)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .unwrap_or(0)
}

/// Solve a row-major rectangular cost matrix. Returns 1 for a feasible
/// assignment, 0 for an infeasible assignment, and -1 for an invalid call.
///
/// # Safety
///
/// `cost_pointer` must address `rows * columns` f64 values,
/// `assignment_pointer` must address `rows` i32 values, and
/// `workspace_pointer` must address the number of bytes reported by
/// `core_hungarian_workspace_bytes`. All regions must be aligned where needed
/// and must not overlap.
#[no_mangle]
pub unsafe extern "C" fn core_hungarian(
    cost_pointer: *const f64,
    rows: u32,
    columns: u32,
    assignment_pointer: *mut i32,
    workspace_pointer: *mut u8,
) -> i32 {
    let rows = rows as usize;
    let columns = columns as usize;
    if rows == 0 {
        return 1;
    }
    if columns == 0 || rows > columns {
        return 0;
    }
    let Some(cost_length) = rows.checked_mul(columns) else {
        return -1;
    };
    if cost_pointer.is_null() || assignment_pointer.is_null() || workspace_pointer.is_null() {
        return -1;
    }

    let costs = slice::from_raw_parts(cost_pointer, cost_length);
    let assignment = slice::from_raw_parts_mut(assignment_pointer, rows);
    assignment.fill(-1);

    let Some(layout) = HungarianLayout::new(rows, columns) else {
        return -1;
    };
    let base = workspace_pointer;
    let u = slice::from_raw_parts_mut(base.add(layout.u_offset).cast::<f64>(), rows + 1);
    let v = slice::from_raw_parts_mut(base.add(layout.v_offset).cast::<f64>(), columns + 1);
    let minimum =
        slice::from_raw_parts_mut(base.add(layout.minimum_offset).cast::<f64>(), columns + 1);
    let p = slice::from_raw_parts_mut(base.add(layout.p_offset).cast::<i32>(), columns + 1);
    let way = slice::from_raw_parts_mut(base.add(layout.way_offset).cast::<i32>(), columns + 1);
    let used = slice::from_raw_parts_mut(base.add(layout.used_offset), columns + 1);

    u.fill(0.0);
    v.fill(0.0);
    p.fill(0);
    way.fill(0);

    for row in 1..=rows {
        p[0] = row as i32;
        let mut column_zero = 0usize;
        minimum.fill(ASSIGNMENT_INFINITY);
        used.fill(0);
        loop {
            used[column_zero] = 1;
            let row_zero = p[column_zero] as usize;
            let mut delta = ASSIGNMENT_INFINITY;
            let mut next_column = 0usize;
            for column in 1..=columns {
                if used[column] != 0 {
                    continue;
                }
                let matrix_value = costs[(row_zero - 1) * columns + column - 1];
                let reduced = matrix_value - u[row_zero] - v[column];
                if reduced < minimum[column] {
                    minimum[column] = reduced;
                    way[column] = column_zero as i32;
                }
                if minimum[column] < delta || (minimum[column] == delta && column < next_column) {
                    delta = minimum[column];
                    next_column = column;
                }
            }
            if !delta.is_finite() || delta >= ASSIGNMENT_INFINITY / 2.0 {
                return 0;
            }
            for column in 0..=columns {
                if used[column] != 0 {
                    let assigned_row = p[column] as usize;
                    u[assigned_row] += delta;
                    v[column] -= delta;
                } else {
                    minimum[column] -= delta;
                }
            }
            column_zero = next_column;
            if p[column_zero] == 0 {
                break;
            }
        }

        loop {
            let previous_column = way[column_zero] as usize;
            p[column_zero] = p[previous_column];
            column_zero = previous_column;
            if column_zero == 0 {
                break;
            }
        }
    }

    for (column, &assigned_row) in p.iter().enumerate().take(columns + 1).skip(1) {
        if assigned_row > 0 {
            assignment[assigned_row as usize - 1] = column as i32 - 1;
        }
    }
    for row in 0..rows {
        let column = assignment[row];
        if column < 0 || costs[row * columns + column as usize] >= ASSIGNMENT_INFINITY / 2.0 {
            return 0;
        }
    }
    1
}

fn fft_1d(
    real: &mut [f64],
    imag: &mut [f64],
    scratch_real: &mut [f64],
    scratch_imag: &mut [f64],
    inverse: bool,
) {
    let length = real.len();
    if length == 0 {
        return;
    }
    if !length.is_power_of_two() {
        dft_1d(real, imag, scratch_real, scratch_imag, inverse);
        return;
    }

    bit_reverse_permutation(real, imag);
    let mut size = 2usize;
    loop {
        let half = size >> 1;
        let sign = if inverse { 1.0 } else { -1.0 };
        let angle = sign * (2.0 * core::f64::consts::PI / size as f64);
        let root_real = unsafe { cos(angle) };
        let root_imag = unsafe { sin(angle) };
        let mut offset = 0usize;
        while offset < length {
            let mut twiddle_real = 1.0;
            let mut twiddle_imag = 0.0;
            for index in 0..half {
                let even = offset + index;
                let odd = even + half;
                let product_real = real[odd] * twiddle_real - imag[odd] * twiddle_imag;
                let product_imag = real[odd] * twiddle_imag + imag[odd] * twiddle_real;
                let even_real = real[even];
                let even_imag = imag[even];
                real[even] = even_real + product_real;
                imag[even] = even_imag + product_imag;
                real[odd] = even_real - product_real;
                imag[odd] = even_imag - product_imag;
                let next_twiddle_real = twiddle_real * root_real - twiddle_imag * root_imag;
                twiddle_imag = twiddle_real * root_imag + twiddle_imag * root_real;
                twiddle_real = next_twiddle_real;
            }
            offset += size;
        }
        if size == length {
            break;
        }
        size <<= 1;
    }

    if inverse {
        let scale = 1.0 / length as f64;
        for index in 0..length {
            real[index] *= scale;
            imag[index] *= scale;
        }
    }
}

fn bit_reverse_permutation(real: &mut [f64], imag: &mut [f64]) {
    let mut reversed = 0usize;
    for index in 1..real.len() {
        let mut bit = real.len() >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            real.swap(index, reversed);
            imag.swap(index, reversed);
        }
    }
}

fn dft_1d(
    real: &mut [f64],
    imag: &mut [f64],
    scratch_real: &mut [f64],
    scratch_imag: &mut [f64],
    inverse: bool,
) {
    let length = real.len();
    scratch_real[..length].copy_from_slice(real);
    scratch_imag[..length].copy_from_slice(imag);
    let sign = if inverse { 1.0 } else { -1.0 };
    for output in 0..length {
        let angle = sign * 2.0 * core::f64::consts::PI * output as f64 / length as f64;
        let root_real = unsafe { cos(angle) };
        let root_imag = unsafe { sin(angle) };
        let mut twiddle_real = 1.0;
        let mut twiddle_imag = 0.0;
        let mut sum_real = 0.0;
        let mut sum_imag = 0.0;
        for input in 0..length {
            sum_real += scratch_real[input] * twiddle_real - scratch_imag[input] * twiddle_imag;
            sum_imag += scratch_real[input] * twiddle_imag + scratch_imag[input] * twiddle_real;
            let next_twiddle_real = twiddle_real * root_real - twiddle_imag * root_imag;
            twiddle_imag = twiddle_real * root_imag + twiddle_imag * root_real;
            twiddle_real = next_twiddle_real;
        }
        let scale = if inverse { 1.0 / length as f64 } else { 1.0 };
        real[output] = sum_real * scale;
        imag[output] = sum_imag * scale;
    }
}

struct HungarianLayout {
    u_offset: usize,
    v_offset: usize,
    minimum_offset: usize,
    p_offset: usize,
    way_offset: usize,
    used_offset: usize,
    bytes: usize,
}

impl HungarianLayout {
    fn new(rows: usize, columns: usize) -> Option<Self> {
        let u_offset = 0usize;
        let v_offset = checked_advance(u_offset, rows.checked_add(1)?, 8)?;
        let minimum_offset = checked_advance(v_offset, columns.checked_add(1)?, 8)?;
        let after_minimum = checked_advance(minimum_offset, columns.checked_add(1)?, 8)?;
        let p_offset = align_up(after_minimum, 4)?;
        let way_offset = checked_advance(p_offset, columns.checked_add(1)?, 4)?;
        let used_offset = checked_advance(way_offset, columns.checked_add(1)?, 4)?;
        let bytes = used_offset.checked_add(columns.checked_add(1)?)?;
        Some(Self {
            u_offset,
            v_offset,
            minimum_offset,
            p_offset,
            way_offset,
            used_offset,
            bytes,
        })
    }
}

fn hungarian_workspace_bytes(rows: usize, columns: usize) -> Option<usize> {
    HungarianLayout::new(rows, columns).map(|layout| layout.bytes)
}

fn checked_advance(offset: usize, count: usize, element_size: usize) -> Option<usize> {
    offset.checked_add(count.checked_mul(element_size)?)
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
}
