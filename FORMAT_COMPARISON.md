# Data Format Comparison: HDF5 vs. Modern Alternatives

## Executive Summary

For your PIV sensor archive project, **HDF5 is still a solid choice** for single-file archival, but **Zarr** offers significant advantages for cloud storage, parallel access, and future-proofing. Consider a **hybrid approach** (Zarr for arrays + Parquet for tables) if you want optimal performance.

## Current Setup Analysis

Your current HDF5 usage:
- **Data types**: 3D velocity arrays (time × height × width), tabular sensor data, time series
- **Organization**: Hierarchical groups by experiment runs
- **Compression**: GZIP (level 4)
- **Chunking**: Time-based chunks for arrays
- **Access pattern**: Write-once, read-many (archival)

## Format Comparison

### 1. Zarr (Recommended for Modern Use)

**Advantages:**
- ✅ **Cloud-native**: Works seamlessly with S3, GCS, Azure Blob
- ✅ **Parallel I/O**: Multiple processes can read/write simultaneously
- ✅ **Better compression**: Supports zstd, lz4, blosc (often 2-3x better than gzip)
- ✅ **Versioning**: Built-in support for versioned datasets
- ✅ **Python-native**: Excellent integration with NumPy, Dask, Xarray
- ✅ **Directory-based**: Easier to inspect, backup, and manage
- ✅ **No file locking**: Multiple readers don't block each other
- ✅ **Active development**: Modern, actively maintained

**Disadvantages:**
- ❌ **Multiple files**: Creates a directory structure (not a single file)
- ❌ **Less mature**: Smaller ecosystem than HDF5
- ❌ **Migration effort**: Requires code changes

**Performance:**
- Compression: zstd typically 2-3x better compression than gzip
- Read speed: 2-5x faster for parallel access
- Write speed: Similar or faster

**Example migration:**
```python
import zarr

# Instead of h5py.File
store = zarr.DirectoryStore('data/processed/all_experiments.zarr')
root = zarr.group(store=store)

# Create arrays with better compression
grp = root.create_group(row.group_path)
piv_group = grp.create_group("piv")
piv_group.create_dataset(
    "u",
    data=u.astype(np.float32),
    compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=2),
    chunks=(1, row.grid_height, row.grid_width),
)
```

### 2. NetCDF4

**Advantages:**
- ✅ **Standardized**: CF conventions widely adopted in climate/oceanography
- ✅ **Good tooling**: ncdump, ncview, Panoply
- ✅ **Similar to HDF5**: Easy migration path
- ✅ **Single file**: Like HDF5

**Disadvantages:**
- ❌ **Similar limitations**: Single-file format, less cloud-friendly
- ❌ **Domain-specific**: Less general-purpose than HDF5
- ❌ **Not as modern**: Similar age to HDF5

**Verdict:** Only consider if you need CF conventions or domain-specific tooling.

### 3. Hybrid: Zarr + Parquet

**Approach:**
- Store PIV arrays in Zarr (optimized for multi-dimensional data)
- Store sensor tables in Parquet (already using for intermediate storage)
- Use a manifest JSON to link them

**Advantages:**
- ✅ **Optimal for each type**: Best format for arrays and tables
- ✅ **Leverage existing code**: Already using Parquet
- ✅ **Flexible**: Can query Parquet independently

**Disadvantages:**
- ❌ **Multiple files**: More complex organization
- ❌ **No single archive**: Need to manage multiple files
- ❌ **More code**: Need to handle two formats

**Verdict:** Good if you want maximum performance and flexibility.

### 4. Stay with HDF5

**Advantages:**
- ✅ **Mature ecosystem**: Widely supported, stable
- ✅ **Single file**: Easy to distribute, backup
- ✅ **Good compression**: GZIP works well
- ✅ **No migration**: Already working
- ✅ **Tooling**: h5py, h5dump, HDFView

**Disadvantages:**
- ❌ **Single-file limitations**: Hard to parallelize writes
- ❌ **Cloud-unfriendly**: Can't efficiently stream from S3
- ❌ **File locking**: Multiple readers can be slower
- ❌ **Older format**: Less modern features

**Verdict:** Perfectly fine for single-machine archival. Only migrate if you need cloud storage or parallel access.

## Recommendations by Use Case

### If you need cloud storage (S3/GCS):
→ **Use Zarr** - Native cloud support, efficient streaming

### If you need parallel read access:
→ **Use Zarr** - Multiple processes can read simultaneously

### If you need maximum compression:
→ **Use Zarr with zstd** - Typically 2-3x better than gzip

### If you need a single file for distribution:
→ **Stay with HDF5** - Single-file format is easier to share

### If you need maximum performance:
→ **Hybrid (Zarr + Parquet)** - Best format for each data type

### If you want minimal changes:
→ **Stay with HDF5** - Already working, mature, stable

## Migration Path (if choosing Zarr)

1. **Phase 1: Parallel support**
   - Add zarr as optional dependency
   - Create `write_datasets_zarr()` function
   - Test with small subset

2. **Phase 2: Full migration**
   - Update `write_datasets()` to use zarr
   - Update output path to `.zarr` directory
   - Update documentation

3. **Phase 3: Optimization**
   - Switch to zstd compression
   - Optimize chunk sizes
   - Add parallel read examples

## Code Example: Zarr Migration

```python
# Current HDF5 code (lines 364-486)
def write_datasets(
    h5f: h5py.File,
    row: ExperimentRow,
    piv_root: Path,
    sensor_root: Path,
) -> None:
    # ... existing code ...
    grp = h5f.create_group(row.group_path)
    piv_group = grp.create_group("piv")
    piv_group.create_dataset(
        "u",
        data=u.astype(np.float32),
        compression="gzip",
        compression_opts=4,
        chunks=(1, row.grid_height, row.grid_width),
    )

# Zarr equivalent
def write_datasets_zarr(
    root: zarr.Group,
    row: ExperimentRow,
    piv_root: Path,
    sensor_root: Path,
) -> None:
    # ... same loading code ...
    grp = root.create_group(row.group_path)
    piv_group = grp.create_group("piv")
    piv_group.create_dataset(
        "u",
        data=u.astype(np.float32),
        compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=2),
        chunks=(1, row.grid_height, row.grid_width),
    )
    # Attributes work the same
    grp.attrs.update({...})
```

## Performance Comparison (Estimated)

| Format | Compression Ratio | Read Speed | Parallel Read | Cloud Support |
|--------|------------------|-----------|---------------|---------------|
| HDF5 (gzip) | 1.0x (baseline) | 1.0x | Limited | Poor |
| Zarr (zstd) | 2-3x better | 2-5x faster | Excellent | Excellent |
| NetCDF4 | Similar to HDF5 | Similar | Limited | Poor |
| Parquet | N/A (tabular) | Fast (tabular) | Good | Good |

## Final Recommendation

**For your use case (long-term archival, scientific data):**

1. **Short term**: **Stay with HDF5** - It works, it's stable, and migration has costs.

2. **Long term**: **Migrate to Zarr** if you:
   - Need cloud storage
   - Want better compression
   - Need parallel access
   - Want future-proofing

3. **Alternative**: **Hybrid approach** if you want maximum performance:
   - Zarr for PIV arrays
   - Parquet for sensor tables (already using)
   - JSON manifest to link them

**Bottom line**: HDF5 is not obsolete, but Zarr is more modern and better suited for cloud/distributed workflows. If you're happy with HDF5 and don't need cloud storage, there's no urgent need to migrate.
