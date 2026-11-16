use super::{AssetError, AssetReader, ResourceKey, invalid};
use std::sync::Arc;

impl AssetReader<'_> {
    /// Resolves a physics material or collision preset hash from installed native libraries.
    /// RED packages store names as strings, so numeric hashes must be resolved before writing.
    ///
    /// # Errors
    /// Returns an error for absent libraries or an unknown hash rather than changing its identity.
    pub fn collision_name(&self, hash: u64) -> Result<String, AssetError> {
        let load = || -> Result<Arc<std::collections::HashMap<u64, String>>, AssetError> {
            let mut names = std::collections::HashMap::from([(0, "None".to_owned())]);
            for path in [
                r"base\physics\physicsmaterials.physmatlib",
                r"engine\physics\collision_presets.json",
            ] {
                let bytes = self.read_bytes(&ResourceKey::parse(path))?;
                let inspection =
                    crate::cr2w::inspect_bytes(&bytes).map_err(|e| invalid(e.to_string()))?;
                for name in inspection.names {
                    let hash = name.value.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
                        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
                    });
                    names.insert(hash, name.value);
                }
            }
            Ok(Arc::new(names))
        };
        let names = if let Some(shared) = self.shared {
            let mut cache = shared
                .collision_names
                .lock()
                .map_err(|_| invalid("physics name cache unavailable"))?;
            if cache.is_none() {
                *cache = Some(load()?);
            }
            cache
                .as_ref()
                .cloned()
                .ok_or_else(|| invalid("physics names absent"))?
        } else {
            load()?
        };
        names.get(&hash).cloned().ok_or_else(|| {
            invalid(format!(
                "physics name hash {hash} absent from installed libraries"
            ))
        })
    }

    /// Resolves native cooked geometry by sector/hash, including always-loaded fallback.
    /// Metadata is indexed once; only requested sector buffers are decompressed.
    ///
    /// # Errors
    /// Returns resource, native cache or cooked topology errors.
    pub fn collision_geometry(
        &self,
        sector: u64,
        geometry: u64,
    ) -> Result<Arc<crate::physics::CollisionGeometry>, AssetError> {
        let key = ResourceKey::parse(
            r"base\worlds\03_night_city\sectors\_generated\collisions\03_night_city.geometry_cache",
        );
        let load = || {
            crate::physics::GeometryCache::new(self.read_bytes(&key)?, self.schema)
                .map(Arc::new)
                .map_err(|e| invalid(e.to_string()))
        };
        let cache = if let Some(shared) = self.shared {
            let mut cache = shared
                .collision
                .lock()
                .map_err(|_| invalid("collision cache unavailable"))?;
            if cache.is_none() {
                *cache = Some(load()?);
            }
            cache
                .as_ref()
                .cloned()
                .ok_or_else(|| invalid("collision cache absent"))?
        } else {
            load()?
        };
        cache
            .mesh(sector, geometry)
            .map_err(|e| invalid(e.to_string()))
    }
}
