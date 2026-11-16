//! Native persistent traffic-lane geometry and connectivity appendices.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;

/// A lane's world-space centreline and horizontal road polygon.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LaneGeometry {
    pub index: u16,
    pub outline: Vec<[f32; 3]>,
    pub polygon: Vec<[f32; 2]>,
}
/// A native connection at metre distances along the two lane centrelines.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LaneExit {
    pub lane: u16,
    pub probability: u8,
    pub sharp_angle: bool,
    pub exit_m: f32,
    pub entry_m: f32,
}
/// Native outgoing and incoming lane references.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LaneConnections {
    pub index: u16,
    pub outgoing: Vec<LaneExit>,
    pub incoming: Vec<u16>,
}
#[derive(Debug, thiserror::Error)]
pub enum TrafficError {
    #[error("invalid traffic data at byte {offset}: {reason}")]
    Invalid { offset: usize, reason: &'static str },
    #[error("expected native traffic resource {0}")]
    Resource(&'static str),
    #[error("invalid traffic appendix encoding: {0}")]
    Base64(#[from] base64::DecodeError),
}
type Result<T> = std::result::Result<T, TrafficError>;
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl Reader<'_> {
    fn invalid(&self, reason: &'static str) -> TrafficError {
        TrafficError::Invalid {
            offset: self.position,
            reason,
        }
    }
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .position
            .checked_add(N)
            .ok_or_else(|| self.invalid("offset overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| self.invalid("truncated appendix"))?;
        let mut result = [0; N];
        result.copy_from_slice(value);
        self.position = end;
        Ok(result)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn float(&mut self) -> Result<f32> {
        let value = f32::from_le_bytes(self.take()?);
        if !value.is_finite() {
            return Err(self.invalid("nonfinite coordinate or distance"));
        }
        Ok(value)
    }
    fn count(&mut self, minimum_size: usize) -> Result<usize> {
        let first = self.byte()?;
        if first & 0x80 != 0 {
            return Err(self.invalid("negative array count"));
        }
        let mut value = usize::from(first & 0x3f);
        if first & 0x40 != 0 {
            let mut shift = 6;
            loop {
                let byte = self.byte()?;
                value |= usize::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
                if shift > 27 {
                    return Err(self.invalid("array count overflow"));
                }
            }
        }
        if value > self.bytes.len().saturating_sub(self.position) / minimum_size {
            return Err(self.invalid("array count exceeds appendix"));
        }
        Ok(value)
    }
    fn entries(&mut self) -> Result<usize> {
        let count = self.u32()? as usize;
        if count > 65536 || count > self.bytes.len().saturating_sub(self.position) / 4 {
            return Err(self.invalid("invalid lane count"));
        }
        Ok(count)
    }
    fn end(&self) -> Result<()> {
        if self.position != self.bytes.len() {
            return Err(self.invalid("trailing traffic appendix bytes"));
        }
        Ok(())
    }
}

/// Decodes `.lane_polygons` appendix bytes, checking counts, identities and finite coordinates.
pub fn decode_polygons(bytes: &[u8]) -> Result<Vec<LaneGeometry>> {
    let mut reader = Reader { bytes, position: 0 };
    let count = reader.entries()?;
    let mut lanes = Vec::with_capacity(count);
    let mut ids = HashSet::new();
    for _ in 0..count {
        let index = reader.u16()?;
        if !ids.insert(index) {
            return Err(reader.invalid("duplicate lane identity"));
        }
        let count = reader.count(12)?;
        let mut outline = Vec::with_capacity(count);
        for _ in 0..count {
            outline.push([reader.float()?, reader.float()?, reader.float()?]);
        }
        let count = reader.count(8)?;
        let mut polygon = Vec::with_capacity(count);
        for _ in 0..count {
            polygon.push([reader.float()?, reader.float()?]);
        }
        lanes.push(LaneGeometry {
            index,
            outline,
            polygon,
        });
    }
    reader.end()?;
    Ok(lanes)
}
/// Decodes `.lane_connections` appendix bytes, preserving native transition stations.
pub fn decode_connections(bytes: &[u8]) -> Result<Vec<LaneConnections>> {
    let mut reader = Reader { bytes, position: 0 };
    let count = reader.entries()?;
    let mut lanes = Vec::with_capacity(count);
    let mut ids = HashSet::new();
    for _ in 0..count {
        let index = reader.u16()?;
        if !ids.insert(index) {
            return Err(reader.invalid("duplicate lane identity"));
        }
        let count = reader.count(12)?;
        let mut outgoing = Vec::with_capacity(count);
        for _ in 0..count {
            let lane = reader.u16()?;
            let probability = reader.byte()?;
            let sharp = reader.byte()?;
            if sharp > 1 {
                return Err(reader.invalid("invalid sharp-angle flag"));
            }
            let exit_m = reader.float()?;
            let entry_m = reader.float()?;
            if exit_m < 0. || entry_m < 0. {
                return Err(reader.invalid("negative transition station"));
            }
            outgoing.push(LaneExit {
                lane,
                probability,
                sharp_angle: sharp != 0,
                exit_m,
                entry_m,
            });
        }
        let count = reader.count(2)?;
        let mut incoming = Vec::with_capacity(count);
        for _ in 0..count {
            incoming.push(reader.u16()?);
        }
        lanes.push(LaneConnections {
            index,
            outgoing,
            incoming,
        });
    }
    reader.end()?;
    Ok(lanes)
}
fn appendix(document: &Value, kind: &'static str) -> Result<Vec<u8>> {
    let root = &document["Data"]["RootChunk"];
    if root["$type"] != kind {
        return Err(TrafficError::Resource(kind));
    }
    let value = root["$rawTail"]
        .as_str()
        .ok_or(TrafficError::Resource(kind))?;
    Ok(STANDARD.decode(value)?)
}
/// Reads lane geometry from a native-decoded CR2W document without changing its opaque data.
pub fn read_polygons(document: &Value) -> Result<Vec<LaneGeometry>> {
    decode_polygons(&appendix(
        document,
        "worldTrafficPersistentLanePolygonResource",
    )?)
}
/// Reads lane connectivity from a native-decoded CR2W document.
pub fn read_connections(document: &Value) -> Result<Vec<LaneConnections>> {
    decode_connections(&appendix(
        document,
        "worldTrafficPersistentLaneConnectionsResource",
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decodes_geometry_and_rejects_truncation() {
        let mut bytes = Vec::new();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(7u16.to_le_bytes());
        bytes.push(2);
        for point in [[1f32, 2., 3.], [4., 5., 6.]] {
            for value in point {
                bytes.extend(value.to_le_bytes());
            }
        }
        bytes.push(0);
        let lanes = decode_polygons(&bytes).unwrap();
        assert_eq!(lanes[0].index, 7);
        assert_eq!(lanes[0].outline[1], [4., 5., 6.]);
        assert!(decode_polygons(&bytes[..bytes.len() - 1]).is_err());
    }
    #[test]
    fn decodes_connection_stations() {
        let mut bytes = Vec::new();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(7u16.to_le_bytes());
        bytes.push(1);
        bytes.extend(8u16.to_le_bytes());
        bytes.extend([255, 0]);
        bytes.extend(25f32.to_le_bytes());
        bytes.extend(4f32.to_le_bytes());
        bytes.push(0);
        let lanes = decode_connections(&bytes).unwrap();
        assert_eq!(lanes[0].outgoing[0].lane, 8);
        assert_eq!(lanes[0].outgoing[0].exit_m, 25.);
        assert_eq!(lanes[0].outgoing[0].entry_m, 4.);
    }
}
