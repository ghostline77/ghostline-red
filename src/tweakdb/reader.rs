use super::{FlatValue, TweakDb, TweakError, TweakId};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn err(&self, reason: impl Into<String>) -> TweakError {
        TweakError::Invalid {
            offset: self.pos,
            reason: reason.into(),
        }
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8], TweakError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| self.err("length overflow"))?;
        let v = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| self.err("truncated value"))?;
        self.pos = end;
        Ok(v)
    }
    fn u8(&mut self) -> Result<u8, TweakError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, TweakError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64, TweakError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn count(&mut self, min: usize) -> Result<usize, TweakError> {
        let n = self.u32()? as usize;
        if n > self.bytes.len().saturating_sub(self.pos) / min.max(1) {
            return Err(self.err("count exceeds remaining table"));
        }
        Ok(n)
    }
    fn vlq(&mut self) -> Result<i32, TweakError> {
        let first = self.u8()?;
        let mut n = u64::from(first & 63);
        let mut more = first & 64 != 0;
        let mut shift = 6;
        while more {
            if shift > 27 {
                return Err(self.err("VLQ overflow"));
            }
            let b = self.u8()?;
            n |= u64::from(b & 127) << shift;
            shift += 7;
            more = b & 128 != 0;
        }
        let n = i32::try_from(n).map_err(|_| self.err("VLQ exceeds signed range"))?;
        Ok(if first & 128 != 0 { -n } else { n })
    }
    fn string(&mut self) -> Result<String, TweakError> {
        let n = self.vlq()?;
        if n <= 0 {
            let b = self.take(n.unsigned_abs() as usize)?;
            String::from_utf8(b.to_vec()).map_err(|_| self.err("invalid UTF-8"))
        } else {
            let b = self.take(n.unsigned_abs() as usize * 2)?;
            String::from_utf16(
                &b.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect::<Vec<_>>(),
            )
            .map_err(|_| self.err("invalid UTF-16"))
        }
    }
    fn float(&mut self) -> Result<Value, TweakError> {
        let f = f32::from_bits(self.u32()?);
        serde_json::Number::from_f64(f64::from(f))
            .map(Value::Number)
            .ok_or_else(|| self.err("nonfinite float"))
    }
    fn value(&mut self, ty: &str) -> Result<Value, TweakError> {
        if let Some(element) = ty.strip_prefix("array:") {
            let n = self.vlq()?;
            if n < 0 || n.unsigned_abs() as usize > self.bytes.len().saturating_sub(self.pos) {
                return Err(self.err("invalid array length"));
            }
            return (0..n)
                .map(|_| self.value(element))
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array);
        }
        Ok(match ty {
            "CName" | "String" => Value::String(self.string()?),
            "TweakDBID" | "raRef:CResource" | "gamedataLocKeyWrapper" | "Uint64" => {
                Value::String(self.u64()?.to_string())
            }
            "Int64" => Value::String(
                i64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes")).to_string(),
            ),
            "Float" => self.float()?,
            "Bool" => {
                let v = self.u8()?;
                if v > 1 {
                    return Err(self.err("invalid boolean"));
                }
                Value::Bool(v != 0)
            }
            "Uint8" => json!(self.u8()?),
            "Int8" => json!(i8::from_le_bytes([self.u8()?])),
            "Uint16" => json!(u16::from_le_bytes(
                self.take(2)?.try_into().expect("two bytes")
            )),
            "Int16" => json!(i16::from_le_bytes(
                self.take(2)?.try_into().expect("two bytes")
            )),
            "Uint32" => json!(self.u32()?),
            "Int32" => json!(i32::from_le_bytes(
                self.take(4)?.try_into().expect("four bytes")
            )),
            "Color" => {
                let b = self.take(4)?;
                json!({"r":b[0],"g":b[1],"b":b[2],"a":b[3]})
            }
            "Vector2" | "Vector3" | "Quaternion" | "EulerAngles" => {
                let fields: &[&str] = match ty {
                    "Vector2" => &["X", "Y"],
                    "Vector3" => &["X", "Y", "Z"],
                    "Quaternion" => &["i", "j", "k", "r"],
                    _ => &["Pitch", "Yaw", "Roll"],
                };
                let mut obj = serde_json::Map::new();
                for k in fields {
                    obj.insert((*k).into(), self.float()?);
                }
                Value::Object(obj)
            }
            _ => return Err(self.err(format!("unhandled type {ty}"))),
        })
    }
}

fn type_names() -> HashMap<u64, String> {
    let mut out = HashMap::new();
    for ty in [
        "CName",
        "String",
        "TweakDBID",
        "raRef:CResource",
        "Float",
        "Bool",
        "Uint8",
        "Uint16",
        "Uint32",
        "Uint64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Color",
        "EulerAngles",
        "Quaternion",
        "Vector2",
        "Vector3",
        "gamedataLocKeyWrapper",
    ] {
        for name in [ty.to_owned(), format!("array:{ty}")] {
            let hash = name.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
            });
            out.insert(hash, name);
        }
    }
    out
}

#[expect(
    clippy::too_many_lines,
    reason = "section validation follows the native table order"
)]
pub(super) fn decode(bytes: &[u8]) -> Result<TweakDb, TweakError> {
    let mut r = Reader { bytes, pos: 0 };
    if r.u32()? != 0x0bb1_db47 {
        return Err(r.err("wrong magic"));
    }
    let blob = r.u32()?;
    let parser = r.u32()?;
    if (blob, parser) != (8, 4) {
        return Err(TweakError::Version { blob, parser });
    }
    let record_checksum = r.u32()?;
    let offsets = [
        r.u32()? as usize,
        r.u32()? as usize,
        r.u32()? as usize,
        r.u32()? as usize,
        bytes.len(),
    ];
    if offsets[0] < 32 || offsets.windows(2).any(|x| x[0] > x[1]) {
        return Err(r.err("invalid section offsets"));
    }
    let mut flats = HashMap::new();
    r = Reader {
        bytes: &bytes[..offsets[1]],
        pos: offsets[0],
    };
    let n = r.count(20)?;
    let mut pools = Vec::with_capacity(n);
    for _ in 0..n {
        pools.push((
            r.u64()?,
            r.u32()? as usize,
            r.u32()? as usize,
            r.u32()? as usize,
        ));
    }
    let header_end = r.pos;
    let types = type_names();
    for (i, (hash, values, keys, start)) in pools.iter().copied().enumerate() {
        let end = pools.get(i + 1).map_or(offsets[1], |p| p.3);
        if start < header_end || start > end || end > offsets[1] {
            return Err(r.err("invalid pool offsets"));
        }
        let ty = types.get(&hash).ok_or(TweakError::Type(hash))?;
        let mut p = Reader {
            bytes: &bytes[..end],
            pos: start,
        };
        let count = p.count(1)?;
        if count > values {
            return Err(p.err("value count exceeds pool capacity"));
        }
        let mut decoded = Vec::with_capacity(count);
        for _ in 0..count {
            decoded.push(Arc::new(FlatValue {
                native_type: ty.clone(),
                value: p.value(ty)?,
            }));
        }
        let count = p.count(12)?;
        if count > keys {
            return Err(p.err("key count exceeds pool capacity"));
        }
        flats.reserve(count);
        for _ in 0..count {
            let id = TweakId(p.u64()?);
            let index = p.u32()? as usize;
            let value = decoded
                .get(index)
                .ok_or_else(|| p.err("flat pool index out of range"))?;
            if flats.insert(id, Arc::clone(value)).is_some() {
                return Err(p.err("duplicate flat identity"));
            }
        }
        if p.pos != end {
            return Err(p.err("trailing pool data"));
        }
    }
    r = Reader {
        bytes: &bytes[..offsets[2]],
        pos: offsets[1],
    };
    let n = r.count(12)?;
    let mut records = BTreeMap::new();
    for _ in 0..n {
        let id = TweakId(r.u64()?);
        if records.insert(id, r.u32()?).is_some() {
            return Err(r.err("duplicate record identity"));
        }
    }
    if r.pos != offsets[2] {
        return Err(r.err("trailing record data"));
    }
    r = Reader {
        bytes: &bytes[..offsets[3]],
        pos: offsets[2],
    };
    let n = r.count(12)?;
    let mut queries = BTreeMap::new();
    for _ in 0..n {
        let id = TweakId(r.u64()?);
        let count = r.count(8)?;
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(TweakId(r.u64()?));
        }
        if queries.insert(id, v).is_some() {
            return Err(r.err("duplicate query"));
        }
    }
    if r.pos != offsets[3] {
        return Err(r.err("trailing query data"));
    }
    r = Reader {
        bytes,
        pos: offsets[3],
    };
    let n = r.count(9)?;
    let mut group_tags = BTreeMap::new();
    for _ in 0..n {
        let id = TweakId(r.u64()?);
        if group_tags.insert(id, r.u8()?).is_some() {
            return Err(r.err("duplicate group tag"));
        }
    }
    if r.pos != bytes.len() {
        return Err(r.err("trailing group tag data"));
    }
    Ok(TweakDb {
        records,
        flats,
        queries,
        group_tags,
        record_checksum,
    })
}

pub(super) fn names(bytes: &[u8]) -> Result<Vec<String>, TweakError> {
    let decoded;
    let bytes = if bytes.starts_with(b"KARK") {
        let mut r = Reader { bytes, pos: 4 };
        let size = r.u32()? as usize;
        // Name dictionaries are developer input; bound decompression allocations.
        if size > 512 * 1024 * 1024 {
            return Err(r.err("name dictionary exceeds 512 MiB"));
        }
        decoded = crate::kraken::decode(&bytes[8..], size)?;
        decoded.as_slice()
    } else {
        bytes
    };
    let mut r = Reader { bytes, pos: 0 };
    r.take(20)?;
    let mut out = Vec::new();
    while r.pos < bytes.len() {
        out.push(r.string()?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let hash = *type_names()
            .iter()
            .find(|(_, n)| n.as_str() == "TweakDBID")
            .unwrap()
            .0;
        let mut b = Vec::new();
        for n in [0x0bb1_db47_u32, 8, 4, 0, 32, 84, 100, 104] {
            b.extend(n.to_le_bytes());
        }
        b.extend(1_u32.to_le_bytes());
        b.extend(hash.to_le_bytes());
        for n in [1_u32, 1, 56, 1] {
            b.extend(n.to_le_bytes());
        }
        b.extend(u64::MAX.to_le_bytes());
        b.extend(1_u32.to_le_bytes());
        b.extend(TweakId::from_name("Items.Test.itemType").0.to_le_bytes());
        b.extend(0_u32.to_le_bytes());
        b.extend(1_u32.to_le_bytes());
        b.extend(TweakId::from_name("Items.Test").0.to_le_bytes());
        b.extend(123_u32.to_le_bytes());
        b.extend(0_u32.to_le_bytes());
        b.extend(0_u32.to_le_bytes());
        b
    }

    #[test]
    fn exact_property_lookup_and_64_bit_values() {
        let db = decode(&fixture()).unwrap();
        let value = db
            .property(TweakId::from_name("Items.Test"), "itemType")
            .unwrap();
        assert_eq!(value.native_type, "TweakDBID");
        assert_eq!(value.value, "18446744073709551615");
        assert_eq!(
            TweakId::from_name("Items.Test").property("itemType"),
            TweakId::from_name("Items.Test.itemType")
        );
        assert_eq!(
            TweakId::from_name("Items.ÃƒÂ©").property("Ã¥Â±Å¾Ã¦â‚¬Â§"),
            TweakId::from_name("Items.ÃƒÂ©.Ã¥Â±Å¾Ã¦â‚¬Â§")
        );
    }

    #[test]
    fn rejects_truncation_and_bad_pool_indices() {
        let mut b = fixture();
        for n in 0..b.len() {
            assert!(decode(&b[..n]).is_err(), "prefix {n}");
        }
        b[80..84].copy_from_slice(&9_u32.to_le_bytes());
        assert!(
            decode(&b)
                .unwrap_err()
                .to_string()
                .contains("index out of range")
        );
        b = fixture();
        b[4..8].copy_from_slice(&9_u32.to_le_bytes());
        assert!(matches!(decode(&b), Err(TweakError::Version { .. })));
    }

    #[test]
    fn strings_arrays_and_malformed_lengths() {
        // Negative length: UTF-8; positive: UTF-16. Array lengths use native VLQ.
        let mut r = Reader {
            bytes: &[0x82, b'h', b'i', 2, b'A', 0, b'B', 0, 2, 1, 0],
            pos: 0,
        };
        assert_eq!(r.string().unwrap(), "hi");
        assert_eq!(r.string().unwrap(), "AB");
        assert_eq!(r.value("array:Bool").unwrap(), json!([true, false]));
        assert!(
            Reader {
                bytes: &[0xff, 0xff, 0xff, 0xff, 0xff],
                pos: 0
            }
            .vlq()
            .is_err()
        );
        assert!(
            Reader {
                bytes: &[0x90],
                pos: 0
            }
            .string()
            .is_err()
        );
    }
}
