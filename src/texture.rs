//! Native uncompressed UI textures and single-image inventory atlases.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::RgbaImage;
use serde_json::{Value, json};

/// Builds a native RGBA8 UI texture document with straight alpha and sRGB colour.
/// Inventory atlas sampling requires the reverse scanline order from a PNG.
/// The caller's image remains in its normal upright orientation.
/// The document can be serialized with [`crate::writer::write`].
#[must_use]
pub fn rgba_ui_texture(image: &RgbaImage) -> Value {
    let width = image.width();
    let height = image.height();
    let size = u64::from(width) * u64::from(height) * 4;
    let texture = image::imageops::flip_vertical(image);
    json!({"Data":{"RootChunk":{"$type":"CBitmapTexture","cookingPlatform":"PLATFORM_PC","width":width,"height":height,
    "setup":{"$type":"STextureGroupSetup","group":"TEXG_Generic_UI","compression":"TCM_None","rawFormat":"TRF_TrueColor","isStreamable":false,"hasMipchain":false,"isGamma":true},
    "renderTextureResource":{"$type":"rendRenderTextureResource","renderResourceBlobPC":{"HandleId":"0","Data":{"$type":"rendRenderTextureBlobPC",
        "header":{"$type":"rendRenderTextureBlobHeader","version":2,"sizeInfo":{"$type":"rendRenderTextureBlobSizeInfo","width":width,"height":height},
            "textureInfo":{"$type":"rendRenderTextureBlobTextureInfo","textureDataSize":size,"sliceSize":size,"dataAlignment":8,"sliceCount":1,"mipCount":1},
            "mipMapInfo":[{"$type":"rendRenderTextureBlobMipMapInfo","layout":{"$type":"rendRenderTextureBlobMemoryLayout","rowPitch":width*4,"slicePitch":size},"placement":{"$type":"rendRenderTextureBlobPlacement","size":size}}],"flags":1},
        "textureData":{"BufferId":"0","Flags":131072,"Bytes":STANDARD.encode(texture.as_raw())}
    }}}}}})
}

/// Builds the three native atlas resolution slots around one full-UV image.
#[must_use]
pub fn single_image_atlas(texture: &str, part: &str) -> Value {
    let slot = json!({"$type":"inkTextureSlot","texture":{"DepotPath":{"$type":"ResourcePath","$storage":"string","$value":texture},"Flags":"Soft"},
        "parts":[{"$type":"inkTextureAtlasMapper","partName":{"$type":"CName","$storage":"string","$value":part},"clippingRectInUVCoords":{"$type":"RectF","Left":0.,"Top":0.,"Right":1.,"Bottom":1.}}]});
    json!({"Data":{"RootChunk":{"$type":"inkTextureAtlas","cookingPlatform":"PLATFORM_PC","slots":{"Elements":[slot.clone(),slot.clone(),slot]}}}})
}
