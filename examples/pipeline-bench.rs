//! Time native Kraken corpus decoding and channel-independent SIMD texture resizing.
use anyhow::{Result, ensure};
use ghostline_red::{archive, kraken};
use image::{Rgba, RgbaImage};
use std::{
    fs::File,
    hint::black_box,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::{Duration, Instant},
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() == 1, "usage: pipeline-bench GAME_ARCHIVE");
    let path = Path::new(&args[0]);
    let index = archive::read_archive(path)?;
    let mut file = File::open(path)?;
    let mut corpus = Vec::new();
    for segment in &index.segments {
        if segment.compressed_size == segment.size
            || segment.compressed_size < 8
            || segment.size > 1_048_576
        {
            continue;
        }
        file.seek(SeekFrom::Start(segment.offset))?;
        let mut bytes = vec![0; segment.compressed_size as usize];
        file.read_exact(&mut bytes)?;
        if bytes.starts_with(b"KARK") {
            let size = u32::from_le_bytes(bytes[4..8].try_into()?);
            corpus.push((bytes, size as usize));
        }
        if corpus.len() == 64 {
            break;
        }
    }
    ensure!(!corpus.is_empty(), "archive has no sampled Kraken segments");
    let mut decode_time = Duration::ZERO;
    let mut decoded_bytes = 0;
    let mut hashes = Vec::new();
    for round in 0..8 {
        for (bytes, size) in &corpus {
            let started = Instant::now();
            let output = kraken::decode(black_box(&bytes[8..]), *size)?;
            decode_time += started.elapsed();
            decoded_bytes += output.len();
            let hash = crc32fast::hash(&output);
            if round == 0 {
                hashes.push(hash);
            }
            black_box(hash);
        }
    }
    let source = RgbaImage::from_fn(2048, 2048, |x, y| {
        Rgba([
            (x.wrapping_mul(7).wrapping_add(y * 3) % 256) as u8,
            (x.wrapping_add(y * 5) % 256) as u8,
            (y % 256) as u8,
            ((x / 32 + y / 32) % 2 * 255) as u8,
        ])
    });
    let started = Instant::now();
    let mut reference = RgbaImage::new(1, 1);
    for _ in 0..8 {
        reference = black_box(image::imageops::resize(
            &source,
            512,
            512,
            image::imageops::FilterType::Triangle,
        ));
    }
    let scalar_seconds = started.elapsed().as_secs_f64();
    use fast_image_resize::{
        FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image,
    };
    let input = Image::from_vec_u8(2048, 2048, source.into_raw(), PixelType::U8x4)?;
    let mut output = Image::new(512, 512, PixelType::U8x4);
    let mut resizer = Resizer::new();
    let options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
        .use_alpha(false);
    let started = Instant::now();
    for _ in 0..8 {
        resizer.resize(black_box(&input), &mut output, &options)?;
    }
    let simd_seconds = started.elapsed().as_secs_f64();
    let max_difference = reference
        .as_raw()
        .iter()
        .zip(output.buffer())
        .map(|(a, b)| a.abs_diff(*b))
        .max();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "archive":args[0],"segments":corpus.len(),"decoded_bytes":decoded_bytes,"decode_seconds":decode_time.as_secs_f64(),
            "decode_mib_per_second":decoded_bytes as f64 / 1_048_576. / decode_time.as_secs_f64(),"decoded_crc32":hashes,
            "resize_scalar_seconds":scalar_seconds,"resize_simd_seconds":simd_seconds,"resize_speedup":scalar_seconds/simd_seconds,
            "resize_max_channel_difference":max_difference,"cpu_extensions":format!("{:?}", fast_image_resize::CpuExtensions::default())
        }))?
    );
    Ok(())
}
