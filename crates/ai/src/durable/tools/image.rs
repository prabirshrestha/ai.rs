//! Port of durable `src/tools/image.ts`: image MIME sniffing by content.

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/// The MIME type of a supported image (`png` but not `apng`, `jpeg`, `gif`, `webp`, `bmp`), by content.
pub fn detect_supported_image_mime_type(buffer: &[u8]) -> Option<&'static str> {
    if buffer.starts_with(&[0xff, 0xd8, 0xff]) {
        return if buffer.get(3) == Some(&0xf7) {
            None
        } else {
            Some("image/jpeg")
        };
    }
    if buffer.starts_with(&PNG_SIGNATURE) {
        return if is_png(buffer) && !is_animated_png(buffer) {
            Some("image/png")
        } else {
            None
        };
    }
    if starts_with_ascii(buffer, 0, "GIF87a") || starts_with_ascii(buffer, 0, "GIF89a") {
        return Some("image/gif");
    }
    if starts_with_ascii(buffer, 0, "RIFF") && starts_with_ascii(buffer, 8, "WEBP") {
        return Some("image/webp");
    }
    if starts_with_ascii(buffer, 0, "BM") && is_bmp(buffer) {
        return Some("image/bmp");
    }
    None
}

fn is_png(buffer: &[u8]) -> bool {
    buffer.len() >= 16
        && read_u32_be(buffer, PNG_SIGNATURE.len()) == 13
        && starts_with_ascii(buffer, 12, "IHDR")
}

fn is_animated_png(buffer: &[u8]) -> bool {
    let mut offset = PNG_SIGNATURE.len() as u64;
    let length = buffer.len() as u64;
    while offset + 8 <= length {
        let chunk_length = u64::from(read_u32_be(buffer, offset as usize));
        let chunk_type_offset = offset as usize + 4;
        if starts_with_ascii(buffer, chunk_type_offset, "acTL") {
            return true;
        }
        if starts_with_ascii(buffer, chunk_type_offset, "IDAT") {
            return false;
        }
        let next_offset = offset + 8 + chunk_length + 4;
        if next_offset <= offset || next_offset > length {
            return false;
        }
        offset = next_offset;
    }
    false
}

fn is_bmp(buffer: &[u8]) -> bool {
    if buffer.len() < 26 {
        return false;
    }
    let declared_file_size = u64::from(read_u32_le(buffer, 2));
    let pixel_data_offset = u64::from(read_u32_le(buffer, 10));
    let dib_header_size = u64::from(read_u32_le(buffer, 14));
    if declared_file_size != 0 && declared_file_size < 26 {
        return false;
    }
    if pixel_data_offset < 14 + dib_header_size {
        return false;
    }
    if declared_file_size != 0 && pixel_data_offset >= declared_file_size {
        return false;
    }

    let (color_planes, bits_per_pixel) = if dib_header_size == 12 {
        (read_u16_le(buffer, 22), read_u16_le(buffer, 24))
    } else if (40..=124).contains(&dib_header_size) {
        if buffer.len() < 30 {
            return false;
        }
        (read_u16_le(buffer, 26), read_u16_le(buffer, 28))
    } else {
        return false;
    };
    color_planes == 1 && [1, 4, 8, 16, 24, 32].contains(&bits_per_pixel)
}

fn byte(buffer: &[u8], offset: usize) -> u32 {
    u32::from(buffer.get(offset).copied().unwrap_or(0))
}

fn read_u16_le(buffer: &[u8], offset: usize) -> u32 {
    byte(buffer, offset) + (byte(buffer, offset + 1) << 8)
}

fn read_u32_be(buffer: &[u8], offset: usize) -> u32 {
    (byte(buffer, offset) << 24)
        + (byte(buffer, offset + 1) << 16)
        + (byte(buffer, offset + 2) << 8)
        + byte(buffer, offset + 3)
}

fn read_u32_le(buffer: &[u8], offset: usize) -> u32 {
    byte(buffer, offset)
        + (byte(buffer, offset + 1) << 8)
        + (byte(buffer, offset + 2) << 16)
        + (byte(buffer, offset + 3) << 24)
}

fn starts_with_ascii(buffer: &[u8], offset: usize, text: &str) -> bool {
    buffer
        .get(offset..offset + text.len())
        .is_some_and(|slice| slice == text.as_bytes())
}
