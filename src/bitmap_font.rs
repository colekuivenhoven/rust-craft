// Tiny built-in 5x7 bitmap font for HUD text.
// We keep it minimal on purpose (digits + uppercase letters + space + 'x').

pub const GLYPH_W: u32 = 5;
pub const GLYPH_H: u32 = 7;

fn glyph_rows(ch: char) -> [u8; 7] {
    // Each u8 uses the lower 5 bits as pixels (bit 4 = leftmost).
    match ch {
        ' ' => [0, 0, 0, 0, 0, 0, 0],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01111, 0b10000, 0b10000, 0b10011, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'x' => [0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0, 0],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '?' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0, 0b00100],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        _ => glyph_rows('?'),
    }
}

pub fn draw_text_quads(
    out: &mut Vec<crate::block::UiVertex>,
    text: &str,
    mut x: f32,
    y: f32,
    pixel_w: f32,
    pixel_h: f32,
    color: [f32; 4],
    screen_w: f32,
    screen_h: f32,
) {
    for ch in text.chars() {
        let ch = if ch.is_ascii_lowercase() {
            ch.to_ascii_uppercase()
        } else {
            ch
        };

        let rows = glyph_rows(ch);
        for (ry, row_bits) in rows.iter().enumerate() {
            for cx in 0..GLYPH_W {
                let mask = 1u8 << (GLYPH_W - 1 - cx);
                if (row_bits & mask) != 0 {
                    let px = x + (cx as f32) * pixel_w;
                    let py = y + (ry as f32) * pixel_h;
                    push_rect_px(out, px, py, pixel_w, pixel_h, color, screen_w, screen_h);
                }
            }
        }

        // spacing
        x += (GLYPH_W as f32 + 1.0) * pixel_w;
    }
}

pub fn push_rect_px(
    out: &mut Vec<crate::block::UiVertex>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: [f32; 4],
    screen_w: f32,
    screen_h: f32,
) {
    // Convert from pixel space (top-left origin) to clip space.
    let x0 = (x / screen_w) * 2.0 - 1.0;
    let x1 = ((x + w) / screen_w) * 2.0 - 1.0;
    let y0 = 1.0 - (y / screen_h) * 2.0;
    let y1 = 1.0 - ((y + h) / screen_h) * 2.0;

    out.extend_from_slice(&[
        crate::block::UiVertex { position: [x0, y0], color },
        crate::block::UiVertex { position: [x1, y0], color },
        crate::block::UiVertex { position: [x1, y1], color },
        crate::block::UiVertex { position: [x0, y0], color },
        crate::block::UiVertex { position: [x1, y1], color },
        crate::block::UiVertex { position: [x0, y1], color },
    ]);
}
