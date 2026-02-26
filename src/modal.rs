//* Reusable block-styled modal dialog system.

// ── Panel sizing ───────────────────────────────────────────────────────────
/// Modal width as a fraction of screen width.  Height is derived from this
/// via MODAL_ASPECT so the panel is always exactly 16:9, regardless of screen shape.
pub const MODAL_W_RATIO: f32 = 0.35;
/// Locked aspect ratio (width / height).
pub const MODAL_ASPECT:  f32 = 16.0 / 9.0;

// ── Fixed pixel structural sizes (do NOT scale with the modal) ────────────
/// Outer border thickness in pixels.
pub const MODAL_BORDER_PX:        f32 = 6.0;
/// Inner bevel thickness in pixels.
pub const MODAL_BEVEL_PX:         f32 = 6.0;
/// Button outer border thickness in pixels.
pub const MODAL_BUTTON_BORDER_PX: f32 = 4.0;
/// Button inner bevel thickness in pixels.
pub const MODAL_BUTTON_BEVEL_PX:  f32 = 4.0;
/// Drop-shadow offset for text in pixels.
pub const MODAL_TEXT_SHADOW_PX:   f32 = 4.0;

// ── Button layout (fractions of panel dimensions) ─────────────────────────
pub const MODAL_BUTTON_W_RATIO:         f32 = 0.705; // fraction of panel width
pub const MODAL_BUTTON_H_RATIO:         f32 = 0.186; // fraction of panel height
pub const MODAL_BUTTON_GAP_RATIO:       f32 = 0.048; // fraction of panel height
/// Y distance from top of panel to the first button (fraction of panel height).
pub const MODAL_BUTTONS_Y_OFFSET_RATIO: f32 = 0.379;

// ── Title ─────────────────────────────────────────────────────────────────
/// Y distance from top of panel to title baseline (fraction of panel height).
pub const MODAL_TITLE_Y_OFFSET_RATIO: f32 = 0.103;
/// Title font scale = panel_h × this ratio   (bitmap glyph height = 7 px).
pub const MODAL_TITLE_SCALE_RATIO:    f32 = 0.01034;
/// Button text font scale = panel_h × this ratio.
pub const MODAL_BTN_TEXT_SCALE_RATIO: f32 = 0.006897;

// ── Sand tile ─────────────────────────────────────────────────────────────
/// Fixed tile size in pixels (sand.png is pixel-art; don't stretch it).
pub const SAND_TILE_PX: f32 = 128.0;

// ── Colors ────────────────────────────────────────────────────────────────
pub const MODAL_BORDER_COLOR:        [f32; 4] = [0.12, 0.09, 0.04, 1.00];
pub const MODAL_BEVEL_LIGHT_COLOR:   [f32; 4] = [0.85, 0.78, 0.58, 0.55];
pub const MODAL_BEVEL_SHADOW_COLOR:  [f32; 4] = [0.05, 0.04, 0.02, 0.55];

pub const MODAL_BUTTON_BG_COLOR:     [f32; 4] = [0.35, 0.30, 0.18, 1.00];
pub const MODAL_BUTTON_HOVER_COLOR:  [f32; 4] = [0.55, 0.50, 0.32, 1.00];
pub const MODAL_BUTTON_BORDER_COLOR: [f32; 4] = [0.08, 0.06, 0.02, 1.00];
pub const MODAL_BUTTON_BEVEL_COLOR:  [f32; 4] = [0.70, 0.64, 0.44, 0.60];
pub const MODAL_BUTTON_TEXT_COLOR:   [f32; 4] = [1.00, 0.98, 0.90, 1.00];
pub const MODAL_BUTTON_SHADOW_COLOR: [f32; 4] = [0.12, 0.10, 0.05, 1.00];

pub const MODAL_TITLE_COLOR:  [f32; 4] = [1.00, 0.98, 0.90, 1.00];
pub const MODAL_TITLE_SHADOW: [f32; 4] = [0.10, 0.08, 0.03, 0.90];

use crate::block::{UiVertex, ModalVertex};
use crate::bitmap_font;

// ─────────────────────────────────────────────────────────────────────────
//  ModalButton
// ─────────────────────────────────────────────────────────────────────────

pub struct ModalButton {
    pub label: &'static str,
    /// Bounds in pixel space (top-left x/y, width, height)
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub hovered: bool,
}

impl ModalButton {
    pub fn contains_px(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.w &&
        py >= self.y && py <= self.y + self.h
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Modal
// ─────────────────────────────────────────────────────────────────────────

pub struct Modal {
    pub visible: bool,
    pub title:   &'static str,
    pub buttons: Vec<ModalButton>,

    // Panel bounds in pixel space (set by update_layout)
    pub panel_x: f32,
    pub panel_y: f32,
    pub panel_w: f32,
    pub panel_h: f32,

    /// Panel width as a fraction of screen width (e.g. 0.35 for the pause menu).
    w_ratio: f32,
    /// Panel width-to-height aspect ratio (e.g. 16.0/9.0).
    aspect: f32,

    // Structural sizes:
    //   border_w / bevel_w / btn_border_w / btn_bevel_w / shadow_offset
    //     → fixed pixels (MODAL_*_PX constants, never scale)
    //   title_scale / btn_text_scale / title_y_offset
    //     → scale with modal height
    border_w:        f32,
    bevel_w:         f32,
    btn_border_w:    f32,
    btn_bevel_w:     f32,
    pub title_scale:     f32,
    btn_text_scale:  f32,
    pub title_y_offset:  f32,
    shadow_offset:   f32,
}

impl Modal {
    /// Create a new modal with the given title and button labels.
    ///
    /// `w_ratio` controls the panel width as a fraction of screen width (e.g. `MODAL_W_RATIO`).
    /// `aspect` controls the width-to-height aspect ratio (e.g. `MODAL_ASPECT`).
    ///
    /// Call `update_layout` before generating vertices.
    pub fn new(title: &'static str, button_labels: &[&'static str], w_ratio: f32, aspect: f32) -> Self {
        let buttons = button_labels.iter().map(|&label| ModalButton {
            label,
            x: 0.0, y: 0.0, w: 0.0, h: 0.0,
            hovered: false,
        }).collect();

        Self {
            visible: false,
            title,
            buttons,
            panel_x: 0.0, panel_y: 0.0,
            panel_w: 0.0, panel_h: 0.0,
            w_ratio,
            aspect,
            // Fixed pixel defaults (same values as the PX constants)
            border_w:       MODAL_BORDER_PX,
            bevel_w:        MODAL_BEVEL_PX,
            btn_border_w:   MODAL_BUTTON_BORDER_PX,
            btn_bevel_w:    MODAL_BUTTON_BEVEL_PX,
            shadow_offset:  MODAL_TEXT_SHADOW_PX,
            title_scale:    3.0,
            btn_text_scale: 2.0,
            title_y_offset: 30.0,
        }
    }

    /// Recalculate all panel, button, and typography sizes from the current
    /// screen dimensions.  Call this on startup, resize, and whenever the
    /// screen changes.
    pub fn update_layout(&mut self, screen_w: f32, screen_h: f32) {
        // Width driven by screen; height derived from the configured aspect ratio.
        let pw = screen_w * self.w_ratio;
        let ph = pw / self.aspect;

        // If the 16:9 height would overflow 90 % of the screen height, clamp
        // the height and re-derive the width from the aspect ratio.
        let (pw, ph) = if ph <= screen_h * 0.90 {
            (pw, ph)
        } else {
            let ph = screen_h * 0.90;
            (ph * MODAL_ASPECT, ph)
        };

        self.panel_w = pw;
        self.panel_h = ph;
        self.panel_x = ((screen_w - pw) * 0.5).round();
        self.panel_y = ((screen_h - ph) * 0.5).round();

        // Borders / bevels / shadow stay at their fixed pixel sizes.
        self.border_w      = MODAL_BORDER_PX;
        self.bevel_w       = MODAL_BEVEL_PX;
        self.btn_border_w  = MODAL_BUTTON_BORDER_PX;
        self.btn_bevel_w   = MODAL_BUTTON_BEVEL_PX;
        self.shadow_offset = MODAL_TEXT_SHADOW_PX;

        // Typography: bitmap font glyphs are 7 px tall; scale × 7 = text height.
        // Both scale proportionally with modal height.
        self.title_scale    = (ph * MODAL_TITLE_SCALE_RATIO).max(1.0);
        self.btn_text_scale = (ph * MODAL_BTN_TEXT_SCALE_RATIO).max(1.0);
        self.title_y_offset = ph * MODAL_TITLE_Y_OFFSET_RATIO;

        // Button dimensions scale with the modal.
        let btn_w   = pw * MODAL_BUTTON_W_RATIO;
        let btn_h   = ph * MODAL_BUTTON_H_RATIO;
        let btn_gap = ph * MODAL_BUTTON_GAP_RATIO;
        let btn_x   = (self.panel_x + (pw - btn_w) * 0.5).round();
        let mut btn_y = (self.panel_y + ph * MODAL_BUTTONS_Y_OFFSET_RATIO).round();

        for btn in &mut self.buttons {
            btn.x = btn_x;
            btn.y = btn_y;
            btn.w = btn_w;
            btn.h = btn_h;
            btn_y += (btn_h + btn_gap).round();
        }
    }

    /// Update hover states from the current cursor position (pixel space).
    pub fn update_hover(&mut self, cursor_px: f32, cursor_py: f32) {
        for btn in &mut self.buttons {
            btn.hovered = btn.contains_px(cursor_px, cursor_py);
        }
    }

    /// Returns the label of the first button that contains (px, py), or None.
    pub fn hit_button(&self, px: f32, py: f32) -> Option<&'static str> {
        self.buttons.iter()
            .find(|b| b.contains_px(px, py))
            .map(|b| b.label)
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Vertex builders
    // ─────────────────────────────────────────────────────────────────────

    /// Sand-textured panel background (ModalVertex, uses modal_sand_pipeline).
    pub fn build_sand_vertices(&self, screen_w: f32, screen_h: f32) -> Vec<ModalVertex> {
        let x0 = self.panel_x;
        let y0 = self.panel_y;
        let x1 = x0 + self.panel_w;
        let y1 = y0 + self.panel_h;

        // UV: exceed 1.0 to tile the sand texture across the panel
        let u1 = self.panel_w / SAND_TILE_PX;
        let v1 = self.panel_h / SAND_TILE_PX;

        let cx0 = px_to_clip_x(x0, screen_w);
        let cx1 = px_to_clip_x(x1, screen_w);
        let cy0 = px_to_clip_y(y0, screen_h);
        let cy1 = px_to_clip_y(y1, screen_h);

        vec![
            ModalVertex { position: [cx0, cy1], tex_coords: [0.0, v1] },
            ModalVertex { position: [cx1, cy1], tex_coords: [u1,  v1] },
            ModalVertex { position: [cx1, cy0], tex_coords: [u1,  0.0] },
            ModalVertex { position: [cx0, cy1], tex_coords: [0.0, v1] },
            ModalVertex { position: [cx1, cy0], tex_coords: [u1,  0.0] },
            ModalVertex { position: [cx0, cy0], tex_coords: [0.0, 0.0] },
        ]
    }

    /// All solid-color UI geometry: border, bevel, buttons, title text (UiVertex).
    pub fn build_ui_vertices(&self, screen_w: f32, screen_h: f32) -> Vec<UiVertex> {
        let mut out: Vec<UiVertex> = Vec::new();
        self.push_border(&mut out, screen_w, screen_h);
        self.push_bevel(&mut out, screen_w, screen_h);
        self.push_buttons(&mut out, screen_w, screen_h);
        self.push_title(&mut out, screen_w, screen_h);
        out
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn push_border(&self, out: &mut Vec<UiVertex>, sw: f32, sh: f32) {
        let b   = self.border_w;
        let ox0 = self.panel_x - b;
        let oy0 = self.panel_y - b;
        let ox1 = self.panel_x + self.panel_w + b;
        let oy1 = self.panel_y + self.panel_h + b;
        let ix0 = self.panel_x;
        let iy0 = self.panel_y;
        let ix1 = self.panel_x + self.panel_w;
        let iy1 = self.panel_y + self.panel_h;
        push_border_rect(out, ox0, oy0, ox1, oy1, ix0, iy0, ix1, iy1,
                         MODAL_BORDER_COLOR, sw, sh);
    }

    fn push_bevel(&self, out: &mut Vec<UiVertex>, sw: f32, sh: f32) {
        let bw  = self.bevel_w;
        let px0 = self.panel_x;
        let py0 = self.panel_y;
        let px1 = self.panel_x + self.panel_w;
        let py1 = self.panel_y + self.panel_h;

        // Lit edges (top + left)
        push_rect_px(out, px0,       py0,       px1 - bw, py0 + bw, MODAL_BEVEL_LIGHT_COLOR,  sw, sh);
        push_rect_px(out, px0,       py0 + bw,  px0 + bw, py1,      MODAL_BEVEL_LIGHT_COLOR,  sw, sh);
        // Shadow edges (bottom + right)
        push_rect_px(out, px0 + bw,  py1 - bw,  px1,      py1,      MODAL_BEVEL_SHADOW_COLOR, sw, sh);
        push_rect_px(out, px1 - bw,  py0,        px1,     py1 - bw, MODAL_BEVEL_SHADOW_COLOR, sw, sh);
    }

    fn push_buttons(&self, out: &mut Vec<UiVertex>, sw: f32, sh: f32) {
        let bw  = self.btn_border_w;
        let bvw = self.btn_bevel_w;
        let s   = self.btn_text_scale;

        for btn in &self.buttons {
            let x0 = btn.x;
            let y0 = btn.y;
            let x1 = btn.x + btn.w;
            let y1 = btn.y + btn.h;

            // Outer border
            push_border_rect(out, x0 - bw, y0 - bw, x1 + bw, y1 + bw,
                             x0, y0, x1, y1, MODAL_BUTTON_BORDER_COLOR, sw, sh);

            // Fill
            let fill = if btn.hovered { MODAL_BUTTON_HOVER_COLOR } else { MODAL_BUTTON_BG_COLOR };
            push_rect_px(out, x0, y0, x1, y1, fill, sw, sh);

            // Bevel (top + left only)
            push_rect_px(out, x0,       y0,       x1 - bvw, y0 + bvw, MODAL_BUTTON_BEVEL_COLOR, sw, sh);
            push_rect_px(out, x0,       y0 + bvw, x0 + bvw, y1,       MODAL_BUTTON_BEVEL_COLOR, sw, sh);

            // Centered label text
            let char_w     = (5.0 + 1.0) * s;
            let label_px_w = btn.label.len() as f32 * char_w;
            let tx = (x0 + (btn.w - label_px_w) * 0.5).round();
            let ty = (y0 + (btn.h - 7.0 * s) * 0.5).round();
            let so = self.shadow_offset;

            bitmap_font::draw_text_quads(out, btn.label,
                tx + so, ty + so, s, s, MODAL_BUTTON_SHADOW_COLOR, sw, sh);
            bitmap_font::draw_text_quads(out, btn.label,
                tx,      ty,      s, s, MODAL_BUTTON_TEXT_COLOR,   sw, sh);
        }
    }

    fn push_title(&self, out: &mut Vec<UiVertex>, sw: f32, sh: f32) {
        let s          = self.title_scale;
        let char_w     = (5.0 + 1.0) * s;
        let title_px_w = self.title.len() as f32 * char_w;
        let tx = (self.panel_x + (self.panel_w - title_px_w) * 0.5).round();
        let ty = (self.panel_y + self.title_y_offset).round();
        let so = self.shadow_offset;

        bitmap_font::draw_text_quads(out, self.title,
            tx + so, ty + so, s, s, MODAL_TITLE_SHADOW, sw, sh);
        bitmap_font::draw_text_quads(out, self.title,
            tx,      ty,      s, s, MODAL_TITLE_COLOR,  sw, sh);
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Coordinate helpers (pixel → clip)
// ─────────────────────────────────────────────────────────────────────────

pub fn px_to_clip_x(px: f32, screen_w: f32) -> f32 {
    (px / screen_w) * 2.0 - 1.0
}

pub fn px_to_clip_y(py: f32, screen_h: f32) -> f32 {
    1.0 - (py / screen_h) * 2.0
}

/// Push a solid-color axis-aligned quad (6 vertices / 2 triangles).
pub fn push_rect_px(
    out: &mut Vec<UiVertex>,
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    color: [f32; 4],
    sw: f32, sh: f32,
) {
    let cx0 = px_to_clip_x(x0, sw);
    let cx1 = px_to_clip_x(x1, sw);
    let cy0 = px_to_clip_y(y0, sh);
    let cy1 = px_to_clip_y(y1, sh);

    out.push(UiVertex { position: [cx0, cy1], color });
    out.push(UiVertex { position: [cx1, cy1], color });
    out.push(UiVertex { position: [cx1, cy0], color });
    out.push(UiVertex { position: [cx0, cy1], color });
    out.push(UiVertex { position: [cx1, cy0], color });
    out.push(UiVertex { position: [cx0, cy0], color });
}

/// Push a hollow border rectangle (outer minus inner) as 4 edge strips.
fn push_border_rect(
    out: &mut Vec<UiVertex>,
    ox0: f32, oy0: f32, ox1: f32, oy1: f32,
    ix0: f32, iy0: f32, ix1: f32, iy1: f32,
    color: [f32; 4],
    sw: f32, sh: f32,
) {
    push_rect_px(out, ox0, oy0, ox1, iy0, color, sw, sh); // top
    push_rect_px(out, ox0, iy1, ox1, oy1, color, sw, sh); // bottom
    push_rect_px(out, ox0, iy0, ix0, iy1, color, sw, sh); // left
    push_rect_px(out, ix1, iy0, ox1, iy1, color, sw, sh); // right
}
