use super::*;

/// Result of a crafting-UI pixel hit-test
#[derive(Debug, Clone, Copy)]
pub(super) enum CraftingHit {
    GridSlot(usize, usize),
    OutputSlot,
    InvSlot(usize),
    None,
}

/// All pixel-space coordinates needed to draw / hit-test the crafting UI.
/// Derived from the crafting modal's panel bounds each frame.
pub(super) struct CraftingLayout {
    pub ct_slot:  f32,
    pub ct_gap:   f32,
    pub grid_w:   f32,
    pub grid_h:   f32,
    pub row1_x:   f32,
    pub row1_y:   f32,
    pub row2_x:   f32,
    pub row2_y:   f32,
    pub out_x:    f32,
    pub out_y:    f32,
    pub arrow_x:  f32,
    pub arrow_y:  f32,
    pub arrow_scale: f32,
}

/// Compute the crafting UI pixel layout from the modal's panel bounds.
pub(super) fn crafting_layout(panel_x: f32, panel_y: f32, panel_w: f32, panel_h: f32) -> CraftingLayout {
    use crate::modal::{MODAL_BORDER_PX, MODAL_BEVEL_PX};
    let bevel_pad    = MODAL_BORDER_PX + MODAL_BEVEL_PX; // 12
    let interior_pad = 8.0f32;
    let content_x    = panel_x + bevel_pad + interior_pad;
    let content_w    = panel_w - 2.0 * (bevel_pad + interior_pad);
    let content_bottom_y = panel_y + panel_h - bevel_pad - interior_pad;

    let ct_gap  = 8.0f32;
    // Slot size chosen so 9 slots + 8 gaps exactly fill the content width
    let ct_slot = ((content_w - 8.0 * ct_gap) / 9.0) * 0.75; // 0.75 scale factor

    // Row 1 horizontal layout (centred in content area)
    let arrow_gap   = (content_w * 0.016).max(10.0);
    let arrow_w     = (ct_slot * 0.40).max(24.0);
    let grid_w      = 3.0 * ct_slot + 2.0 * ct_gap;
    let grid_h      = grid_w;
    let row1_w      = grid_w + arrow_gap + arrow_w + arrow_gap + ct_slot;
    let row1_x      = content_x + (content_w - row1_w) * 0.5;

    // Row 1 vertical start: below the crafting modal's title (fixed 20px offset + title height)
    let title_scale  = 10.5f32; // The scale of the AREA used by the title. Naming should probably be updated here!
    let title_h      = 7.0 * title_scale;
    let title_offset = bevel_pad + interior_pad; // align title to top of content area
    let title_end_y  = panel_y + title_offset + title_h;
    let row1_y       = title_end_y + panel_h * 0.12;

    // Row 2 (inventory): anchored to bottom of content area, horizontally centered
    let row2_y = content_bottom_y - ct_slot;
    let inv_w  = 9.0 * ct_slot + 8.0 * ct_gap;
    let row2_x = content_x + (content_w - inv_w) * 0.5;

    // Output slot (centred vertically in grid area)
    let out_x = row1_x + grid_w + arrow_gap + arrow_w + arrow_gap;
    let out_y = row1_y + (grid_h - ct_slot) * 0.5;

    // Arrow text position (centred vertically in grid area, between grid and output)
    let arrow_scale = 7.5f32;
    let arrow_char_h = 7.0 * arrow_scale;
    let arrow_x = row1_x + grid_w + arrow_gap;
    let arrow_y = row1_y + (grid_h - arrow_char_h) * 0.5;

    CraftingLayout { ct_slot, ct_gap, grid_w, grid_h, row1_x, row1_y,
        row2_x, row2_y, out_x, out_y, arrow_x, arrow_y, arrow_scale }
}

impl State {
    pub fn open_crafting_ui(&mut self) {
        let sw = self.size.width as f32;
        let sh = self.size.height as f32;
        self.crafting_modal.update_layout(sw, sh);
        self.crafting_modal.title_scale = 4.0;
        self.crafting_modal.title_y_offset = (crate::modal::MODAL_BORDER_PX + crate::modal::MODAL_BEVEL_PX) + 8.0;
        self.crafting_ui_open = true;
        self.crafting_grid = CraftingGrid::default();
        self.crafting_held = None;
        self.crafting_output = None;
        // Initialise per-slot selected quantities to 1.0 (or 0 if slot empty)
        for i in 0..9 {
            self.crafting_inv_qty[i] = self.player.inventory
                .get_slot(i)
                .map(|_| 1.0)
                .unwrap_or(0.0);
        }
        self.release_mouse();
    }

    pub fn close_crafting_ui(&mut self) {
        // Return held item to inventory
        if let Some((bt, qty)) = self.crafting_held.take() {
            self.player.inventory.add_item(bt, qty);
        }
        // Return every grid item to inventory
        for row in 0..3 {
            for col in 0..3 {
                if let Some((bt, qty)) = self.crafting_grid.slots[row][col].take() {
                    self.player.inventory.add_item(bt, qty);
                }
            }
        }
        self.crafting_ui_open = false;
        self.crafting_output = None;
        self.capture_mouse();
    }

    /// Returns which crafting-UI element the pixel coordinate (px, py) is over.
    pub(super) fn crafting_slot_hit(&self, px: f32, py: f32, _sw: f32, _sh: f32) -> CraftingHit {
        let lo = crafting_layout(
            self.crafting_modal.panel_x, self.crafting_modal.panel_y,
            self.crafting_modal.panel_w, self.crafting_modal.panel_h,
        );
        // 3x3 grid
        for row in 0..3usize {
            for col in 0..3usize {
                let sx = lo.row1_x + col as f32 * (lo.ct_slot + lo.ct_gap);
                let sy = lo.row1_y + row as f32 * (lo.ct_slot + lo.ct_gap);
                if px >= sx && px < sx + lo.ct_slot && py >= sy && py < sy + lo.ct_slot {
                    return CraftingHit::GridSlot(row, col);
                }
            }
        }
        // Output slot
        if px >= lo.out_x && px < lo.out_x + lo.ct_slot
            && py >= lo.out_y && py < lo.out_y + lo.ct_slot {
            return CraftingHit::OutputSlot;
        }
        // Inventory row
        for i in 0..9usize {
            let sx = lo.row2_x + i as f32 * (lo.ct_slot + lo.ct_gap);
            let sy = lo.row2_y;
            if px >= sx && px < sx + lo.ct_slot && py >= sy && py < sy + lo.ct_slot {
                return CraftingHit::InvSlot(i);
            }
        }
        CraftingHit::None
    }

    /// Consume crafting grid inputs for one craft.
    /// For Recipe 1 (Wood -> Planks): clears the single occupied grid slot.
    /// For Recipe 2 (2x2 Planks -> CraftingTable): clears the four 2x2 slots.
    pub(super) fn consume_crafting_inputs(&mut self) {
        // Simply clear all non-empty grid slots (recipes use exact slot contents)
        for row in 0..3 {
            for col in 0..3 {
                self.crafting_grid.slots[row][col] = None;
            }
        }
    }
}
