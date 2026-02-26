use crate::block::BlockType;

pub type GridSlot = Option<(BlockType, f32)>;

#[derive(Clone)]
pub struct CraftingGrid {
    pub slots: [[GridSlot; 3]; 3], // [row][col]
}

impl Default for CraftingGrid {
    fn default() -> Self {
        Self {
            slots: [[None; 3]; 3],
        }
    }
}

/// Returns the crafted output `(BlockType, qty)` if the grid matches a recipe, else `None`.
pub fn match_recipe(grid: &CraftingGrid) -> Option<(BlockType, f32)> {
    // Collect non-empty cells as (row, col, block_type, qty)
    let filled: Vec<(usize, usize, BlockType, f32)> = (0..3)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .filter_map(|(r, c)| grid.slots[r][c].map(|(bt, qty)| (r, c, bt, qty)))
        .collect();

    // ── Recipe 1: single Wood slot → Planks × 4 (proportional) ──────────
    if filled.len() == 1 {
        let (_, _, bt, qty) = filled[0];
        if bt == BlockType::Wood {
            return Some((BlockType::Planks, qty * 4.0));
        }
    }

    // ── Recipe 2: 2×2 of Planks (1.0 each) anywhere in grid → CraftingTable ──
    if filled.len() == 4 {
        // Try all four possible top-left corners of a 2×2 inside a 3×3
        for (tr, tc) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
            let corners = [
                (tr,     tc),
                (tr,     tc + 1),
                (tr + 1, tc),
                (tr + 1, tc + 1),
            ];
            let all_match = corners.iter().all(|&(r, c)| {
                matches!(grid.slots[r][c], Some((BlockType::Planks, qty)) if (qty - 1.0).abs() < 0.001)
            });
            // Also verify the 4 filled cells are exactly these 4 corners
            let exact = all_match && filled.iter().all(|&(fr, fc, _, _)| {
                corners.contains(&(fr, fc))
            });
            if exact {
                return Some((BlockType::CraftingTable, 1.0));
            }
        }
    }

    None
}
