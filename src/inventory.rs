use crate::block::BlockType;

#[derive(Debug, Clone)]
pub struct ItemStack {
    pub block_type: BlockType,
    pub count: u32,
}

impl ItemStack {
    pub fn new(block_type: BlockType, count: u32) -> Self {
        Self { block_type, count }
    }
}

pub struct Inventory {
    pub slots: Vec<Option<ItemStack>>,
    pub selected_slot: usize,
    pub size: usize,
}

impl Inventory {
    pub fn new(size: usize) -> Self {
        Self {
            slots: vec![None; size],
            selected_slot: 0,
            size,
        }
    }

    pub fn add_item(&mut self, block_type: BlockType, count: u32) -> bool {
        // Try to stack with existing items
        for slot in &mut self.slots {
            if let Some(stack) = slot {
                if stack.block_type == block_type && stack.count < 64 {
                    let add_count = (64 - stack.count).min(count);
                    stack.count += add_count;
                    if add_count == count {
                        return true;
                    }
                }
            }
        }

        // Find empty slot
        for slot in &mut self.slots {
            if slot.is_none() {
                *slot = Some(ItemStack::new(block_type, count));
                return true;
            }
        }

        false
    }

    pub fn remove_item(&mut self, slot_index: usize, count: u32) -> Option<ItemStack> {
        if slot_index >= self.size {
            return None;
        }

        if let Some(stack) = &mut self.slots[slot_index] {
            if stack.count <= count {
                self.slots[slot_index].take()
            } else {
                stack.count -= count;
                Some(ItemStack::new(stack.block_type, count))
            }
        } else {
            None
        }
    }

    pub fn get_selected_item(&self) -> Option<&ItemStack> {
        self.slots[self.selected_slot].as_ref()
    }

    pub fn select_next(&mut self) {
        self.selected_slot = (self.selected_slot + 1) % self.size;
    }

    pub fn select_previous(&mut self) {
        self.selected_slot = if self.selected_slot == 0 {
            self.size - 1
        } else {
            self.selected_slot - 1
        };
    }

    pub fn get_slot(&self, index: usize) -> Option<&ItemStack> {
        if index < self.size {
            self.slots[index].as_ref()
        } else {
            None
        }
    }
}
