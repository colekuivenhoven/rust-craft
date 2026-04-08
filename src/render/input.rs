use super::*;
use super::crafting_ui::*;

impl State {
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW => {
                        self.camera_controller.forward = is_pressed;
                        true
                    }
                    KeyCode::KeyS => {
                        self.camera_controller.backward = is_pressed;
                        true
                    }
                    KeyCode::KeyA => {
                        self.camera_controller.left = is_pressed;
                        true
                    }
                    KeyCode::KeyD => {
                        self.camera_controller.right = is_pressed;
                        true
                    }
                    KeyCode::Space => {
                        self.camera_controller.jump_held = is_pressed;
                        true
                    }
                    KeyCode::ShiftLeft => {
                        self.camera_controller.shift_held = is_pressed;
                        true
                    }
                    KeyCode::KeyE => {
                        if is_pressed {
                            self.show_inventory = !self.show_inventory;
                            println!("=== INVENTORY ===");
                            for (i, slot) in self.player.inventory.slots.iter().enumerate() {
                                if let Some(stack) = slot {
                                    let marker = if i == self.player.inventory.selected_slot {
                                        ">"
                                    } else {
                                        " "
                                    };
                                    println!(
                                        "{} Slot {}: {:?} x{}",
                                        marker, i, stack.block_type, stack.count
                                    );
                                }
                            }
                        }
                        true
                    }
                    KeyCode::Digit1
                    | KeyCode::Digit2
                    | KeyCode::Digit3
                    | KeyCode::Digit4
                    | KeyCode::Digit5
                    | KeyCode::Digit6
                    | KeyCode::Digit7
                    | KeyCode::Digit8
                    | KeyCode::Digit9 => {
                        if is_pressed && !self.crafting_ui_open {
                            let num = match keycode {
                                KeyCode::Digit1 => 0,
                                KeyCode::Digit2 => 1,
                                KeyCode::Digit3 => 2,
                                KeyCode::Digit4 => 3,
                                KeyCode::Digit5 => 4,
                                KeyCode::Digit6 => 5,
                                KeyCode::Digit7 => 6,
                                KeyCode::Digit8 => 7,
                                KeyCode::Digit9 => 8,
                                _ => 0,
                            };
                            self.player.inventory.selected_slot = num;
                        }
                        true
                    }
                    KeyCode::F1 => {
                        if is_pressed {
                            self.show_chunk_outlines = !self.show_chunk_outlines;
                            println!("Chunk outlines: {}", if self.show_chunk_outlines { "ON" } else { "OFF" });
                        }
                        true
                    }
                    KeyCode::F2 => {
                        if is_pressed {
                            self.noclip_mode = !self.noclip_mode;
                            println!("Noclip: {}", if self.noclip_mode { "ON" } else { "OFF" });
                        }
                        true
                    }
                    KeyCode::F3 => {
                        if is_pressed {
                            self.show_enemy_hitboxes = !self.show_enemy_hitboxes;
                            println!("Enemy hitboxes: {}", if self.show_enemy_hitboxes { "ON" } else { "OFF" });
                        }
                        true
                    }
                    KeyCode::F4 => {
                        if is_pressed {
                            self.smooth_lighting = !self.smooth_lighting;
                            println!("Smooth lighting: {}", if self.smooth_lighting { "ON" } else { "OFF" });
                            // Mark all loaded chunks dirty so they get re-meshed with the new setting
                            for chunk in self.world.chunks.values_mut() {
                                chunk.dirty = true;
                            }
                        }
                        true
                    }
                    KeyCode::F5 => {
                        if is_pressed {
                            self.hud_enabled = !self.hud_enabled;
                            println!("HUD: {}", if self.hud_enabled { "ON" } else { "OFF" });
                        }
                        true
                    }
                    _ => false,
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                if self.crafting_ui_open && is_pressed {
                    let (px, py) = self.cursor_pos_px;
                    let sw = self.size.width as f32;
                    let sh = self.size.height as f32;
                    let hit = self.crafting_slot_hit(px, py, sw, sh);
                    match hit {
                        CraftingHit::GridSlot(row, col) => {
                            if let Some(held) = self.crafting_held.take() {
                                let existing = self.crafting_grid.slots[row][col].take();
                                self.crafting_grid.slots[row][col] = Some(held);
                                if let Some(old) = existing {
                                    self.crafting_held = Some(old);
                                }
                            } else if let Some(item) = self.crafting_grid.slots[row][col].take() {
                                self.crafting_held = Some(item);
                            }
                            self.crafting_output = match_recipe(&self.crafting_grid);
                        }
                        CraftingHit::OutputSlot => {
                            if self.crafting_held.is_none() {
                                if let Some((bt, qty)) = self.crafting_output.take() {
                                    self.consume_crafting_inputs();
                                    self.player.inventory.add_item(bt, qty);
                                    self.crafting_output = match_recipe(&self.crafting_grid);
                                }
                            }
                        }
                        CraftingHit::InvSlot(i) => {
                            if let Some(held) = self.crafting_held.take() {
                                self.player.inventory.add_item(held.0, held.1);
                                self.crafting_inv_qty[i] = if self.player.inventory.get_slot(i).is_some() { 1.0 } else { 0.0 };
                            } else {
                                let pick = self.player.inventory.get_slot(i).map(|s| {
                                    (s.block_type, self.crafting_inv_qty[i].min(s.count))
                                });
                                if let Some((bt, qty)) = pick {
                                    if qty > 0.0 {
                                        self.player.inventory.remove_item(i, qty);
                                        self.crafting_held = Some((bt, qty));
                                        self.crafting_inv_qty[i] = if self.player.inventory.get_slot(i).is_some() { 1.0 } else { 0.0 };
                                    }
                                }
                            }
                        }
                        CraftingHit::None => {
                            if let Some((bt, qty)) = self.crafting_held.take() {
                                self.player.inventory.add_item(bt, qty);
                            }
                        }
                    }
                } else {
                    self.mouse_pressed = is_pressed;
                    self.left_mouse_held = is_pressed;
                    if !is_pressed {
                        self.breaking_state = None;
                    }
                }
                true
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                if self.mouse_captured {
                    // Priority: pick up a hovered dropped item before placing a block
                    if let Some(idx) = self.hovered_dropped_item {
                        if idx < self.dropped_item_manager.items.len() {
                            let collected = self.dropped_item_manager.collect_item(idx);
                            self.player.inventory.add_item(collected.block_type, collected.value);
                            self.hovered_dropped_item = None;
                        }
                    } else if self.hovered_crafting_table {
                        self.open_crafting_ui();
                    } else {
                        self.handle_block_place();
                    }
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_f = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y / 40.0) as f32,
                };
                if self.crafting_ui_open {
                    // Adjust selected quantity for hovered inventory slot
                    if let Some(i) = self.crafting_hovered_inv {
                        let max_qty = self.player.inventory
                            .get_slot(i).map(|s| s.count).unwrap_or(0.0);
                        if max_qty > 0.0 {
                            let step = 0.25f32;
                            let new_qty = (self.crafting_inv_qty[i] + scroll_f * step)
                                .clamp(step, max_qty.max(step));
                            // Round to nearest 0.25
                            self.crafting_inv_qty[i] = (new_qty / step).round() * step;
                        }
                    }
                } else if self.mouse_captured {
                    let scroll_amount = scroll_f as i32;
                    if scroll_amount != 0 {
                        let slots = self.player.inventory.size;
                        let current = self.player.inventory.selected_slot as i32;
                        let new_slot = (current - scroll_amount).rem_euclid(slots as i32) as usize;
                        self.player.inventory.selected_slot = new_slot;
                    }
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let px = position.x as f32;
                let py = position.y as f32;
                self.handle_modal_cursor_moved(px, py);
                if self.crafting_ui_open {
                    let sw = self.size.width as f32;
                    let sh = self.size.height as f32;
                    self.crafting_hovered_grid = None;
                    self.crafting_hovered_inv = None;
                    self.crafting_hovered_output = false;
                    match self.crafting_slot_hit(px, py, sw, sh) {
                        CraftingHit::GridSlot(r, c) => self.crafting_hovered_grid = Some((r, c)),
                        CraftingHit::InvSlot(i) => {
                            self.crafting_hovered_inv = Some(i);
                            // Initialise qty on first hover to 1.0
                            if self.crafting_inv_qty[i] == 0.0 {
                                if self.player.inventory.get_slot(i).is_some() {
                                    self.crafting_inv_qty[i] = 1.0;
                                }
                            }
                        }
                        CraftingHit::OutputSlot => self.crafting_hovered_output = true,
                        CraftingHit::None => {}
                    }
                }
                false // don't consume — other handlers may need it
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, dx: f64, dy: f64) {
        if self.mouse_captured {
            self.camera_controller.process_mouse(dx as f32, dy as f32);
        }
    }
}
